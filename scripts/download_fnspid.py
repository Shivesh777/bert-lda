"""Download and filter the FNSPID news file (nasdaq_exteral_data.csv, 23 GB).

The file is fetched from Hugging Face by N_WORKERS parallel HTTP range
requests. Each worker streams its byte slice through pandas.read_csv, keeps
rows dated within [DATE_MIN, DATE_MAX] with a non-empty article body and a
plausible ticker, truncates bodies to ARTICLE_TRUNC characters and writes
parquet shards to data/fnspid_raw/. Workers other than the first realign to
the next record boundary. At the end of a slice the parser stops inside the
straddling record, and pandas discards the chunk being parsed at that point
(up to 20,000 raw rows, before the date filter), so the rows kept depend on
N_WORKERS: keep config.N_WORKERS = 6 to rebuild the same corpus.

Each worker records the bytes it consumed in data/fnspid_status/. The run
fails if any worker read less than its slice, which guards against a
connection drop being mistaken for the end of the slice.

Usage: python scripts/download_fnspid.py            (all workers)
       python scripts/download_fnspid.py 2 3        (rerun selected workers)
"""
from pathlib import Path
import io
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

import config

URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
       "Stock_news/nasdaq_exteral_data.csv")
TOTAL_SIZE = 23232979597          # bytes; checked against the server at start
OUT_DIR = ROOT / "data" / "fnspid_raw"
STATUS_DIR = ROOT / "data" / "fnspid_status"
ARTICLE_TRUNC = 4000
HEADER = ("Unnamed: 0,Date,Article_title,Stock_symbol,Url,Publisher,Author,"
          "Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary\n")
USECOLS = ["Date", "Article_title", "Stock_symbol", "Url", "Article"]
# A record starts with an optional numeric index and an ISO timestamp.
MARKER = re.compile(rb"\n(?:\d+(?:\.\d*)?)?,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,")
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,7}$")


class SliceStream(io.RawIOBase):
    """File-like object over one HTTP byte range, realigned to a record start."""

    def __init__(self, start, end, skip_to_marker):
        self.start, self.end = start, end
        self.skip_to_marker = skip_to_marker
        self.prepend = HEADER.encode() if skip_to_marker else b""
        self.buf = b""
        self.consumed = 0
        headers = {"Range": f"bytes={start}-{end}"}
        self.resp = requests.get(URL, headers=headers, stream=True, timeout=120)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(chunk_size=1 << 20)
        if skip_to_marker:
            window = b""
            while True:
                chunk = next(self.it, None)
                if chunk is None:
                    return
                self.consumed += len(chunk)
                window += chunk
                m = MARKER.search(window)
                if m:
                    self.buf = window[m.start() + 1:]
                    return
                window = window[-64:]

    def readable(self):
        return True

    def readinto(self, b):
        need = len(b)
        out = b""
        if self.prepend:
            out, self.prepend = self.prepend, b""
        while len(out) < need:
            if self.buf:
                take = min(need - len(out), len(self.buf))
                out += self.buf[:take]
                self.buf = self.buf[take:]
                continue
            chunk = next(self.it, None)
            if chunk is None:
                break
            self.consumed += len(chunk)
            self.buf = chunk
        b[:len(out)] = out
        return len(out)


def filter_chunk(chunk):
    d = chunk.dropna(subset=["Date", "Article", "Stock_symbol"])
    if not len(d):
        return d
    dates = d["Date"].str.slice(0, 10)
    d = d[dates.str.match(r"\d{4}-\d{2}-\d{2}$", na=False)]
    dates = d["Date"].str.slice(0, 10)
    in_window = (dates >= config.DATE_MIN) & (dates <= config.DATE_MAX)
    d = d[in_window & d["Stock_symbol"].str.match(SYMBOL_RE, na=False)]
    if not len(d):
        return d
    return d.assign(
        date=d["Date"].str.slice(0, 10),
        Article=d["Article"].str.slice(0, ARTICLE_TRUNC),
    )[["date", "Article_title", "Stock_symbol", "Url", "Article"]]


def run_worker(widx):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    slice_size = TOTAL_SIZE // config.N_WORKERS
    start = widx * slice_size
    end = TOTAL_SIZE - 1 if widx == config.N_WORKERS - 1 else (widx + 1) * slice_size - 1
    status_path = STATUS_DIR / f"worker{widx}.json"

    for attempt in range(5):
        stream = None
        try:
            stream = SliceStream(start, end, skip_to_marker=(widx > 0))
            reader = pd.read_csv(
                io.BufferedReader(stream, buffer_size=1 << 20),
                chunksize=20000, usecols=USECOLS, dtype=str,
                on_bad_lines="skip", engine="c",
            )
            kept = seen = shard = 0
            t0 = time.time()
            for f in OUT_DIR.glob(f"w{widx}_s*.parquet"):
                f.unlink()
            while True:
                try:
                    chunk = next(reader)
                except StopIteration:
                    break
                except pd.errors.ParserError:
                    # Expected once per slice: the range ends inside a quoted
                    # field. Chunks returned so far are complete; the chunk
                    # in progress is discarded (see the module docstring).
                    break
                seen += len(chunk)
                d = filter_chunk(chunk)
                if len(d):
                    tbl = pa.table({c: pa.array(d[c].tolist(), type=pa.string())
                                    for c in d.columns})
                    pq.write_table(tbl, OUT_DIR / f"w{widx}_s{shard:04d}.parquet",
                                   compression="zstd")
                    shard += 1
                    kept += len(d)
            slice_bytes = end - start + 1
            complete = stream.consumed >= slice_bytes - (1 << 20)
            status = {"worker": widx, "rows_seen": seen, "rows_kept": kept,
                      "shards": shard, "slice_bytes": slice_bytes,
                      "consumed_bytes": stream.consumed,
                      "elapsed_s": round(time.time() - t0), "done": complete}
            status_path.write_text(json.dumps(status))
            if not complete:
                raise IOError(f"worker {widx} consumed {stream.consumed} of "
                              f"{slice_bytes} bytes")
            return
        except Exception:
            (STATUS_DIR / f"worker{widx}_err{attempt}.txt").write_text(
                traceback.format_exc())
            time.sleep(10)
        finally:
            if stream is not None:
                stream.resp.close()
    status_path.write_text(json.dumps({"worker": widx, "done": False,
                                       "failed": True}))


def check_remote_size():
    head = requests.head(URL, allow_redirects=True, timeout=60)
    size = int(head.headers.get("Content-Length", 0))
    if size and size != TOTAL_SIZE:
        raise SystemExit(f"remote file is {size} bytes, expected {TOTAL_SIZE}; "
                         "update TOTAL_SIZE before downloading")


if __name__ == "__main__":
    check_remote_size()
    ids = [int(a) for a in sys.argv[1:]] or list(range(config.N_WORKERS))
    with ProcessPoolExecutor(max_workers=len(ids)) as ex:
        list(ex.map(run_worker, ids))
    failed = [i for i in ids
              if not json.loads((STATUS_DIR / f"worker{i}.json").read_text()).get("done")]
    if failed:
        raise SystemExit(f"workers {failed} did not complete; rerun them: "
                         f"python scripts/download_fnspid.py {' '.join(map(str, failed))}")
    total = sum(json.loads((STATUS_DIR / f"worker{i}.json").read_text())["rows_kept"]
                for i in ids)
    print(f"download complete: {total:,} rows kept in {OUT_DIR}")
