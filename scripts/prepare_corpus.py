"""Consolidate the downloaded shards into the article corpus.

Reads data/fnspid_raw/*.parquet shard by shard, drops bodies shorter than
MIN_CHARS, removes duplicates (same URL, or same date and title) and keeps at
most MAX_PER_DAY articles per calendar day. The per-day subsample is chosen
by a seeded hash of the URL, so the result does not depend on shard order.

Output: data/corpus.parquet with columns date, Article_title, Stock_symbol,
Article.
"""
from pathlib import Path
import hashlib
import heapq
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

import config

RAW = ROOT / "data" / "fnspid_raw"
OUT = ROOT / "data" / "corpus.parquet"
HASH_SEED = b"corpus-v1"
COLUMNS = ["date", "Article_title", "Stock_symbol", "Url", "Article"]


def hkey(s):
    return hashlib.blake2b(HASH_SEED + s.encode("utf-8", "ignore"),
                           digest_size=8).digest()


def main():
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no shards in {RAW}; run scripts/download_fnspid.py first")
    print(f"{len(files)} shards")
    seen_url, seen_date_title = set(), set()
    day_reservoir = defaultdict(list)     # date -> heap of (inverted key, row)
    n_raw = n_short = n_dup = 0
    for fi, f in enumerate(files):
        table = pq.read_table(f, columns=COLUMNS)
        cols = [table.column(c).to_pylist() for c in COLUMNS]
        for date, title, sym, url, art in zip(*cols):
            n_raw += 1
            if art is None or len(art) < config.MIN_CHARS:
                n_short += 1
                continue
            hu = hkey(url or "")
            hd = hkey((date or "") + "|" + (title or ""))
            if hu in seen_url or hd in seen_date_title:
                n_dup += 1
                continue
            seen_url.add(hu)
            seen_date_title.add(hd)
            key = hkey((url or "") + "#s")
            item = (bytes(255 - b for b in key), (date, title, sym, art))
            heap = day_reservoir[date]
            if len(heap) < config.MAX_PER_DAY:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)   # keep the MAX_PER_DAY smallest keys
        if fi % 20 == 0:
            print(f"  shard {fi}/{len(files)} raw={n_raw:,} short={n_short:,} "
                  f"dup={n_dup:,} days={len(day_reservoir):,}")

    rows = []
    for date in sorted(day_reservoir):
        rows.extend(item[1] for item in sorted(day_reservoir[date], reverse=True))
    print(f"raw={n_raw:,} short={n_short:,} dup={n_dup:,} kept={len(rows):,} "
          f"days={len(day_reservoir):,}")

    table = pa.table({
        "date": pa.array([r[0] for r in rows]),
        "Article_title": pa.array([r[1] for r in rows]),
        "Stock_symbol": pa.array([r[2] for r in rows]),
        "Article": pa.array([r[3] for r in rows]),
    })
    pq.write_table(table, OUT, compression="zstd")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
