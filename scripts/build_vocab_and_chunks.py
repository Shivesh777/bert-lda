"""Build the shared vocabulary, the article-term count matrix and the
transformer passages from data/corpus.parquet.

Outputs in data/:
  chunks.parquet    leading passages per article (chunk, doc_index, weight)
  vocab.json        VOCAB_SIZE unigrams and bigrams, learned on a 60,000
                    article sample (seed 0), min_df 10, max_df 0.4
  counts.npz        sparse article x term counts over the fixed vocabulary
  doc_meta.parquet  article date and term count (used for daily aggregation)

Both topic-model branches read counts.npz and vocab.json, so their term
distributions and coherence scores are defined on the same terms.
"""
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import config
from src.textprep import STOPWORDS, clean_text
from src.stage1_bert import chunk_articles

DATA = ROOT / "data"
CORPUS = DATA / "corpus.parquet"


def main():
    t0 = time.time()
    df = pq.read_table(CORPUS).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    texts = df["Article_title"].fillna("") + ". " + df["Article"].fillna("")
    dates = df["date"]
    del df
    gc.collect()
    print(f"{len(texts):,} articles [{time.time() - t0:.0f}s]")

    # Transformer passages: paragraph-based chunks of at most 200 words,
    # keeping the first MAX_CHUNKS_PER_DOC eligible chunks of each article.
    chunks, doc_index, chunk_w = chunk_articles(texts.tolist(), max_words=200)
    keep = np.zeros(len(doc_index), dtype=bool)
    seen = {}
    for i, d in enumerate(doc_index):
        seen[d] = seen.get(d, 0) + 1
        keep[i] = seen[d] <= config.MAX_CHUNKS_PER_DOC
    chunks = [c for c, k in zip(chunks, keep) if k]
    doc_index, chunk_w = doc_index[keep], chunk_w[keep]
    pq.write_table(pa.table({"chunk": pa.array(chunks),
                             "doc_index": pa.array(doc_index.tolist()),
                             "weight": pa.array(chunk_w.tolist())}),
                   DATA / "chunks.parquet", compression="zstd")
    print(f"{len(chunks):,} chunks saved [{time.time() - t0:.0f}s]")
    del chunks, doc_index, chunk_w, keep, seen
    gc.collect()

    # Vocabulary: learned on a random sample (a bigram dictionary over the
    # full corpus does not fit in memory), then applied to every article.
    rng = np.random.default_rng(config.SEED)
    sample = rng.choice(len(texts), size=min(60000, len(texts)), replace=False)
    learner = CountVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS,
                              max_features=config.VOCAB_SIZE, min_df=10,
                              max_df=0.4, token_pattern=r"[a-z]{3,}")
    learner.fit(clean_text(texts.iloc[sample]))
    vocab = learner.get_feature_names_out().tolist()
    print(f"vocabulary learned [{time.time() - t0:.0f}s]")

    counter = CountVectorizer(ngram_range=(1, 2), vocabulary=vocab,
                              token_pattern=r"[a-z]{3,}")
    parts = []
    batch = 20000
    for b0 in range(0, len(texts), batch):
        block = clean_text(texts.iloc[b0:b0 + batch])
        parts.append(counter.transform(block).astype(np.float32))
        print(f"  counts {b0 + len(block):,}/{len(texts):,} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    counts = sparse.vstack(parts).tocsr()
    n_tokens = np.asarray(counts.sum(axis=1)).ravel()
    del parts, texts
    gc.collect()

    sparse.save_npz(DATA / "counts.npz", counts)
    json.dump(vocab, open(DATA / "vocab.json", "w"))
    pq.write_table(pa.table({"date": pa.array(dates),
                             "n_tokens": pa.array(n_tokens.tolist())}),
                   DATA / "doc_meta.parquet")
    print(f"vocab {len(vocab):,}, nonzeros {counts.nnz:,} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
