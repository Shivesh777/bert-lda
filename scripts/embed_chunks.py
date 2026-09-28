"""Encode the passages in data/chunks.parquet with the frozen sentence
transformer (config.ENCODER). Embeddings are unit-normalised float32.

Blocks of 20,000 passages are saved to data/emb_ckpt/ as they finish, so an
interrupted run resumes. The blocks are then merged into
data/chunk_embeddings.npz (arrays emb, doc_index, chunk_w).

Usage: python scripts/embed_chunks.py
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

import config

DATA = ROOT / "data"
CKPT = DATA / "emb_ckpt"
BLOCK = 20000


def main():
    from sentence_transformers import SentenceTransformer

    CKPT.mkdir(exist_ok=True)
    table = pq.read_table(DATA / "chunks.parquet")
    chunks = table.column("chunk").to_pylist()
    n = len(chunks)
    print(f"{n:,} passages, encoder {config.ENCODER}")
    model = SentenceTransformer(config.ENCODER, device=None)

    t0 = time.time()
    for b0 in range(0, n, BLOCK):
        path = CKPT / f"block_{b0:07d}.npy"
        if path.exists():
            continue
        emb = model.encode(chunks[b0:b0 + BLOCK], batch_size=128,
                           show_progress_bar=False, convert_to_numpy=True,
                           normalize_embeddings=True).astype(np.float32)
        np.save(path, emb)
        done = min(b0 + BLOCK, n)
        rate = done / max(time.time() - t0, 1)
        print(f"  {done:,}/{n:,} ({rate:.0f}/s, {(n - done) / rate / 60:.0f} min left)",
              flush=True)

    blocks = [np.load(f) for f in sorted(CKPT.glob("block_*.npy"))]
    emb = np.vstack(blocks)
    assert emb.shape[0] == n, "embedding count does not match passage count"
    np.savez_compressed(DATA / "chunk_embeddings.npz", emb=emb,
                        doc_index=table.column("doc_index").to_numpy(),
                        chunk_w=table.column("weight").to_numpy())
    print(f"saved {emb.shape} to data/chunk_embeddings.npz [{(time.time() - t0) / 60:.0f} min]")


if __name__ == "__main__":
    main()
