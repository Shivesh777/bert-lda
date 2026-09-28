"""Topic coherence (NPMI, paper Eq. 9) for each fitted variant.

For every topic the ten highest-weighted terms are taken from
results/phi_{variant}.npy and NPMI is averaged over the 45 term pairs, using
document co-occurrence in data/counts.npz. Per-topic scores are written to
results/coherence.json (one list per variant); the script prints the mean.

Usage: python scripts/coherence.py [variant ...]
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse

import config
from src.experiment import atomic_json

TOP_N = 10
EPS = 1e-12


def npmi_scores(phi, presence, doc_freq, n_docs):
    scores = []
    for l in range(phi.shape[0]):
        top = np.argsort(phi[l])[::-1][:TOP_N]
        sub = presence[:, top].toarray()
        co = (sub.T @ sub) / n_docs
        pairs = []
        for i in range(TOP_N):
            for j in range(i + 1, TOP_N):
                pij = co[i, j] + EPS
                pi, pj = doc_freq[top[i]] + EPS, doc_freq[top[j]] + EPS
                pairs.append(np.log(pij / (pi * pj)) / -np.log(pij))
        scores.append(float(np.mean(pairs)))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", default=list(config.COHERENCE_VARIANTS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    counts = sparse.load_npz(args.data_dir / "counts.npz")
    presence = (counts > 0).astype(np.float32).tocsc()
    n_docs = presence.shape[0]
    doc_freq = np.asarray(presence.sum(axis=0)).ravel() / n_docs

    out_path = args.results_dir / "coherence.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    for variant in args.variants:
        phi_path = args.results_dir / f"phi_{variant}.npy"
        if not phi_path.exists():
            print(f"skip {variant}: {phi_path.name} not found")
            continue
        scores = npmi_scores(np.load(phi_path), presence, doc_freq, n_docs)
        out[variant] = scores
        print(f"{variant}: mean NPMI {np.mean(scores):.4f}  "
              f"median {np.median(scores):.4f}")
    atomic_json(out_path, out)


if __name__ == "__main__":
    main()
