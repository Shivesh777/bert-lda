"""Compare two fitted topic variants.

Reports mean NPMI, top-n term diversity, the cosine similarity of matched
term distributions (topics are aligned by maximum cosine similarity, since
topic order is arbitrary) and the agreement of the daily attention series.
Used in the paper to check that the sparse-soft head changes the term lists
but leaves daily attention unchanged (--require-same-attention).

Usage: python scripts/compare_topic_models.py st_spherical_soft st_spherical \
           --require-same-attention
Output: results/comparison_{baseline}_vs_{candidate}.json
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.experiment import atomic_json


def topic_diversity(phi, top_n):
    top = np.argpartition(phi, -top_n, axis=1)[:, -top_n:]
    return float(len(np.unique(top)) / top.size)


def read_theta(path):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--attention-atol", type=float, default=1e-6)
    parser.add_argument("--require-same-attention", action="store_true",
                        help="fail if the daily attention series differ")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    res = args.results_dir

    base_phi = np.load(res / f"phi_{args.baseline}.npy")
    cand_phi = np.load(res / f"phi_{args.candidate}.npy")
    if base_phi.shape != cand_phi.shape:
        raise ValueError("the two variants must share L and the vocabulary")

    def unit(rows):
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)

    sims = unit(base_phi) @ unit(cand_phi).T
    base_order, cand_order = linear_sum_assignment(-sims)
    order = cand_order[np.argsort(base_order)]
    matched_cosine = sims[np.arange(len(order)), order]

    coherence = {}
    if (res / "coherence.json").exists():
        coherence = json.load(open(res / "coherence.json"))

    base_theta = read_theta(res / f"theta_daily_{args.baseline}.parquet")
    cand_theta = read_theta(res / f"theta_daily_{args.candidate}.parquet")
    dates = base_theta.index.intersection(cand_theta.index)
    delta = base_theta.loc[dates].to_numpy() - cand_theta.loc[dates].to_numpy()
    max_delta = float(np.max(np.abs(delta)))
    invariant = max_delta <= args.attention_atol
    if args.require_same_attention and not invariant:
        raise RuntimeError(f"daily attention differs: max |delta| {max_delta:.3e} "
                           f"exceeds {args.attention_atol:.3e}")

    aligned = cand_theta.iloc[:, order]
    aligned.columns = base_theta.columns
    corr = [float(base_theta.loc[dates].iloc[:, l].corr(aligned.loc[dates].iloc[:, l]))
            for l in range(base_phi.shape[0])]

    def mean_npmi(variant):
        values = coherence.get(variant)
        return None if values is None else float(np.mean(values))

    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n_topics": int(base_phi.shape[0]),
        "mean_npmi": {"baseline": mean_npmi(args.baseline),
                      "candidate": mean_npmi(args.candidate)},
        "topic_diversity": {"top_n": args.top_n,
                            "baseline": topic_diversity(base_phi, args.top_n),
                            "candidate": topic_diversity(cand_phi, args.top_n)},
        "aligned_term_cosine": {"mean": float(np.mean(matched_cosine)),
                                "median": float(np.median(matched_cosine)),
                                "minimum": float(np.min(matched_cosine))},
        "aligned_attention_correlation": {"mean": float(np.nanmean(corr)),
                                          "median": float(np.nanmedian(corr)),
                                          "minimum": float(np.nanmin(corr))},
        "same_order_attention": {"max_absolute_difference": max_delta,
                                 "mean_absolute_difference": float(np.mean(np.abs(delta))),
                                 "tolerance": args.attention_atol,
                                 "invariant": invariant},
        "candidate_topic_for_each_baseline_topic": order.tolist(),
    }
    out = res / f"comparison_{args.baseline}_vs_{args.candidate}.json"
    atomic_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
