"""Topic-count ablation (paper Appendix B): refit both text layers at several
values of L on the corpus, vocabulary and embeddings produced by the main
pipeline, then score coherence and topic diversity at every L.

    python ablation/sweep_topics.py                 # L from config.SWEEP_L
    python ablation/sweep_topics.py --L 50 70 80    # explicit list
    python ablation/sweep_topics.py --arms st       # transformer branch only

Requires data/counts.npz, data/vocab.json, data/doc_meta.parquet and
data/chunk_embeddings.npz from steps 3 and 5 of the main pipeline.

Per L the fits are written next to the main results with an _L{L} suffix:

    results/theta_daily_lda_L{L}.parquet, phi_lda_L{L}.npy, labels, meta
    results/centroids_st_spherical_soft_L{L}.npy       (spherical k-means)
    results/theta_daily_st_spherical_L{L}.parquet, ...  (sparse-soft head,
                                                         same centroids)

The summary (mean and median NPMI, top-10 diversity, fit diagnostics) is
written to ablation/results/sweep_summary.{json,csv,md}. Every step resumes:
finished variants are skipped, spherical passes and Gibbs sweeps checkpoint.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import config

SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
OUT = ROOT / "ablation" / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=list(config.SWEEP_L))
    p.add_argument("--arms", default="lda,st", help="comma list from {lda,st}")
    p.add_argument("--device", default=config.TOPIC_CLUSTER_DEVICE)
    return p.parse_args()


def run(command, env_extra=None, name=""):
    env = {**os.environ, **(env_extra or {})}
    print(f"\n[run ] {name}", flush=True)
    started = time.time()
    code = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env).returncode
    print(f"[done] {name} in {(time.time() - started) / 60:.1f} min", flush=True)
    if code != 0:
        sys.exit(f"step failed: {name} (exit {code}); rerun to resume")


def meta_matches(variant, **expected):
    path = RESULTS / f"meta_{variant}.json"
    if not path.exists() or not (RESULTS / f"phi_{variant}.npy").exists():
        return False
    meta = json.loads(path.read_text())
    return all(meta.get(k) == v for k, v in expected.items())


def fit_lda(L):
    variant = f"lda_L{L}"
    if meta_matches(variant, L=L):
        print(f"[skip] {variant}")
    else:
        run([SCRIPTS / "fit_lda.py", "--variant", variant, "--drop-checkpoint"],
            {"TOPIC_L": str(L)}, name=variant)
    return [variant]


def fit_st(L, device):
    soft = f"st_spherical_soft_L{L}"
    sparse = f"st_spherical_L{L}"
    env = {"TOPIC_L": str(L)}
    if meta_matches(soft, L=L, clusterer="spherical", term_weighting="soft"):
        print(f"[skip] {soft}")
    else:
        run([SCRIPTS / "fit_transformer_topics.py", "--variant", soft,
             "--clusterer", "spherical", "--term-weighting", "soft",
             "--device", device], env, name=soft)
    if meta_matches(sparse, L=L, clusterer="spherical", term_weighting="sparse_soft"):
        print(f"[skip] {sparse}")
    else:
        run([SCRIPTS / "fit_transformer_topics.py", "--variant", sparse,
             "--clusterer", "spherical", "--term-weighting", "sparse_soft",
             "--device", device, "--centroids-path",
             RESULTS / f"centroids_{soft}.npy"], env, name=sparse)
    return [soft, sparse]


def diversity(variant, top_n=10):
    phi = np.load(RESULTS / f"phi_{variant}.npy")
    top = np.argpartition(phi, -top_n, axis=1)[:, -top_n:]
    return float(len(np.unique(top)) / top.size)


def summarize(variants_by_L):
    coherence = json.loads((RESULTS / "coherence.json").read_text())
    rows = []
    for L, variants in variants_by_L.items():
        for variant in variants:
            meta = json.loads((RESULTS / f"meta_{variant}.json").read_text())
            scores = coherence[variant]
            arm = ("LDA" if variant.startswith("lda") else
                   "Spherical ST, soft head" if "spherical_soft" in variant else
                   "Spherical ST, sparse-soft head")
            rows.append({
                "arm": arm, "L": L, "variant": variant,
                "npmi_mean": float(np.mean(scores)),
                "npmi_median": float(np.median(scores)),
                "npmi_sd": float(np.std(scores, ddof=1)),
                "diversity_top10": diversity(variant),
                "lda_ll_per_word": meta.get("ll_per_word"),
                "cluster_mean_cosine": meta.get("cluster_objective"),
                "cluster_iterations": meta.get("cluster_iterations"),
            })
    rows.sort(key=lambda r: (r["arm"], r["L"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_summary.json").write_text(json.dumps(
        {"settings": {"encoder": config.ENCODER, "temperature": config.TOPIC_TEMPERATURE,
                      "seed": config.SEED, "lda_gibbs_iters": config.LDA_GIBBS_ITERS,
                      "vocab_size": config.VOCAB_SIZE},
         "rows": rows}, indent=2))
    with (OUT / "sweep_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| arm | L | mean NPMI | median NPMI | diversity |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['L']} | {r['npmi_mean']:.4f} | "
                     f"{r['npmi_median']:.4f} | {r['diversity_top10']:.3f} |")
    (OUT / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    args = parse_args()
    arms = {a.strip() for a in args.arms.split(",")}
    variants_by_L = {}
    for L in args.L:
        variants = []
        if "lda" in arms:
            variants += fit_lda(L)
        if "st" in arms:
            variants += fit_st(L, args.device)
        variants_by_L[L] = variants
        run([SCRIPTS / "coherence.py", *variants], name=f"NPMI at L={L}")
    summarize(variants_by_L)


if __name__ == "__main__":
    main()
