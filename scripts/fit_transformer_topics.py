"""Fit the transformer branch: cluster the passage embeddings into L topics,
form article and daily attention, and describe each topic with c-TF-IDF over
the shared vocabulary.

The three transformer specifications in the paper are

  st_frozen           --clusterer minibatch_kmeans --term-weighting hard
  st_spherical_soft   --clusterer spherical        --term-weighting soft
  st_spherical        --clusterer spherical        --term-weighting sparse_soft

st_spherical and st_spherical_soft share one set of centroids: fit
st_spherical_soft first, then pass --centroids-path results/centroids_st_spherical_soft.npy
so the second run only changes the term head. Spherical clustering
checkpoints each Lloyd pass in data/clustering_checkpoints/ and resumes.

Outputs in results/: theta_daily_{variant}.parquet, phi_{variant}.npy,
centroids_{variant}.npy, labels_{variant}.json, meta_{variant}.json.

Usage: python scripts/fit_transformer_topics.py --variant st_spherical
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

import config
from src.experiment import atomic_json
from src.stage1_bert import fit_bert_topics

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CHECKPOINTS = DATA / "clustering_checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="st_spherical", help="output file suffix")
    p.add_argument("--clusterer", choices=("spherical", "minibatch_kmeans"),
                   default=config.TOPIC_CLUSTERER)
    p.add_argument("--term-weighting", choices=("soft", "hard", "sparse_soft"),
                   default=config.TOPIC_TERM_WEIGHTING)
    p.add_argument("--device", default=config.TOPIC_CLUSTER_DEVICE,
                   help="auto, cpu, cuda or cuda:N")
    p.add_argument("--embedding-path", type=Path,
                   default=DATA / "chunk_embeddings.npz")
    p.add_argument("--centroids-path", type=Path,
                   help="reuse fitted centroids and skip clustering")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    return p.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.variant):
        raise ValueError("variant may contain only letters, digits, _, . and -")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    archive = np.load(args.embedding_path)
    embeddings = archive["emb"]
    doc_index = archive["doc_index"]
    chunk_weights = archive["chunk_w"]
    counts = sparse.load_npz(args.data_dir / "counts.npz")
    vocab = json.load(open(args.data_dir / "vocab.json"))
    meta = pq.read_table(args.data_dir / "doc_meta.parquet").to_pandas()
    print(f"{args.variant}: embeddings {embeddings.shape}, L={config.L} "
          f"[{time.time() - t0:.0f}s]")

    # Checkpoints are keyed on the embedding file and clustering settings so a
    # stale checkpoint is never resumed.
    stat = args.embedding_path.stat()
    identity = {"embeddings": f"{args.embedding_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}",
                "shape": [int(v) for v in embeddings.shape],
                "clusterer": args.clusterer, "L": config.L, "seed": config.SEED,
                "n_init": config.TOPIC_CLUSTER_N_INIT,
                "init_size": config.TOPIC_CLUSTER_INIT_SIZE}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    centroids = np.load(args.centroids_path) if args.centroids_path else None
    checkpoint = (args.checkpoint_dir / f"topics_{args.variant}.npz"
                  if args.clusterer == "spherical" and centroids is None else None)

    model, clusterer = fit_bert_topics(
        None, counts, vocab, meta["date"], meta["n_tokens"].to_numpy(),
        L=config.L, seed=config.SEED, temperature=config.TOPIC_TEMPERATURE,
        embeddings=embeddings, doc_index=doc_index, chunk_weights=chunk_weights,
        clusterer=args.clusterer, term_weighting=args.term_weighting,
        cluster_batch_size=config.TOPIC_CLUSTER_BATCH_SIZE,
        cluster_max_iter=config.TOPIC_CLUSTER_MAX_ITER,
        cluster_n_init=config.TOPIC_CLUSTER_N_INIT,
        cluster_tol=config.TOPIC_CLUSTER_TOL,
        cluster_init_size=config.TOPIC_CLUSTER_INIT_SIZE,
        cluster_device=args.device, cluster_checkpoint=checkpoint,
        cluster_checkpoint_fingerprint=fingerprint,
        assignment_batch_size=config.TOPIC_ASSIGNMENT_BATCH_SIZE,
        precomputed_centroids=centroids,
        centroid_source=str(args.centroids_path) if args.centroids_path else None,
        ctfidf_top_k=config.TOPIC_CTFIDF_TOP_K,
        ctfidf_power=config.TOPIC_CTFIDF_POWER,
        ctfidf_confidence_power=config.TOPIC_CTFIDF_CONFIDENCE_POWER,
        ctfidf_confidence_quantile=config.TOPIC_CTFIDF_CONFIDENCE_QUANTILE,
        ctfidf_sublinear_tf=config.TOPIC_CTFIDF_SUBLINEAR_TF,
        model_name=config.ENCODER, variant=args.variant,
    )

    theta = model.theta_daily.copy()
    theta.columns = [str(c) for c in theta.columns]
    theta.reset_index(names="date").to_parquet(
        args.results_dir / f"theta_daily_{args.variant}.parquet")
    np.save(args.results_dir / f"phi_{args.variant}.npy", model.phi)
    np.save(args.results_dir / f"centroids_{args.variant}.npy", clusterer.cluster_centers_)
    atomic_json(args.results_dir / f"labels_{args.variant}.json", model.labels)
    atomic_json(args.results_dir / f"meta_{args.variant}.json",
                {**model.meta, "cluster_fingerprint": fingerprint,
                 "cluster_checkpoint": str(checkpoint) if checkpoint else None})
    print(f"saved results/*_{args.variant}.* [{time.time() - t0:.0f}s]")
    for l in range(0, config.L, 8):
        print(f"  {l:3d}: {model.labels[l]}")


if __name__ == "__main__":
    main()
