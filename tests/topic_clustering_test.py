"""Fast CPU checks for spherical clustering and soft-weighted c-TF-IDF."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

from src.spherical_kmeans import SphericalKMeans
from src.stage1_bert import (
    ctfidf_phi,
    fit_bert_topics,
    sparsify_topic_weights,
    top_chunk_candidates,
)


failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        failures.append(name)


rng = np.random.default_rng(12)
true_centers = np.eye(3, 18, dtype=np.float32)
labels = np.repeat(np.arange(3), 250)
embeddings = true_centers[labels] + rng.normal(0, 0.08, (len(labels), 18))
embeddings = embeddings.astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

test_directory = ROOT / "tests" / "_topic_checkpoint_test"
if test_directory.exists():
    shutil.rmtree(test_directory)
test_directory.mkdir()
try:
    checkpoint = test_directory / "spherical.npz"
    model = SphericalKMeans(
        3, batch_size=128, max_iter=6, n_init=4, init_size=300,
        random_state=4, device="cpu", checkpoint_path=checkpoint,
        checkpoint_fingerprint="synthetic-v1",
    ).fit(embeddings)
    predicted = np.argmax(embeddings @ model.cluster_centers_.T, axis=1)
    purity = sum(np.max(np.bincount(labels[predicted == cluster], minlength=3))
                 for cluster in range(3)) / len(labels)
    check("spherical centroids have unit norm",
          np.allclose(np.linalg.norm(model.cluster_centers_, axis=1), 1, atol=1e-6))
    check("spherical clustering recovers separated topics", purity > 0.98,
          f"purity={purity:.3f}")
    check("spherical objective is high", model.objective_ > 0.9,
          f"mean cosine={model.objective_:.3f}")
    check("spherical checkpoint written", checkpoint.exists())

    resumed = SphericalKMeans(
        3, batch_size=128, max_iter=8, n_init=4, init_size=300,
        random_state=4, device="cpu", checkpoint_path=checkpoint,
        checkpoint_fingerprint="synthetic-v1",
    ).fit(embeddings)
    check("spherical checkpoint resumes", resumed.n_iter_ >= model.n_iter_)
finally:
    shutil.rmtree(test_directory)


counts = sparse.csr_matrix([
    [10, 0, 1],
    [0, 10, 1],
    [6, 4, 0],
])
theta = np.asarray([
    [0.95, 0.05],
    [0.05, 0.95],
    [0.60, 0.40],
])
soft_phi = ctfidf_phi(counts, theta, L=2, smooth=0.1)
hard_phi = ctfidf_phi(counts, theta.argmax(axis=1), L=2, smooth=0.1)
sublinear_phi = ctfidf_phi(
    counts, theta, L=2, smooth=0.1, sublinear_tf=True
)
check("soft c-TF-IDF rows lie on simplex",
      np.allclose(soft_phi.sum(axis=1), 1, atol=1e-10))
check("soft c-TF-IDF preserves topic-specific terms",
      soft_phi[0, 0] > soft_phi[0, 1] and soft_phi[1, 1] > soft_phi[1, 0])
check("soft c-TF-IDF differs from hard article assignment",
      not np.allclose(soft_phi, hard_phi))
check("sublinear TF rows lie on simplex",
      np.allclose(sublinear_phi.sum(axis=1), 1, atol=1e-10))
check("sublinear TF changes topic-word weights",
      not np.allclose(sublinear_phi, soft_phi))

sparse_input = np.asarray([
    [0.70, 0.20, 0.10],
    [0.40, 0.35, 0.25],
    [0.90, 0.06, 0.04],
    [0.50, 0.30, 0.20],
])
sparse_weights, sparse_meta = sparsify_topic_weights(
    sparse_input, top_k=2, power=2.0, confidence_power=1.0,
    confidence_quantile=0.5,
)
check("sparse-soft retains at most top-k topics",
      np.all(np.count_nonzero(sparse_weights, axis=1) <= 2))
check("sparse-soft filters low-margin documents",
      np.count_nonzero(sparse_weights.sum(axis=1) == 0) >= 2)
check("sparse-soft records filtering diagnostics",
      0 < sparse_meta["ctfidf_fraction_retained"] <= 0.5)

tiny_embeddings = np.asarray([
    [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0],
    [0.1, 0.9, 0.0], [0.8, 0.2, 0.0], [0.2, 0.8, 0.0],
], dtype=np.float32)
tiny_centroids = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
)
tiny_counts = sparse.csr_matrix([
    [4, 0, 1, 0], [3, 1, 0, 0], [0, 4, 0, 1], [1, 3, 0, 0]
])
tiny_model, tiny_clusterer = fit_bert_topics(
    None,
    tiny_counts,
    ["alpha", "beta", "gamma", "delta"],
    pd.Series(pd.to_datetime([
        "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"
    ])),
    np.asarray([3, 4, 5, 6]),
    L=2,
    embeddings=tiny_embeddings,
    doc_index=np.asarray([0, 0, 1, 2, 3, 3]),
    chunk_weights=np.ones(6),
    precomputed_centroids=tiny_centroids,
    centroid_source="synthetic.npy",
    term_weighting="sparse_soft",
    ctfidf_confidence_quantile=0.0,
    ctfidf_sublinear_tf=True,
    cluster_device="cpu",
)
check("precomputed centroids skip clustering",
      tiny_model.meta["cluster_reused"]
      and tiny_model.meta["cluster_iterations"] == 0)
check("precomputed-centroid output validates", tiny_model.validate())
check("precomputed centroids are normalized",
      np.allclose(np.linalg.norm(tiny_clusterer.cluster_centers_, axis=1), 1))

candidate_indices, candidate_scores, candidate_device = top_chunk_candidates(
    tiny_embeddings, tiny_centroids, n_candidates=2, batch_size=3,
    device="cpu",
)
check("representative chunks have expected shape",
      candidate_indices.shape == (2, 2)
      and candidate_scores.shape == (2, 2))
check("representative chunks use cosine ranking",
      set(candidate_indices[0]) == {0, 1}
      and set(candidate_indices[1]) == {2, 3})
check("representative chunk device is recorded", candidate_device == "cpu")

print()
print("FAILURES:", failures if failures else "none")
raise SystemExit(1 if failures else 0)

