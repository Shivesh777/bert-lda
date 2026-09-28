"""Streamed spherical k-means for unit-normalized text embeddings.

The implementation performs exact spherical Lloyd updates while scanning the
embedding matrix in bounded-memory blocks. Assignment maximizes cosine
similarity and every updated centroid is projected back to the unit sphere.
CUDA is used for assignment/accumulation when requested and available.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("spherical k-means received a zero-length vector")
    return values / norms


class SphericalKMeans:
    """Spherical k-means with streamed full-corpus centroid updates.

    ``batch_size`` controls memory use, not the statistical objective: each
    iteration assigns every observation before updating the centroids. A
    small MiniBatchKMeans fit supplies robust k-means++ initialization; all
    subsequent optimization uses cosine assignments and unit centroids.
    """

    def __init__(self, n_clusters: int, *, batch_size: int = 16384,
                 max_iter: int = 10, n_init: int = 10, tol: float = 1e-5,
                 init_size: int = 100000, random_state: int = 0,
                 device: str | None = "auto", checkpoint_path: str | Path | None = None,
                 checkpoint_fingerprint: str = "", verbose: bool = False):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.init_size = int(init_size)
        self.random_state = int(random_state)
        self.device = "auto" if device is None else str(device)
        self.checkpoint_path = (Path(checkpoint_path)
                                if checkpoint_path is not None else None)
        self.checkpoint_fingerprint = str(checkpoint_fingerprint)
        self.verbose = bool(verbose)

    def _effective_device(self) -> str:
        requested = self.device.lower()
        if requested == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if requested.startswith("cuda"):
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("CUDA clustering requires PyTorch") from exc
            if not torch.cuda.is_available():
                raise RuntimeError(f"requested {self.device}, but CUDA is unavailable")
            return requested
        if requested != "cpu":
            raise ValueError("cluster device must be 'auto', 'cpu', or a CUDA device")
        return "cpu"

    def _initialize(self, embeddings, sample_weight, rng) -> np.ndarray:
        n_samples = embeddings.shape[0]
        n_init_rows = min(
            n_samples, max(self.init_size, self.n_clusters * 20)
        )
        indices = np.sort(rng.choice(n_samples, size=n_init_rows, replace=False))
        initial_data = _normalize_rows(
            np.array(embeddings[indices], dtype=np.float32, copy=True)
        )
        initializer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=min(self.batch_size, n_init_rows),
            n_init=self.n_init,
            max_no_improvement=30,
        )
        weights = None if sample_weight is None else np.asarray(sample_weight)[indices]
        initializer.fit(initial_data, sample_weight=weights)
        return _normalize_rows(initializer.cluster_centers_)

    def _load_checkpoint(self, n_samples: int, n_features: int):
        path = self.checkpoint_path
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            expected = {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
            }
            observed = {key: int(saved[key]) for key in expected}
            fingerprint = str(saved["fingerprint"].item())
            if observed != expected or fingerprint != self.checkpoint_fingerprint:
                raise RuntimeError(
                    f"spherical clustering checkpoint mismatch at {path}; "
                    "use a new output variant or remove only this checkpoint"
                )
            return {
                "centers": np.asarray(saved["centers"], dtype=np.float32),
                "n_iter": int(saved["n_iter"]),
                "converged": bool(saved["converged"]),
                "history": np.asarray(saved["objective_history"], dtype=float).tolist(),
            }

    def _save_checkpoint(self, n_samples: int, n_features: int,
                         centers: np.ndarray, n_iter: int, converged: bool,
                         objective_history: list[float]) -> None:
        path = self.checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            np.savez(
                handle,
                centers=np.asarray(centers, dtype=np.float32),
                n_iter=np.int64(n_iter),
                converged=np.bool_(converged),
                objective_history=np.asarray(objective_history, dtype=np.float64),
                n_samples=np.int64(n_samples),
                n_features=np.int64(n_features),
                n_clusters=np.int64(self.n_clusters),
                random_state=np.int64(self.random_state),
                fingerprint=np.asarray(self.checkpoint_fingerprint),
            )
        os.replace(temp, path)

    def _numpy_epoch(self, embeddings, centers, sample_weight):
        n_samples, n_features = embeddings.shape
        sums = np.zeros((self.n_clusters, n_features), dtype=np.float64)
        counts = np.zeros(self.n_clusters, dtype=np.float64)
        objective_sum = 0.0
        weight_sum = 0.0
        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            batch = _normalize_rows(
                np.array(embeddings[start:stop], dtype=np.float32, copy=True)
            )
            weights = (np.ones(stop - start, dtype=np.float32)
                       if sample_weight is None else
                       np.asarray(sample_weight[start:stop], dtype=np.float32))
            similarities = batch @ centers.T
            labels = similarities.argmax(axis=1)
            best = similarities[np.arange(len(labels)), labels]
            membership = sparse.csr_matrix(
                (weights, (labels, np.arange(len(labels)))),
                shape=(self.n_clusters, len(labels)),
            )
            sums += membership @ batch
            counts += np.bincount(labels, weights=weights,
                                  minlength=self.n_clusters)
            objective_sum += float(np.dot(best, weights))
            weight_sum += float(weights.sum())
        return sums, counts, objective_sum / max(weight_sum, 1e-12)

    def _torch_epoch(self, embeddings, centers, sample_weight, device):
        import torch

        n_samples, n_features = embeddings.shape
        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
        sums_t = torch.zeros((self.n_clusters, n_features), dtype=torch.float32,
                             device=device)
        counts_t = torch.zeros(self.n_clusters, dtype=torch.float32, device=device)
        objective_sum = 0.0
        weight_sum = 0.0
        with torch.inference_mode():
            for start in range(0, n_samples, self.batch_size):
                stop = min(start + self.batch_size, n_samples)
                batch = torch.as_tensor(
                    np.array(embeddings[start:stop], dtype=np.float32, copy=True),
                    device=device,
                )
                batch = torch.nn.functional.normalize(batch, dim=1)
                weights = (torch.ones(stop - start, dtype=torch.float32, device=device)
                           if sample_weight is None else
                           torch.as_tensor(np.asarray(sample_weight[start:stop]),
                                           dtype=torch.float32, device=device))
                best, labels = torch.max(batch @ centers_t.T, dim=1)
                sums_t.index_add_(0, labels, batch * weights[:, None])
                counts_t.index_add_(0, labels, weights)
                objective_sum += float(torch.dot(best, weights).item())
                weight_sum += float(weights.sum().item())
        return (sums_t.cpu().numpy(), counts_t.cpu().numpy(),
                objective_sum / max(weight_sum, 1e-12))

    def fit(self, embeddings, sample_weight=None):
        if len(embeddings.shape) != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        n_samples, n_features = map(int, embeddings.shape)
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be at least n_clusters")
        if sample_weight is not None and len(sample_weight) != n_samples:
            raise ValueError("sample_weight length must equal n_samples")

        rng = np.random.default_rng(self.random_state)
        restored = self._load_checkpoint(n_samples, n_features)
        if restored is None:
            centers = self._initialize(embeddings, sample_weight, rng)
            first_iteration = 0
            converged = False
            history = []
        else:
            centers = _normalize_rows(restored["centers"])
            first_iteration = restored["n_iter"]
            converged = restored["converged"]
            history = restored["history"]
            if self.verbose:
                print(f"  resumed spherical k-means at iteration {first_iteration}")

        effective_device = self._effective_device()
        for iteration in range(first_iteration, self.max_iter):
            if converged:
                break
            if effective_device.startswith("cuda"):
                sums, counts, objective = self._torch_epoch(
                    embeddings, centers, sample_weight, effective_device
                )
            else:
                sums, counts, objective = self._numpy_epoch(
                    embeddings, centers, sample_weight
                )

            updated = centers.copy()
            nonempty = counts > 0
            updated[nonempty] = (sums[nonempty] /
                                 counts[nonempty, None]).astype(np.float32)
            for cluster in np.flatnonzero(~nonempty):
                replacement = np.array(
                    embeddings[int(rng.integers(n_samples))],
                    dtype=np.float32, copy=True,
                )[None, :]
                updated[cluster] = _normalize_rows(replacement)[0]
            updated = _normalize_rows(updated)
            shift = float(np.mean(1.0 - np.sum(centers * updated, axis=1)))
            centers = updated
            history.append(float(objective))
            converged = shift <= self.tol
            completed = iteration + 1
            self._save_checkpoint(
                n_samples, n_features, centers, completed, converged, history
            )
            if self.verbose:
                print(f"  spherical iteration {completed}/{self.max_iter}: "
                      f"mean cosine={objective:.6f}, centroid shift={shift:.3e}" +
                      (" [converged]" if converged else ""), flush=True)

        self.cluster_centers_ = centers
        self.n_iter_ = first_iteration if not history else len(history)
        self.objective_history_ = history
        self.objective_ = history[-1] if history else float("nan")
        self.converged_ = converged
        self.device_ = effective_device
        return self
