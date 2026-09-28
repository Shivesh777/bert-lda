"""Stage 1, transformer branch: topic layer with the same output interface as LDA.

chunk -> frozen sentence-transformer embeddings -> spherical k-means (k=L)
on unit-normalized embeddings -> per-article soft attention theta as the
token-weighted share of the article's chunks in each topic -> soft-weighted
c-TF-IDF over the SHARED vocabulary, row-normalized, as phi-hat.

Output is an AttentionModel identical in structure to the LDA variant.
"""
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from types import SimpleNamespace

from .interfaces import AttentionModel
from .spherical_kmeans import SphericalKMeans
from .textprep import aggregate_daily
from .stage1_lda import auto_label


def chunk_articles(texts, max_words=180, min_words=8):
    """Split each article into paragraph-ish chunks capped at max_words.

    Returns (chunks list[str], doc_index ndarray, chunk_weights ndarray)
    where chunk_weights = word counts (token weighting for attention shares).
    """
    chunks, doc_idx, weights = [], [], []
    for i, txt in enumerate(texts):
        if not isinstance(txt, str) or not txt.strip():
            continue
        paras = [p.strip() for p in txt.split("\n") if len(p.split()) >= min_words]
        if not paras:
            paras = [txt.strip()]
        for p in paras:
            words = p.split()
            for j in range(0, len(words), max_words):
                piece = " ".join(words[j:j + max_words])
                if len(piece.split()) >= min_words:
                    chunks.append(piece)
                    doc_idx.append(i)
                    weights.append(len(piece.split()))
    return chunks, np.asarray(doc_idx), np.asarray(weights, dtype=float)


def embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=256, device=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(chunks, batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def soft_assign(emb, centroids, temperature=0.07):
    """Softmax over cosine similarity to centroids. Low temperature ~ hard."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    sims = emb @ normalize(centroids, axis=1).T
    z = sims / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def ctfidf_phi(counts, doc_topics, L=None, smooth=1.0,
               sublinear_tf=False):
    """Class-based TF-IDF from hard labels or soft document-topic weights.

    ``doc_topics`` may be a length-n vector of hard topic ids or an
    ``(n_docs, L)`` matrix on the simplex.  In the soft case,
    every document-term count contributes fractionally to every topic instead
    of assigning the complete article to its dominant topic.
    """
    doc_topics = np.asarray(doc_topics)
    n_docs, V = counts.shape
    if sublinear_tf:
        if sparse.issparse(counts):
            weighted_counts = counts.astype(np.float64, copy=True).tocsr()
            if np.any(weighted_counts.data < 0):
                raise ValueError("term counts must be nonnegative")
            weighted_counts.data = 1.0 + np.log(weighted_counts.data)
        else:
            weighted_counts = np.asarray(counts, dtype=np.float64)
            if np.any(weighted_counts < 0):
                raise ValueError("term counts must be nonnegative")
            positive = weighted_counts > 0
            transformed = np.zeros_like(weighted_counts)
            transformed[positive] = 1.0 + np.log(weighted_counts[positive])
            weighted_counts = transformed
    else:
        weighted_counts = counts

    if doc_topics.ndim == 1:
        if len(doc_topics) != n_docs:
            raise ValueError("hard topic vector length must equal document count")
        if L is None:
            L = int(doc_topics.max()) + 1
        labels = doc_topics.astype(int, copy=False)
        if np.any(labels < 0) or np.any(labels >= L):
            raise ValueError("hard topic id outside [0, L)")
        membership = sparse.csr_matrix(
            (np.ones(n_docs), (np.arange(n_docs), labels)),
            shape=(n_docs, L),
        )
        tf = membership.T @ weighted_counts
        tf = tf.toarray() if sparse.issparse(tf) else np.asarray(tf)
    elif doc_topics.ndim == 2:
        if doc_topics.shape[0] != n_docs:
            raise ValueError("soft topic matrix rows must equal document count")
        if L is None:
            L = doc_topics.shape[1]
        if doc_topics.shape[1] != L:
            raise ValueError("soft topic matrix columns must equal L")
        if not np.all(np.isfinite(doc_topics)) or np.any(doc_topics < 0):
            raise ValueError("soft topic weights must be finite and nonnegative")
        tf = np.asarray(weighted_counts.T @ doc_topics, dtype=np.float64).T
    else:
        raise ValueError("doc_topics must be hard ids or a soft weight matrix")

    tf = np.asarray(tf, dtype=np.float64) + smooth       # (L, V)
    # c-TF-IDF: term freq within class, reweighted by inverse cross-class freq
    avg = tf.sum() / max(L, 1)
    idf = np.log1p(avg / tf.sum(axis=0))
    w = tf * idf[None, :]
    phi = w / w.sum(axis=1, keepdims=True)
    return phi


def sparsify_topic_weights(doc_theta, top_k=2, power=2.0,
                           confidence_power=1.0,
                           confidence_quantile=0.5):
    """Make soft c-TF-IDF membership selective without changing attention.

    Each document keeps only its ``top_k`` topic probabilities. Those values
    are sharpened by ``power`` and renormalized. Documents below the requested
    top-1 minus top-2 margin quantile contribute no words, while retained
    documents are weighted by ``margin ** confidence_power``. The returned
    matrix is intended only for c-TF-IDF; daily topic attention should continue
    to use the original ``doc_theta``.
    """
    theta = np.asarray(doc_theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] == 0:
        raise ValueError("doc_theta must be a nonempty 2-D matrix")
    if not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError("doc_theta must be finite and nonnegative")
    if not 1 <= top_k <= theta.shape[1]:
        raise ValueError("top_k must be between 1 and the number of topics")
    if power <= 0:
        raise ValueError("power must be positive")
    if confidence_power < 0:
        raise ValueError("confidence_power must be nonnegative")
    if not 0 <= confidence_quantile < 1:
        raise ValueError("confidence_quantile must be in [0, 1)")

    n_docs, n_topics = theta.shape
    top_indices = np.argpartition(theta, n_topics - top_k, axis=1)[:, -top_k:]
    rows = np.arange(n_docs)[:, None]
    retained = np.power(theta[rows, top_indices], power)
    retained_sum = retained.sum(axis=1, keepdims=True)
    retained = np.divide(
        retained, retained_sum, out=np.zeros_like(retained),
        where=retained_sum > 0,
    )

    if n_topics == 1:
        margins = theta[:, 0].copy()
    else:
        top_two = np.partition(theta, n_topics - 2, axis=1)[:, -2:]
        margins = top_two.max(axis=1) - top_two.min(axis=1)
    cutoff = (float(np.quantile(margins, confidence_quantile))
              if n_docs else 0.0)
    confident = margins >= cutoff
    confidence = np.power(margins, confidence_power)
    confidence[~confident] = 0.0
    retained *= confidence[:, None]

    weights = np.zeros_like(theta)
    weights[rows, top_indices] = retained
    diagnostics = {
        "ctfidf_top_k": int(top_k),
        "ctfidf_power": float(power),
        "ctfidf_confidence_power": float(confidence_power),
        "ctfidf_confidence_quantile": float(confidence_quantile),
        "ctfidf_margin_cutoff": cutoff,
        "ctfidf_fraction_retained": float(confident.mean()) if n_docs else 0.0,
        "ctfidf_mean_document_weight": float(weights.sum(axis=1).mean())
        if n_docs else 0.0,
    }
    return weights, diagnostics


def _resolve_assignment_device(device):
    requested_device = str(device).lower()
    torch = None
    if requested_device == "auto" or requested_device.startswith("cuda"):
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError:
            if requested_device.startswith("cuda"):
                raise RuntimeError(
                    "CUDA requested, but PyTorch is not installed"
                )
    if requested_device == "auto":
        effective_device = "cuda" if (
            torch is not None and torch.cuda.is_available()
        ) else "cpu"
    elif requested_device.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but torch.cuda.is_available() is false"
            )
        effective_device = requested_device
    elif requested_device == "cpu":
        effective_device = "cpu"
    else:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    return effective_device, torch


def top_chunk_candidates(embeddings, centroids, n_candidates=50,
                         batch_size=65536, device="auto"):
    """Return the nearest chunk indices and cosines for every topic.

    The calculation is streamed and retains only ``n_candidates`` per topic,
    so it is safe for the full embedding archive. The caller can then enforce
    distinct documents and attach the corresponding text metadata.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if len(embeddings.shape) != 2:
        raise ValueError("embeddings must be two-dimensional")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must contain at least one row")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    centers = np.asarray(centroids, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != embeddings.shape[1]:
        raise ValueError("centroid and embedding dimensions do not match")
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(center_norms == 0) or not np.all(np.isfinite(centers)):
        raise ValueError("centroids must be finite and nonzero")
    centers = centers / center_norms
    effective_device, torch = _resolve_assignment_device(device)
    if effective_device.startswith("cuda"):
        center_tensor = torch.as_tensor(
            centers, dtype=torch.float32, device=effective_device
        )

    best_scores = None
    best_indices = None
    for start in range(0, embeddings.shape[0], batch_size):
        stop = min(start + batch_size, embeddings.shape[0])
        batch = np.asarray(embeddings[start:stop], dtype=np.float32)
        local_k = min(n_candidates, stop - start)
        if effective_device.startswith("cuda"):
            with torch.inference_mode():
                batch_tensor = torch.as_tensor(
                    batch, dtype=torch.float32, device=effective_device
                )
                batch_tensor = torch.nn.functional.normalize(batch_tensor, dim=1)
                similarities = batch_tensor @ center_tensor.T
                local_scores, local_indices = torch.topk(
                    similarities, k=local_k, dim=0
                )
                local_scores = local_scores.T.cpu().numpy()
                local_indices = local_indices.T.cpu().numpy() + start
        else:
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("embeddings contain a zero-length vector")
            similarities = (batch / norms) @ centers.T
            local_indices = np.argpartition(
                similarities, similarities.shape[0] - local_k, axis=0
            )[-local_k:, :]
            local_scores = np.take_along_axis(
                similarities, local_indices, axis=0
            )
            local_indices = local_indices.T + start
            local_scores = local_scores.T

        if best_scores is None:
            best_scores = local_scores
            best_indices = local_indices
        else:
            merged_scores = np.concatenate((best_scores, local_scores), axis=1)
            merged_indices = np.concatenate((best_indices, local_indices), axis=1)
            keep = min(n_candidates, merged_scores.shape[1])
            selected = np.argpartition(
                merged_scores, merged_scores.shape[1] - keep, axis=1
            )[:, -keep:]
            best_scores = np.take_along_axis(merged_scores, selected, axis=1)
            best_indices = np.take_along_axis(merged_indices, selected, axis=1)

    order = np.argsort(best_scores, axis=1)[:, ::-1]
    return (
        np.take_along_axis(best_indices, order, axis=1),
        np.take_along_axis(best_scores, order, axis=1),
        effective_device,
    )


def _document_topic_weights(embeddings, centroids, doc_index, chunk_weights,
                            n_docs, temperature, batch_size=65536,
                            device="auto"):
    """Aggregate soft chunk assignments without materializing n_chunks x L."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    L = centroids.shape[0]
    doc_theta = np.zeros((n_docs, L), dtype=np.float64)
    assignment_device, torch = _resolve_assignment_device(device)

    centroids = np.asarray(centroids, dtype=np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(centroid_norms == 0):
        raise ValueError("centroids must have nonzero norm")
    unit_centroids = centroids / centroid_norms
    if assignment_device.startswith("cuda"):
        centroid_tensor = torch.as_tensor(
            unit_centroids, dtype=torch.float32, device=assignment_device
        )

    for start in range(0, len(doc_index), batch_size):
        stop = min(start + batch_size, len(doc_index))
        embedding_batch = np.asarray(embeddings[start:stop], dtype=np.float32)
        if assignment_device.startswith("cuda"):
            with torch.inference_mode():
                embedding_tensor = torch.as_tensor(
                    embedding_batch, dtype=torch.float32,
                    device=assignment_device,
                )
                embedding_tensor = torch.nn.functional.normalize(
                    embedding_tensor, dim=1
                )
                probabilities = torch.softmax(
                    (embedding_tensor @ centroid_tensor.T) / temperature, dim=1
                ).cpu().numpy()
        else:
            probabilities = soft_assign(
                embedding_batch, unit_centroids, temperature
            )
        weights = np.asarray(chunk_weights[start:stop], dtype=np.float64)
        np.add.at(
            doc_theta,
            np.asarray(doc_index[start:stop], dtype=int),
            probabilities * weights[:, None],
        )
    row_sums = doc_theta.sum(axis=1, keepdims=True)
    empty = row_sums.ravel() == 0
    doc_theta[~empty] /= row_sums[~empty]
    doc_theta[empty] = 1.0 / L
    return doc_theta, assignment_device


def fit_bert_topics(texts_raw, counts, vocab, dates, n_tokens, L=60, seed=0,
                    temperature=0.07, embeddings=None, doc_index=None,
                    chunk_weights=None, chunks=None, clusterer="spherical",
                    term_weighting="soft", cluster_batch_size=16384,
                    cluster_max_iter=10, cluster_n_init=10,
                    cluster_tol=1e-5, cluster_init_size=100000,
                    cluster_device="auto", cluster_checkpoint=None,
                    cluster_checkpoint_fingerprint="",
                    assignment_batch_size=65536,
                    precomputed_centroids=None, centroid_source=None,
                    ctfidf_top_k=2, ctfidf_power=2.0,
                    ctfidf_confidence_power=1.0,
                    ctfidf_confidence_quantile=0.5,
                    ctfidf_sublinear_tf=False,
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    variant="bert"):
    """texts_raw: original (uncleaned) article texts for chunking/embedding.
    counts/vocab: SHARED vocabulary count matrix (for c-TF-IDF phi-hat).
    embeddings (+doc_index, chunk_weights): precomputed to skip embedding.
    """
    if embeddings is None:
        chunks, doc_index, chunk_weights = chunk_articles(texts_raw)
        embeddings = embed_chunks(chunks, model_name=model_name)

    cluster_reused = precomputed_centroids is not None
    if cluster_reused:
        centers = np.asarray(precomputed_centroids, dtype=np.float32)
        if centers.shape != (L, embeddings.shape[1]):
            raise ValueError(
                "precomputed centroids must have shape "
                f"({L}, {embeddings.shape[1]}), got {centers.shape}"
            )
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(centers)):
            raise ValueError("precomputed centroids must be finite and nonzero")
        km = SimpleNamespace(
            cluster_centers_=centers / norms,
            n_iter_=0,
            objective_=None,
            device_="loaded",
        )
    elif clusterer == "spherical":
        km = SphericalKMeans(
            n_clusters=L,
            random_state=seed,
            batch_size=cluster_batch_size,
            max_iter=cluster_max_iter,
            n_init=cluster_n_init,
            tol=cluster_tol,
            init_size=cluster_init_size,
            device=cluster_device,
            checkpoint_path=cluster_checkpoint,
            checkpoint_fingerprint=cluster_checkpoint_fingerprint,
            verbose=True,
        ).fit(embeddings)
    elif clusterer == "minibatch_kmeans":
        # Euclidean mini-batch k-means: the frozen ST baseline in the paper.
        km = MiniBatchKMeans(n_clusters=L, random_state=seed, batch_size=4096,
                             n_init=10, max_no_improvement=50)
        km.fit(embeddings)
    else:
        raise ValueError("clusterer must be 'spherical' or 'minibatch_kmeans'")

    n_docs = counts.shape[0]
    doc_theta, assignment_device = _document_topic_weights(
        embeddings, km.cluster_centers_, doc_index, chunk_weights, n_docs,
        temperature, batch_size=assignment_batch_size, device=cluster_device,
    )

    weighting_meta = {}
    if term_weighting == "soft":
        topic_membership = doc_theta
    elif term_weighting == "hard":
        topic_membership = doc_theta.argmax(axis=1)
    elif term_weighting == "sparse_soft":
        topic_membership, weighting_meta = sparsify_topic_weights(
            doc_theta,
            top_k=ctfidf_top_k,
            power=ctfidf_power,
            confidence_power=ctfidf_confidence_power,
            confidence_quantile=ctfidf_confidence_quantile,
        )
    else:
        raise ValueError(
            "term_weighting must be 'soft', 'hard', or 'sparse_soft'"
        )
    phi = ctfidf_phi(
        counts, topic_membership, L,
        sublinear_tf=ctfidf_sublinear_tf,
    )

    theta_daily = aggregate_daily(doc_theta, dates, n_tokens)
    objective = getattr(km, "objective_", None)
    objective = (float(objective) if objective is not None
                 and np.isfinite(objective) else None)
    model = AttentionModel(
        theta_daily=theta_daily, phi=phi, vocab=list(vocab),
        labels=auto_label(phi, vocab),
        doc_theta=None,
        meta={"variant": variant, "L": L, "encoder": model_name,
              "temperature": temperature, "seed": seed,
              "n_chunks": int(len(doc_index)), "clusterer": clusterer,
              "term_weighting": term_weighting,
              "ctfidf_sublinear_tf": bool(ctfidf_sublinear_tf),
              "cluster_reused": bool(cluster_reused),
              "centroid_source": centroid_source,
              "cluster_iterations": int(getattr(km, "n_iter_", 0)),
              "cluster_converged": getattr(km, "converged_", None),
              "cluster_objective": objective,
              "cluster_device": str(getattr(km, "device_", "cpu")),
              "assignment_device": assignment_device,
              **weighting_meta},
    )
    model.validate()
    return model, km

