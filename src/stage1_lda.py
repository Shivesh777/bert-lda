"""Stage 1, LDA branch, in-process sklearn variational LDA.

scripts/fit_lda.py fits the reported model with tomotopy (collapsed Gibbs);
this module is used by the smoke test and small experiments.

text -> per-article theta (L-simplex) -> term-count-weighted daily attention.
phi = topic-term distributions straight from the fitted model.
"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from .interfaces import AttentionModel
from .textprep import aggregate_daily


def auto_label(phi, vocab, n=3):
    labels = []
    for l in range(phi.shape[0]):
        idx = np.argsort(phi[l])[::-1][:n]
        labels.append(" / ".join(vocab[i] for i in idx))
    return labels


def fit_lda(counts, vocab, dates, n_tokens, L=60, seed=0, max_iter=25,
            learning_method="online", batch_size=4096):
    """counts: csr (n_docs, V) from the shared vocabulary."""
    lda = LatentDirichletAllocation(
        n_components=L, random_state=seed, max_iter=max_iter,
        learning_method=learning_method, batch_size=batch_size,
        doc_topic_prior=None, topic_word_prior=None, n_jobs=-1,
    )
    doc_theta = lda.fit_transform(counts)               # already normalized rows
    doc_theta = doc_theta / doc_theta.sum(axis=1, keepdims=True)
    phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    theta_daily = aggregate_daily(doc_theta, dates, n_tokens)
    model = AttentionModel(
        theta_daily=theta_daily, phi=phi, vocab=list(vocab),
        labels=auto_label(phi, vocab),
        doc_theta=None,
        meta={"variant": "lda", "L": L, "max_iter": max_iter, "seed": seed},
    )
    model.validate()
    return model, lda
