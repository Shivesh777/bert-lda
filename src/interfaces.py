"""Shared data contracts for the narrative pipeline.

Stage 1 (LDA or BERT) must emit an AttentionModel with identical structure,
so Stages 2-4 do not depend on the topic model: theta rows lie on the
L-simplex and phi rows are distributions over the shared vocabulary.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AttentionModel:
    """Output contract of any Stage 1 topic layer.

    theta_daily : DataFrame, index=trading date (datetime), columns=topic ids 0..L-1.
                  Each row is on the simplex (term-count-weighted average of
                  per-article attention that day).
    phi         : ndarray (L, V). Row l is a probability distribution over the
                  vocabulary (LDA: topic-term matrix; BERT: normalized c-TF-IDF).
    vocab       : list[str], length V, shared between variants.
    labels      : list[str], length L, human-readable topic names (cosmetic).
    doc_theta   : optional per-article attention for diagnostics/retrieval.
    meta        : free-form provenance (model name, hyperparameters).
    """
    theta_daily: pd.DataFrame
    phi: np.ndarray
    vocab: list
    labels: list
    doc_theta: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    def validate(self, atol=1e-6):
        th = self.theta_daily.to_numpy()
        assert (th >= -atol).all(), "theta has negative entries"
        rs = th.sum(axis=1)
        assert np.allclose(rs[rs > 0], 1.0, atol=1e-4), "theta rows not on simplex"
        assert self.phi.shape[0] == self.theta_daily.shape[1], "L mismatch phi vs theta"
        assert self.phi.shape[1] == len(self.vocab), "V mismatch phi vs vocab"
        assert np.allclose(self.phi.sum(axis=1), 1.0, atol=1e-4), "phi rows not distributions"
        assert len(self.labels) == self.phi.shape[0]
        return True

    def top_terms(self, l, n=10):
        idx = np.argsort(self.phi[l])[::-1][:n]
        return [self.vocab[i] for i in idx]
