"""Corpus preparation shared by both Stage 1 variants.

Builds the common vocabulary (unigrams + bigrams, doc-frequency filtered)
and the article-term count matrix used by LDA directly and by the BERT
variant's c-TF-IDF term head. Both variants use the same vocabulary so
their phi matrices and coherence scores are comparable.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

EXTRA_STOP = {
    "said", "says", "mr", "ms", "inc", "corp", "co", "ltd", "llc",
    "company", "companies", "share", "shares", "stock", "stocks",
    "percent", "pct", "quarter", "year", "week", "month", "reuters",
    "nasdaq", "views", "opinions", "author", "herein", "com", "www",
    "http", "https", "zacks", "motley", "fool", "benzinga",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    s = s.str.replace(r"https?://\S+", " ", regex=True)
    s = s.str.replace(r"[^A-Za-z\s]", " ", regex=True)
    s = s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def build_vocab_and_counts(texts, max_features=15000, min_df=20, max_df=0.4):
    """Returns (count_matrix csr [n_docs, V], vocab list, n_tokens per doc)."""
    vec = CountVectorizer(
        ngram_range=(1, 2), stop_words=STOPWORDS,
        max_features=max_features, min_df=min_df, max_df=max_df,
        token_pattern=r"[a-z]{3,}",
    )
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out().tolist()
    n_tokens = np.asarray(X.sum(axis=1)).ravel()
    return X.tocsr(), vocab, n_tokens


def aggregate_daily(doc_theta: np.ndarray, dates: pd.Series,
                    weights: np.ndarray) -> pd.DataFrame:
    """Term-count-weighted average of per-article theta by calendar day.

    doc_theta : (n_docs, L) rows on simplex
    dates     : per-doc date (datetime-like)
    weights   : per-doc term counts (daily attention is term-count weighted)
    """
    w = np.maximum(weights.astype(float), 1.0)
    df = pd.DataFrame(doc_theta * w[:, None])
    df["_date"] = pd.to_datetime(dates).values
    df["_w"] = w
    g = df.groupby("_date")
    num = g[[c for c in df.columns if c not in ("_date", "_w")]].sum()
    den = g["_w"].sum()
    theta_daily = num.div(den, axis=0)
    theta_daily.columns = range(theta_daily.shape[1])
    return theta_daily.sort_index()
