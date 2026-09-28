"""Out-of-sample MVE evaluation and selection reporting."""
import numpy as np
import pandas as pd

from .stage4_sparse_ipca import fit_sparse_ipca, mve_stats, tune_lambda


def oos_mve(X, y, m_ids, months, K=3, first_oos_frac=0.6, retrain_every=12,
            lam=None, lam_grid=None, seed=0, verbose=True):
    """Expanding-window OOS MVE portfolio, retrained every `retrain_every` months.

    Trains on months [0, T0), retrains every `retrain_every` months. At each
    retrain: (re)tune lam if lam is None, estimate Gamma and training-period
    MVE weights. OOS month t: f_t from ridge cross-section with FIXED Gamma
    (portfolio weights depend only on start-of-month instruments), portfolio
    return = w_mve' f_t.
    """
    n_months = int(m_ids.max()) + 1
    T0 = int(n_months * first_oos_frac)
    port, oos_months = [], []
    Gamma, w = None, None
    for t in range(T0, n_months):
        if (t - T0) % retrain_every == 0 or Gamma is None:
            train_sel = m_ids < t
            Xtr, ytr, mtr = X[train_sel], y[train_sel], m_ids[train_sel]
            if lam is None:
                best, _ = tune_lambda(Xtr, ytr, mtr, K=K, lams=lam_grid,
                                      seed=seed, verbose=False)
                fit = best["fit"]
            else:
                fit = fit_sparse_ipca(Xtr, ytr, mtr, K=K, lam=lam, seed=seed)
            Gamma = fit["Gamma"]
            stats = mve_stats(fit["F"])
            w = stats["weights"]
            if verbose:
                print(f"retrain at m={t}: lam={fit['lam']:.2e} "
                      f"active={int(fit['active'].sum())} IS-SR={stats['sharpe_ann']:.2f}")
        sel = m_ids == t
        if not sel.any():
            continue
        B = X[sel] @ Gamma
        f_t = np.linalg.solve(B.T @ B + np.eye(K), B.T @ y[sel])
        port.append(float(w @ f_t))
        oos_months.append(months[t])
    port = pd.Series(port, index=pd.PeriodIndex(oos_months, freq="M"),
                     name="mve_oos")
    return port


def sharpe(series, ann=12):
    s = series.dropna()
    return float(s.mean() / s.std() * np.sqrt(ann))


def report_selection(fit, labels):
    """Selected narratives with row norms, sorted by importance."""
    norms = np.linalg.norm(fit["Gamma"][1:], axis=1)
    rows = [{"topic": l, "label": labels[l], "row_norm": norms[l]}
            for l in range(len(norms)) if norms[l] > 1e-10]
    return (pd.DataFrame(rows).sort_values("row_norm", ascending=False)
            .reset_index(drop=True))
