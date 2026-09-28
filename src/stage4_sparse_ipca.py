"""Stage 4: sparse IPCA (Bybee, Kelly and Su, 2023, Eq. 8; paper Eq. 8 for the
grouped multi-horizon form).

min_{Gamma,f}  sum_{i,t} (r_{i,t} - c_{i,t-} Gamma f_t)^2
             + lam * n_obs * sum_{l>=1} ||Gamma_l||_2      (group lasso, rows)
             + sum_t ||f_t||_2^2                            (ridge on factors)

Instruments are standardized upstream (sigma_l = 1); the constant row (l=0)
is unpenalized. Estimated by alternating least squares:
  f-step  : per month, ridge cross-section regression on B = X Gamma.
  G-step  : block coordinate descent over rows with exact group prox
            (1-D root-finding on the row norm via eigendecomp of A_l).
"""
import numpy as np


def f_step(X, y, m_ids, Gamma, n_months):
    K = Gamma.shape[1]
    B = X @ Gamma
    F = np.zeros((n_months, K))
    for m in range(n_months):
        sel = m_ids == m
        if not sel.any():
            continue
        Bm, ym = B[sel], y[sel]
        F[m] = np.linalg.solve(Bm.T @ Bm + np.eye(K), Bm.T @ ym)
    return F


def _row_prox(A, b, lam):
    """argmin_g g'Ag - 2b'g + lam*||g||_2 via eigendecomp + Newton on s=||g||."""
    nb = np.linalg.norm(b)
    if nb <= lam / 2:
        return np.zeros_like(b)
    d, Q = np.linalg.eigh(A)
    d = np.maximum(d, 1e-12)
    bq = Q.T @ b
    # solve h(s) = sum bq_k^2/(d_k + lam/(2s))^2 - s^2 = 0 for s > 0
    lo, hi = 1e-12, nb / max(d.min(), 1e-12)
    for _ in range(100):
        s = 0.5 * (lo + hi)
        val = np.sum(bq ** 2 / (d + lam / (2 * s)) ** 2) - s ** 2
        if val > 0:
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    return Q @ (bq / (d + lam / (2 * s)))


def gamma_step(X, y, m_ids, F, Gamma, lam_total, n_sweeps=3, groups=None):
    """BCD over rows of Gamma. lam_total = lam * n_obs (row 0 unpenalized)."""
    n_obs, P = X.shape
    K = F.shape[1]
    Fo = F[m_ids]                                    # (n_obs, K)
    yhat = ((X @ Gamma) * Fo).sum(axis=1)
    if groups is not None:
        groups = [np.asarray(group, dtype=int) for group in groups]
        for _ in range(n_sweeps):
            # Constant row remains unpenalized.
            xl = X[:, 0]
            contrib = xl * (Fo @ Gamma[0])
            e = y - (yhat - contrib)
            A = (Fo * (xl ** 2)[:, None]).T @ Fo
            b = Fo.T @ (xl * e)
            new0 = np.linalg.solve(A + 1e-10 * np.eye(K), b)
            yhat += xl * (Fo @ new0) - contrib
            Gamma[0] = new0
            for idx in groups:
                # Jointly update all horizon rows belonging to one topic.
                Z = (X[:, idx, None] * Fo[:, None, :]).reshape(n_obs, -1)
                old = Gamma[idx].reshape(-1)
                contrib = Z @ old
                e = y - (yhat - contrib)
                new = _row_prox(Z.T @ Z, Z.T @ e, lam_total)
                yhat += Z @ (new - old)
                Gamma[idx] = new.reshape(len(idx), K)
        return Gamma
    for _ in range(n_sweeps):
        for l in range(P):
            xl = X[:, l]
            contrib_l = xl * (Fo @ Gamma[l])
            e = y - (yhat - contrib_l)               # partial residual
            xw = xl ** 2
            A = (Fo * xw[:, None]).T @ Fo
            b = Fo.T @ (xl * e)
            if l == 0:
                new = np.linalg.solve(A + 1e-10 * np.eye(K), b)
            else:
                new = _row_prox(A, b, lam_total)
            yhat += xl * (Fo @ new) - contrib_l
            Gamma[l] = new
    return Gamma


def objective(X, y, m_ids, Gamma, F, lam_total, groups=None):
    Fo = F[m_ids]
    resid = y - ((X @ Gamma) * Fo).sum(axis=1)
    if groups is None:
        pen_g = lam_total * np.linalg.norm(Gamma[1:], axis=1).sum()
    else:
        pen_g = lam_total * sum(np.linalg.norm(Gamma[np.asarray(group, dtype=int)])
                                for group in groups)
    pen_f = (F ** 2).sum()
    return float(resid @ resid + pen_g + pen_f)


def dense_init(X, y, m_ids, K, seed=0, iters=15, groups=None):
    """Unpenalized ALS (lam=0) from random init. Establishes the scale of
    (Gamma, F) jointly; used as warm start for every penalized fit. Without
    it, large-lam fits collapse into the degenerate fixed point Gamma=0, F=0
    (f-step ridge shrinks F when B=X Gamma is small, small F kills all rows
    in the Gamma-step, and the death is self-reinforcing)."""
    n_obs, P = X.shape
    n_months = int(m_ids.max()) + 1
    rng = np.random.default_rng(seed)
    Gamma = rng.standard_normal((P, K)) * 0.1
    for _ in range(iters):
        F = f_step(X, y, m_ids, Gamma, n_months)
        Gamma = gamma_step(X, y, m_ids, F, Gamma, lam_total=0.0,
                           n_sweeps=1, groups=groups)
    return Gamma


def lambda_grid(X, y, m_ids, Gamma_dense, n_grid=12, span=3e3, groups=None):
    """Data-adaptive grid: lam_max = the penalty at which the strongest row
    dies given the dense fit's factors; grid descends geometrically."""
    n_obs = X.shape[0]
    n_months = int(m_ids.max()) + 1
    F = f_step(X, y, m_ids, Gamma_dense, n_months)
    Fo = F[m_ids]
    yhat = ((X @ Gamma_dense) * Fo).sum(axis=1)
    norms = []
    if groups is None:
        for l in range(1, X.shape[1]):
            xl = X[:, l]
            e = y - (yhat - xl * (Fo @ Gamma_dense[l]))
            norms.append(np.linalg.norm(Fo.T @ (xl * e)))
    else:
        for group in groups:
            idx = np.asarray(group, dtype=int)
            Z = (X[:, idx, None] * Fo[:, None, :]).reshape(n_obs, -1)
            old = Gamma_dense[idx].reshape(-1)
            e = y - (yhat - Z @ old)
            norms.append(np.linalg.norm(Z.T @ e))
    lam_max = 2 * max(norms) / n_obs
    return np.geomspace(lam_max, lam_max / span, n_grid)


def fit_sparse_ipca(X, y, m_ids, K=3, lam=1e-4, max_iter=50, tol=1e-6,
                    seed=0, Gamma0=None, verbose=False, groups=None):
    """Returns dict with Gamma (P,K), F (n_months,K), active row mask, obj path."""
    n_obs, P = X.shape
    n_months = int(m_ids.max()) + 1
    lam_total = lam * n_obs
    Gamma = (Gamma0.copy() if Gamma0 is not None
             else dense_init(X, y, m_ids, K, seed=seed, groups=groups))
    path = []
    prev = np.inf
    for it in range(max_iter):
        F = f_step(X, y, m_ids, Gamma, n_months)
        Gamma = gamma_step(X, y, m_ids, F, Gamma, lam_total, groups=groups)
        obj = objective(X, y, m_ids, Gamma, F, lam_total, groups=groups)
        path.append(obj)
        if abs(prev - obj) / max(abs(prev), 1.0) < tol:
            break
        prev = obj
    F = f_step(X, y, m_ids, Gamma, n_months)
    active = (np.linalg.norm(Gamma[1:], axis=1) > 1e-10 if groups is None else
              np.asarray([np.linalg.norm(Gamma[np.asarray(group, dtype=int)]) > 1e-10
                          for group in groups]))
    return {"Gamma": Gamma, "F": F, "active": active, "obj_path": path,
            "lam": lam, "K": K, "n_iter": it + 1}


def mve_stats(F, months_mask=None, ann=12):
    """In-sample MVE Sharpe of factors F (n_months, K)."""
    f = F if months_mask is None else F[months_mask]
    f = f[(np.abs(f).sum(axis=1) > 0)]
    mu = f.mean(axis=0)
    Sig = np.cov(f.T) if f.shape[1] > 1 else np.array([[f.var()]])
    Sig = np.atleast_2d(Sig)
    w = np.linalg.solve(Sig + 1e-12 * np.eye(len(mu)), mu)
    sr_m = float(mu @ w / np.sqrt(max(w @ Sig @ w, 1e-18)))
    return {"sharpe_ann": sr_m * np.sqrt(ann), "weights": w, "mu": mu, "Sigma": Sig}


def tune_lambda(X, y, m_ids, K=3, lams=None, seed=0, verbose=True,
                criterion="validation", val_frac=0.25, n_lams=12, groups=None):
    """Pick lam on a warm-started decreasing path.

    criterion="insample"   : paper-faithful, maximize in-sample MVE Sharpe.
    criterion="validation" : walk-forward split (default). Fit on the first
      (1-val_frac) of months; freeze Gamma and the training MVE weights;
      realize factor returns on the validation months; maximize the
      validation Sharpe of that portfolio. This avoids the in-sample
      criterion's tendency to reward overfit factor means.
    """
    n_months = int(m_ids.max()) + 1
    t_split = int(np.ceil(n_months * (1 - val_frac)))
    tr = m_ids < t_split
    # dense warm start computed ONCE on the relevant sample; every lam on the
    # path starts from it (never from a sparser fit: avoids dead-row hysteresis)
    Xf, yf, mf = (X, y, m_ids) if criterion == "insample" else (X[tr], y[tr], m_ids[tr])
    Gamma_dense = dense_init(Xf, yf, mf, K, seed=seed, groups=groups)
    if lams is None:
        lams = lambda_grid(Xf, yf, mf, Gamma_dense, n_grid=n_lams, groups=groups)
    results = []
    for lam in np.sort(lams)[::-1]:
        if criterion == "insample":
            fit = fit_sparse_ipca(X, y, m_ids, K=K, lam=lam, seed=seed,
                                  Gamma0=Gamma_dense, groups=groups)
            sr = mve_stats(fit["F"])["sharpe_ann"]
        else:
            fit = fit_sparse_ipca(X[tr], y[tr], m_ids[tr], K=K, lam=lam,
                                  seed=seed, Gamma0=Gamma_dense, groups=groups)
            if fit["active"].sum() == 0:
                sr = -np.inf
            else:
                w = mve_stats(fit["F"])["weights"]
                F_all = f_step(X, y, m_ids, fit["Gamma"], n_months)
                port = F_all[t_split:] @ w
                sr = float(port.mean() / max(port.std(), 1e-12) * np.sqrt(12))
        n_active = int(fit["active"].sum())
        results.append({"lam": lam, "sharpe": sr, "n_active": n_active, "fit": fit})
        if verbose:
            print(f"lam={lam:.2e}  SR={sr: .3f}  active={n_active}")
    feas = [r for r in results if r["n_active"] > 0]
    sr_best = max(r["sharpe"] for r in feas)
    # sparsity tie-break (1-SE-rule analog): among lams within tol of the best
    # Sharpe, take the sparsest model (largest lam)
    tol = 0.05 * abs(sr_best) if np.isfinite(sr_best) else 0.0
    near = [r for r in feas if r["sharpe"] >= sr_best - tol]
    best = max(near, key=lambda r: r["lam"])
    if criterion == "validation":
        # refit on ALL months at the chosen lam
        final = fit_sparse_ipca(X, y, m_ids, K=K, lam=best["lam"], seed=seed,
                                Gamma0=best["fit"]["Gamma"], groups=groups)
        best = {**best, "fit": final, "n_active": int(final["active"].sum())}
    return best, results
