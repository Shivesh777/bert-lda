"""Stage 3: stock-month covariance instruments.

For stock i and month t: cov_hat_{i,t} (1 x L) between i's daily returns and
the L daily narrative shocks, over a trailing exponentially-weighted window
that ends on the last trading day before month t begins (no look-ahead). Instruments c_{i,t} = [1, cov_hat_{i,t}].

returns: DataFrame index=date, columns=permno/ticker, values=daily returns.
z      : DataFrame index=date, columns=0..L-1 narrative shocks.
"""
import numpy as np
import pandas as pd


def month_starts(dates: pd.DatetimeIndex):
    months = pd.PeriodIndex(dates, freq="M")
    return pd.Series(dates, index=dates).groupby(months).first()


def ewma_weights(n, halflife):
    k = np.arange(n)[::-1]                       # oldest .. newest
    w = 0.5 ** (k / halflife)
    return w / w.sum()


def covariance_instruments(returns: pd.DataFrame, z: pd.DataFrame,
                           halflife: int = 60, lookback: int = 252,
                           min_obs: int = 120):
    """Returns dict: month (Period) -> DataFrame (stocks x L) of covariances.

    For month t, uses daily data in the `lookback` trading days ending on the
    last day BEFORE t's first trading day.
    """
    common = returns.index.intersection(z.index)
    R = returns.loc[common].sort_index()
    Z = z.loc[common].sort_index()
    dates = R.index
    zm = Z.to_numpy()
    L = zm.shape[1]

    months = pd.PeriodIndex(dates, freq="M").unique().sort_values()
    out = {}
    for m in months[1:]:
        first_in_m = dates[dates >= m.start_time.tz_localize(dates.tz) if dates.tz
                           else dates >= m.start_time]
        first_in_m = first_in_m[pd.PeriodIndex(first_in_m, freq="M") == m]
        if len(first_in_m) == 0:
            continue
        end_pos = dates.searchsorted(first_in_m[0])   # first day of month m
        start_pos = max(0, end_pos - lookback)
        if end_pos - start_pos < min_obs:
            continue
        idx = slice(start_pos, end_pos)
        Rw = R.iloc[idx]
        Zw = zm[idx]
        w = ewma_weights(end_pos - start_pos, halflife)

        Zc = Zw - (w[:, None] * Zw).sum(axis=0, keepdims=True)
        Rv = Rw.to_numpy()
        valid = (~np.isnan(Rv)).sum(axis=0) >= min_obs
        cols = Rw.columns[valid]
        Rv = Rv[:, valid]
        Rmask = np.isnan(Rv)
        Rv0 = np.where(Rmask, 0.0, Rv)
        wobs = w[:, None] * (~Rmask)                  # per-stock weights on obs
        wsum = wobs.sum(axis=0)
        rbar = (Rv0 * w[:, None]).sum(axis=0) / np.maximum(wsum, 1e-12)
        Rc = np.where(Rmask, 0.0, Rv - rbar[None, :])
        cov = (Rc * w[:, None]).T @ Zc                # (n_stocks, L)
        cov = cov / np.maximum(wsum, 1e-12)[:, None]
        out[m] = pd.DataFrame(cov, index=cols, columns=range(L))
    return out


def build_panel(cov_by_month: dict, returns_monthly: pd.DataFrame,
                standardize=True):
    """Assemble estimation panel.

    Aligns instruments c_{i,t} (from cov of month t) with realized month-t
    returns r_{i,t}: cov window ends before t starts, so this is predictive.

    Returns X (n_obs, L+1) instruments incl. constant, y (n_obs,) returns,
    month_ids (n_obs,), stock_ids, months list, scales sigma_l used.
    """
    rows_X, rows_y, m_ids, s_ids = [], [], [], []
    months = sorted(cov_by_month.keys())
    rm = returns_monthly.copy()
    rm.index = pd.PeriodIndex(rm.index, freq="M")
    for mi, m in enumerate(months):
        if m not in rm.index:
            continue
        cov = cov_by_month[m]
        r = rm.loc[m]
        common = cov.index.intersection(r.dropna().index)
        if len(common) == 0:
            continue
        rows_X.append(cov.loc[common].to_numpy())
        rows_y.append(r.loc[common].to_numpy())
        m_ids.append(np.full(len(common), mi))
        s_ids.append(common.to_numpy())
    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)
    m_ids = np.concatenate(m_ids)
    s_ids = np.concatenate(s_ids)

    scales = X.std(axis=0)
    if standardize:
        X = X / np.maximum(scales, 1e-12)[None, :]
    X = np.hstack([np.ones((X.shape[0], 1)), X])      # constant first
    return X, y, m_ids, s_ids, months, scales
