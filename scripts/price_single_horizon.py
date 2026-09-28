"""Stages 2 to 4 with a single covariance horizon: attention shocks,
EWMA covariance instruments (half-life 60, 252-day lookback), sparse IPCA
with K factors, and the out-of-sample MVE portfolio.

Usage: python scripts/price_single_horizon.py VARIANT CRSP_DAILY.parquet [K] [FF_DAILY.zip]

  VARIANT   suffix of results/theta_daily_{VARIANT}.parquet
  CRSP      daily returns with columns date, permno (or ticker), ret
  K         number of latent factors (default 3)
  FF zip    Kenneth French daily factors; if given, excess returns are used

Outputs in results/: selection_{VARIANT}_{total|excess}.csv,
oos_{VARIANT}_{total|excess}.parquet and summary_{VARIANT}_{total|excess}.json
(suffix override: OUT_SUFFIX).
"""
from pathlib import Path
import json
import os
import sys
import zipfile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.stage2_shocks import shocks
from src.stage3_covariances import covariance_instruments, build_panel
from src.stage4_sparse_ipca import tune_lambda, mve_stats
from src.evaluation import oos_mve, sharpe, report_selection

RESULTS = ROOT / "results"


def load_theta(variant):
    df = pq.read_table(RESULTS / f"theta_daily_{variant}.parquet").to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = range(df.shape[1])
    return df


def to_trading_days(theta, trading_days):
    """Move each calendar day's attention to the next trading day and
    average within trading day (weekend and holiday news is carried forward)."""
    td = pd.DatetimeIndex(sorted(trading_days))
    pos = td.searchsorted(theta.index, side="left")
    valid = pos < len(td)
    mapped = theta[valid].copy()
    mapped.index = td[pos[valid]]
    return mapped.groupby(level=0).mean()


def load_ff_daily_rf(path):
    """Daily risk-free rate (decimal) from the Kenneth French CSV zip."""
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(name)
    ff = pd.read_csv(BytesIO(raw), skiprows=4)
    ff["date"] = pd.to_datetime(ff[ff.columns[0]].astype(str).str.strip(),
                                format="%Y%m%d", errors="coerce")
    ff["rf"] = pd.to_numeric(ff["RF"], errors="coerce") / 100.0
    return ff.dropna(subset=["date", "rf"]).set_index("date")["rf"].sort_index()


def load_daily_returns(path):
    daily = pd.read_csv(path) if str(path).endswith(".csv") else pq.read_table(path).to_pandas()
    cols = {c.lower(): c for c in daily.columns}
    id_col = next(cols[c] for c in ["id", "permno", "ticker", "tick"] if c in cols)
    daily = daily.rename(columns={id_col: "id", cols.get("ret", "ret"): "ret",
                                  cols.get("date", "date"): "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce").astype(float)
    return daily


def monthly_returns(daily, rf_daily=None):
    """Compound daily to monthly returns; months with fewer than 15 daily
    observations are dropped. Subtracts the compounded risk-free rate if given."""
    daily = daily.copy()
    daily["m"] = daily["date"].dt.to_period("M")
    grp = daily.groupby(["id", "m"])["ret"]
    mret = grp.apply(lambda s: float(np.prod(1 + s.dropna()) - 1))
    mret = mret[grp.count() >= 15].unstack("id")
    if rf_daily is not None:
        rf = rf_daily.rename("rf").to_frame()
        rf["m"] = rf.index.to_period("M")
        rf_monthly = rf.groupby("m")["rf"].apply(lambda s: float(np.prod(1 + s) - 1))
        mret = mret.sub(rf_monthly, axis=0)
    return mret


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    variant, returns_path = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rf_path = sys.argv[4] if len(sys.argv) > 4 else None
    suffix = os.getenv("OUT_SUFFIX", "_excess" if rf_path else "_total")

    theta = load_theta(variant)
    daily = load_daily_returns(returns_path)
    rf_daily = None
    if rf_path:
        rf_daily = load_ff_daily_rf(rf_path)
        daily = daily.join(rf_daily, on="date")
        missing = int(daily["rf"].isna().sum())
        if missing:
            raise ValueError(f"risk-free rate missing for {missing} daily rows")
        daily["ret_excess"] = daily["ret"] - daily["rf"]
        R = daily.pivot_table(index="date", columns="id", values="ret_excess")
    else:
        R = daily.pivot_table(index="date", columns="id", values="ret")
    print(f"[{variant}] theta {theta.shape}, returns {R.shape}")

    z = shocks(to_trading_days(theta, R.index), window=5)
    print(f"shocks {z.shape}: {z.index.min().date()} to {z.index.max().date()}")
    cov = covariance_instruments(R, z, halflife=60, lookback=252, min_obs=120)
    print(f"instrument months: {len(cov)}")

    mret = monthly_returns(daily, rf_daily=rf_daily)
    X, y, m_ids, s_ids, months, scales = build_panel(cov, mret)
    print(f"panel: {X.shape[0]:,} obs, {len(months)} months, {X.shape[1] - 1} instruments")

    best, _ = tune_lambda(X, y, m_ids, K=K, criterion="validation", verbose=True)
    fit = best["fit"]
    labels = json.load(open(RESULTS / f"labels_{variant}.json"))
    sel = report_selection(fit, labels)
    sel.to_csv(RESULTS / f"selection_{variant}{suffix}.csv", index=False)
    is_stats = mve_stats(fit["F"])
    print(f"\nselected {len(sel)} narratives; in-sample MVE Sharpe {is_stats['sharpe_ann']:.2f}")
    print(sel.head(15).to_string())

    port = oos_mve(X, y, m_ids, months, K=K, first_oos_frac=0.6,
                   retrain_every=12, lam=None, seed=0)
    port.to_frame().to_parquet(RESULTS / f"oos_{variant}{suffix}.parquet")

    ew = mret.mean(axis=1)                    # equal-weight market, same universe
    ew.index = pd.PeriodIndex(ew.index, freq="M")
    summary = {
        "variant": variant, "K": K, "lam": fit["lam"],
        "return_type": "excess" if rf_path else "total",
        "n_active": int(fit["active"].sum()),
        "panel_obs": int(X.shape[0]), "n_months": len(months),
        "is_sharpe": is_stats["sharpe_ann"],
        "oos_sharpe_nf": sharpe(port),
        "oos_sharpe_ewmkt": sharpe(ew.reindex(port.index)),
        "oos_months": len(port),
    }
    json.dump(summary, open(RESULTS / f"summary_{variant}{suffix}.json", "w"), indent=1)
    print("\nSUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
