"""Stages 2 to 4 with grouped multi-horizon covariances (paper Section 3.6).

Covariance instruments are estimated at half-lives of 20, 60 and 120 trading
days on excess returns, giving 3L instruments plus an intercept. Sparse IPCA
penalises the three rows of each narrative jointly (paper Eq. 8). The
stock-month panel is cached under data/cache/ and reused when the inputs are
unchanged.

Usage: python scripts/price_multi_horizon.py VARIANT CRSP_DAILY.parquet FF_DAILY.zip OUTPUT_DIR

Outputs in OUTPUT_DIR: oos.parquet (monthly OOS portfolio returns),
diagnostics.json (each refit), selection_last_retrain.csv, summary.json.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.price_single_horizon import (load_daily_returns, load_ff_daily_rf,
                                          load_theta, monthly_returns, to_trading_days)
from src.evaluation import sharpe
from src.experiment import atomic_json
from src.stage2_shocks import shocks
from src.stage3_covariances import build_panel, covariance_instruments
from src.stage4_sparse_ipca import mve_stats, tune_lambda

HALFLIVES = [20, 60, 120]
K = 3
LOOKBACK, MIN_OBS, SHOCK_WINDOW = 252, 120, 5
FIRST_OOS_FRAC, RETRAIN_EVERY, N_LAMS = 0.6, 12, 8


def file_identity(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def build_or_load_panel(variant, returns_path, rf_path, cache):
    cache.mkdir(parents=True, exist_ok=True)
    names = ["X", "y", "m_ids", "s_ids", "scales"]
    files = {n: cache / f"{n}.npy" for n in names}
    months_path, manifest_path = cache / "months.json", cache / "manifest.json"
    identity = {"variant": variant, "halflives": HALFLIVES, "lookback": LOOKBACK,
                "min_obs": MIN_OBS, "shock_window": SHOCK_WINDOW,
                "theta": file_identity(ROOT / "results" / f"theta_daily_{variant}.parquet"),
                "returns": file_identity(returns_path), "rf": file_identity(rf_path)}
    complete = all(f.exists() for f in files.values()) and months_path.exists()
    if complete and manifest_path.exists() and json.loads(manifest_path.read_text()) == identity:
        arrays = {n: np.load(files[n], allow_pickle=(n == "s_ids")) for n in names}
        months = [pd.Period(m, freq="M") for m in json.loads(months_path.read_text())]
        print(f"loaded cached panel X={arrays['X'].shape}")
        return arrays["X"], arrays["y"], arrays["m_ids"], arrays["s_ids"], months, arrays["scales"]

    theta = load_theta(variant)
    daily = load_daily_returns(returns_path)
    rf = load_ff_daily_rf(rf_path)
    daily = daily.join(rf, on="date")
    if daily["rf"].isna().any():
        raise ValueError("risk-free series does not cover the CRSP dates")
    daily["ret_excess"] = daily["ret"] - daily["rf"]
    returns = daily.pivot_table(index="date", columns="id", values="ret_excess")
    z = shocks(to_trading_days(theta, returns.index), window=SHOCK_WINDOW)

    per_horizon = []
    for halflife in HALFLIVES:
        print(f"covariance instruments, half-life {halflife}")
        per_horizon.append(covariance_instruments(returns, z, halflife=halflife,
                                                  lookback=LOOKBACK, min_obs=MIN_OBS))
    common = sorted(set.intersection(*(set(p) for p in per_horizon)))
    combined = {}
    for month in common:
        frames = []
        for halflife, panel in zip(HALFLIVES, per_horizon):
            frame = panel[month].copy()
            frame.columns = [f"topic_{c}_hl_{halflife}" for c in frame.columns]
            frames.append(frame)
        combined[month] = pd.concat(frames, axis=1, join="inner")

    panel = build_panel(combined, monthly_returns(daily, rf_daily=rf))
    X, y, m_ids, s_ids, months, scales = panel
    for n, value in zip(names, [X, y, m_ids, s_ids, scales]):
        np.save(files[n], value)
    atomic_json(months_path, [str(m) for m in months])
    atomic_json(manifest_path, identity)
    print(f"built panel X={X.shape}")
    return panel


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    variant, returns_path, rf_path, output_dir = sys.argv[1:5]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache" / f"{variant}_excess_multihorizon"
    X, y, m_ids, _, months, _ = build_or_load_panel(variant, returns_path, rf_path, cache)

    topics = (X.shape[1] - 1) // len(HALFLIVES)
    groups = [[1 + h * topics + l for h in range(len(HALFLIVES))] for l in range(topics)]
    first_oos = int((int(m_ids.max()) + 1) * FIRST_OOS_FRAC)
    portfolio, dates, diagnostics = [], [], []
    Gamma = weights = fit = None
    for month_id in range(first_oos, int(m_ids.max()) + 1):
        if (month_id - first_oos) % RETRAIN_EVERY == 0 or Gamma is None:
            train = m_ids < month_id
            best, _ = tune_lambda(X[train], y[train], m_ids[train], K=K, seed=0,
                                  verbose=False, n_lams=N_LAMS, groups=groups)
            fit = best["fit"]
            Gamma = fit["Gamma"]
            stats = mve_stats(fit["F"])
            weights = stats["weights"]
            diagnostics.append({"retrain_month_id": month_id,
                                "retrain_month": str(months[month_id]),
                                "lambda": float(fit["lam"]),
                                "active_topics": int(fit["active"].sum()),
                                "is_sharpe": float(stats["sharpe_ann"])})
            print(f"refit at month {month_id} ({months[month_id]}): lambda={fit['lam']:.3g}, "
                  f"active={int(fit['active'].sum())}, IS Sharpe={stats['sharpe_ann']:.2f}")
        selected = m_ids == month_id
        if not selected.any():
            continue
        loadings = X[selected] @ Gamma
        factor_return = np.linalg.solve(loadings.T @ loadings + np.eye(K),
                                        loadings.T @ y[selected])
        portfolio.append(float(weights @ factor_return))
        dates.append(months[month_id])

    series = pd.Series(portfolio, index=pd.PeriodIndex(dates, freq="M"), name="mve_oos")
    series.to_frame().to_parquet(output / "oos.parquet")
    atomic_json(output / "diagnostics.json", diagnostics)

    labels = json.load(open(ROOT / "results" / f"labels_{variant}.json"))
    norms = [float(np.linalg.norm(fit["Gamma"][g])) for g in groups]
    selection = pd.DataFrame([{"topic": l, "label": labels[l], "group_norm": n}
                              for l, n in enumerate(norms) if n > 1e-10])
    selection.sort_values("group_norm", ascending=False).to_csv(
        output / "selection_last_retrain.csv", index=False)

    meta_path = ROOT / "results" / f"meta_{variant}.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    summary = {"variant": variant, "return_type": "excess", "K": K,
               "halflives": HALFLIVES, "grouped_by_topic": True,
               "lambda_grid_size": N_LAMS, "panel_obs": int(len(y)),
               "n_features": int(X.shape[1] - 1),
               "topic_clusterer": meta.get("clusterer"),
               "topic_term_weighting": meta.get("term_weighting"),
               "oos_sharpe_nf": sharpe(series), "oos_months": int(len(series))}
    atomic_json(output / "summary.json", summary)
    print("SUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
