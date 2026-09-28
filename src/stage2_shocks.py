"""Stage 2: attention levels -> narrative shocks.

z_tau = theta_tau - (1/W) * sum_{j=1..W} theta_{tau-j}   (W = 5 trading days)

Trailing window only (no look-ahead). Rows with insufficient history dropped.
"""
import pandas as pd


def shocks(theta_daily: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    ma = theta_daily.rolling(window=window, min_periods=window).mean().shift(1)
    z = theta_daily - ma
    return z.dropna(how="any")
