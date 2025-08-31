from __future__ import annotations
import pandas as pd


def make_vol_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    returns = df['close'].groupby(df['symbol']).pct_change()
    future_vol = returns.groupby(df['symbol']).rolling(horizon).std().droplevel(0)
    future_vol = future_vol.groupby(df['symbol']).shift(-horizon + 1)
    return future_vol
