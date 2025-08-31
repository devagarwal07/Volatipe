from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)

# --- Volatility estimators ---

def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    return (1.0 / (4.0 * np.log(2))) * (np.log(high / low) ** 2)

def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return 0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)

def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)

def yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, prev_close: pd.Series) -> pd.Series:
    k = 0.34 / (1.34 + (len(open_) + 1) / (len(open_) - 1))
    overnight = np.log(open_ / prev_close) ** 2
    open_close = np.log(close / open_) ** 2
    rs = rogers_satchell(open_, high, low, close)
    return overnight + k * open_close + (1 - k) * rs


def realized_vol(close: pd.Series, windows: List[int]) -> pd.DataFrame:
    returns = close.pct_change()
    out = {}
    for w in windows:
        out[f'rv_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(out)


def add_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    for c in cols:
        for l in lags:
            df[f"{c}_lag{l}"] = df[c].shift(l)
    return df


def build_feature_frame(
    ohlcv: pd.DataFrame,
    cfg: dict,
    min_symbol_history: Optional[int] = None,
    max_nan_col_frac: Optional[float] = None,
    min_valid_cols: Optional[int] = None,
    min_valid_cols_frac: Optional[float] = None,
    fill: bool = True,
) -> pd.DataFrame:
    """Build base feature frame with relaxed NaN handling.

    Steps:
      1. Per-symbol feature computation (rolling metrics will introduce NaNs).
      2. Filter out symbols with insufficient history (min_symbol_history).
      3. Drop columns whose NaN fraction > max_nan_col_frac.
      4. Forward fill then backward fill within each symbol if fill=True.
      5. Keep rows that meet min_valid_cols or min_valid_cols_frac criteria.
    """

    fcfg = cfg.get('features', {})
    # Defaults from config or provided overrides
    min_symbol_history = min_symbol_history or fcfg.get('min_symbol_history', 30)
    max_nan_col_frac = max_nan_col_frac if max_nan_col_frac is not None else fcfg.get('max_nan_col_frac', 0.5)
    min_valid_cols = min_valid_cols or fcfg.get('min_valid_cols')  # can be None
    min_valid_cols_frac = (
        min_valid_cols_frac if min_valid_cols_frac is not None else fcfg.get('min_valid_cols_frac')
    )

    feats = []
    for sym, g in ohlcv.groupby('symbol'):
        g = g.sort_index()
        if len(g) < min_symbol_history:
            logger.debug(f"Skipping {sym}: only {len(g)} rows (< {min_symbol_history})")
            continue
        f = pd.DataFrame(index=g.index)
        f['close'] = g['close']
        f['ret'] = g['close'].pct_change()
        f['parkinson'] = parkinson(g['high'], g['low']).rolling(10).mean()
        f['gk'] = garman_klass(g['open'], g['high'], g['low'], g['close']).rolling(10).mean()
        f['rs'] = rogers_satchell(g['open'], g['high'], g['low'], g['close']).rolling(10).mean()
        prev_close = g['close'].shift(1)
        f['yz'] = yang_zhang(g['open'], g['high'], g['low'], g['close'], prev_close).rolling(10).mean()
        rv_df = realized_vol(g['close'], fcfg['realized_vol_windows'])
        f = f.join(rv_df)
        f['symbol'] = sym
        feats.append(f)

    if not feats:
        logger.warning("No symbols produced features; returning empty frame.")
        return pd.DataFrame()

    full = pd.concat(feats)
    full = add_lags(full, ['close', 'ret'], fcfg['lag_days'])

    # Separate symbol column for operations
    symbol_col = full['symbol']
    feature_cols = [c for c in full.columns if c != 'symbol']
    feat_df = full[feature_cols]

    # 1. Drop columns with too many NaNs
    if max_nan_col_frac is not None:
        col_nan_frac = feat_df.isna().mean()
        to_drop = col_nan_frac[col_nan_frac > max_nan_col_frac].index.tolist()
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} columns with NaN frac > {max_nan_col_frac}: {to_drop[:10]}{'...' if len(to_drop)>10 else ''}")
            feat_df = feat_df.drop(columns=to_drop)

    # 2. Fill missing values within each symbol (ffill then bfill)
    if fill:
        feat_df = feat_df.join(symbol_col)
        feat_df = feat_df.groupby('symbol', group_keys=False).apply(lambda d: d.ffill().bfill())
        symbol_col = feat_df['symbol']
        feat_df = feat_df.drop(columns=['symbol'])

    # 3. Apply row-level validity filter
    if min_valid_cols is not None or min_valid_cols_frac is not None:
        non_null_counts = feat_df.notna().sum(axis=1)
        total_cols = feat_df.shape[1]
        if min_valid_cols is None and min_valid_cols_frac is not None:
            # derive min_valid_cols from fraction
            min_valid_cols = int(np.ceil(min_valid_cols_frac * total_cols))
        if min_valid_cols is not None:
            before = len(feat_df)
            mask = non_null_counts >= min_valid_cols
            feat_df = feat_df[mask]
            symbol_col = symbol_col.loc[feat_df.index]
            logger.info(f"Filtered rows by min_valid_cols={min_valid_cols}: {before} -> {len(feat_df)}")

    # 4. Final cleanup: drop any rows still all-NaN (should be none after fill)
    feat_df = feat_df.dropna(how='all')

    feat_df['symbol'] = symbol_col
    # Reorder columns with symbol first
    cols = ['symbol'] + [c for c in feat_df.columns if c != 'symbol']
    final_df = feat_df[cols]
    logger.info(f"Final feature frame shape: {final_df.shape}")
    return final_df
