from pathlib import Path
import pandas as pd
import pickle
from typing import Any, List
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=True)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_pickle(obj: Any, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csv(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=True, index_col=0)


# ---------------------------------------------------------------------------
# Domain-specific helpers
# ---------------------------------------------------------------------------
def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column names to lowercase expected by feature builders."""
    rename_map = {
        'Date': 'date', 'Datetime': 'date',
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'close',
        'Volume': 'volume'
    }
    cols_lower = {c: rename_map.get(c, c).lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    return df


def load_all_data(raw_dir: str | Path) -> pd.DataFrame:
    """Load all per-symbol raw CSV files from a directory and stack into one DataFrame.

    Expected filename pattern: <SYMBOL>_* .csv where SYMBOL becomes the symbol identifier.
    Required columns (case-insensitive): Date, Open, High, Low, Close, Volume.
    Returns a multi-symbol stacked frame with index = datetime (sorted) and column 'symbol'.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in raw_dir.glob('*.csv'):
        try:
            symbol_part = csv_path.stem.replace('_stock_data', '')
            # Some symbols may contain caret or dots – keep as-is
            symbol = symbol_part
            df = pd.read_csv(csv_path, parse_dates=['Date'], infer_datetime_format=True)
            df = _normalize_ohlcv_columns(df)
            if 'date' not in df.columns:
                logger.warning(f"Skipping {csv_path.name}: missing date column after normalization")
                continue
            required = {'open', 'high', 'low', 'close', 'volume'}
            if not required.issubset(df.columns):
                logger.warning(f"Skipping {csv_path.name}: missing required columns {required - set(df.columns)}")
                continue
            df = df.set_index('date').sort_index()
            df['symbol'] = symbol.replace('.NS', '').replace('.csv', '')  # lightly clean
            frames.append(df[['open', 'high', 'low', 'close', 'volume', 'symbol']])
        except Exception as e:
            logger.warning(f"Failed to load {csv_path.name}: {e}")
            continue

    if not frames:
        logger.warning("No raw data files loaded.")
        return pd.DataFrame(columns=['open','high','low','close','volume','symbol'])

    all_df = pd.concat(frames).sort_index()
    # Drop duplicate index-symbol combos if any
    all_df = all_df[~all_df.index.duplicated(keep='last')]
    return all_df

