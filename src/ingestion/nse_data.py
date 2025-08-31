"""NSE & Yahoo Finance data ingestion module.
Skeleton implementation: fetches OHLCV for indices & equities via yfinance
and leaves placeholders for direct NSE scraping / API logic (session mgmt, rate limiting).
"""
from __future__ import annotations
import argparse
from datetime import datetime, date
from typing import List, Optional
import pandas as pd
import yfinance as yf
from pathlib import Path
from ..utils.config_loader import config_loader
from ..utils.logging import get_logger
from ..utils.data_utils import save_parquet

logger = get_logger(__name__)

INDEX_SYMBOL_MAP = {
    "^NSEI": "^NSEI",
    "^CNXIT": "^CNXIT",
    "^NSEBANK": "^NSEBANK",
    "^CNXPHARMA": "^CNXPHARMA",
}


def fetch_yf(symbols: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = {}
    for sym in symbols:
        try:
            logger.info(f"Downloading {sym} from yfinance")
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                logger.warning(f"No data for {sym}")
                continue
            df.columns = [c.lower() for c in df.columns]
            df['symbol'] = sym
            data[sym] = df
        except Exception as e:  # noqa
            logger.exception(f"Failed to download {sym}: {e}")
    if not data:
        return pd.DataFrame()
    out = pd.concat(data.values(), axis=0)
    out.index.name = 'date'
    return out


def save_dataset(df: pd.DataFrame, name: str, root: Path):
    if df.empty:
        logger.warning(f"Empty dataframe for {name}, skipping save")
        return
    path = root / f"{name}.parquet"
    save_parquet(df, path)
    logger.info(f"Saved {name} -> {path}")


def main(start_date: str, end_date: Optional[str]):
    cfg = config_loader.load("config.yaml")
    symbols_idx = cfg['data']['symbols']['indices']
    symbols_eq = cfg['data']['symbols']['equities_top10']

    raw_dir = Path(cfg['data']['root_dir']) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    idx_df = fetch_yf(symbols_idx, start_date, end_date)
    save_dataset(idx_df, "indices_ohlcv", raw_dir)

    eq_df = fetch_yf(symbols_eq, start_date, end_date)
    save_dataset(eq_df, "equities_ohlcv", raw_dir)

    # TODO: Add direct NSE fetch (session cookies, headers, rate limit, fallback)
    # TODO: Add FII/DII, options chain ingestion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=False, default="2018-01-01")
    parser.add_argument("--end-date", required=False, default=None)
    args = parser.parse_args()
    end = args.end_date or date.today().isoformat()
    main(args.start_date, end)
