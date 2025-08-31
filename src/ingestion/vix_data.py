"""India VIX data ingestion placeholder.
Currently uses yfinance ^INDIAVIX as proxy. Extend with NSE direct download + parsing.
"""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import pandas as pd
import yfinance as yf
from ..utils.logging import get_logger
from ..utils.config_loader import config_loader
from ..utils.data_utils import save_parquet

logger = get_logger(__name__)


def fetch_vix(start: str, end: str) -> pd.DataFrame:
    sym = "^INDIAVIX"
    df = yf.download(sym, start=start, end=end, progress=False)
    if df.empty:
        logger.warning("No VIX data returned")
        return df
    df.columns = [c.lower() for c in df.columns]
    df['symbol'] = sym
    df.index.name = 'date'
    return df


def main(start_date: str, end_date: str):
    cfg = config_loader.load("config.yaml")
    raw_dir = Path(cfg['data']['root_dir']) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_vix(start_date, end_date)
    if not df.empty:
        save_parquet(df, raw_dir / 'indiavix.parquet')
        logger.info("Saved India VIX data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    a = parser.parse_args()
    main(a.start_date, a.end_date)
