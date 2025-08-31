"""Feature engineering pipeline.
Reads raw CSVs in data/raw produced by download_data.ps1, stacks them with symbol column,
invokes build_feature_frame, and writes per-symbol engineered feature CSVs
consumable by PowerShell training script (data/processed/<SYMBOL>_engineered_features.csv).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import re
import sys

# Ensure project root on path when executed as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.build_features import build_feature_frame  # noqa: E402
from src.utils.config_loader import config_loader  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

def load_raw_csvs(raw_dir: Path):
    files = list(raw_dir.glob('*_stock_data.csv'))
    if not files:
        logger.warning(f"No raw stock CSVs found in {raw_dir}")
        return pd.DataFrame()
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, parse_dates=['Date'])
        except Exception as e:  # noqa
            logger.error(f"Failed reading {fp}: {e}")
            continue
        # Derive symbol root
        m = re.match(r'(.*)_stock_data\.csv', fp.name, re.IGNORECASE)
        sym = m.group(1).upper() if m else fp.stem.upper()
        df.columns = [c.lower() for c in df.columns]
        expected = {'open','high','low','close','volume'}
        if not expected.issubset(df.columns):
            logger.warning(f"Missing expected columns in {fp.name}; has {df.columns}")
            continue
        keep = ['date','open','high','low','close','volume']
        df = df[keep]
        df['symbol'] = sym
        df = df.set_index('date').sort_index()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames)
    return stacked

def main(symbol_filter: list[str] | None):
    cfg = config_loader.load('config.yaml')
    raw_dir = Path(cfg['data']['root_dir']) / 'raw'
    proc_dir = Path(cfg['data']['root_dir']) / 'processed'
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_csvs(raw_dir)
    if raw.empty:
        logger.error("No raw data to process; aborting")
        return 1
    if symbol_filter:
        raw = raw[raw['symbol'].isin([s.upper() for s in symbol_filter])]
        if raw.empty:
            logger.error("Symbol filter removed all data")
            return 1

    logger.info(f"Building features for {raw['symbol'].nunique()} symbols over {len(raw)} rows")
    feats = build_feature_frame(raw, cfg)
    logger.info(f"Feature frame shape: {feats.shape}")

    # Write per-symbol engineered features
    count = 0
    for sym, g in feats.groupby('symbol'):
        out_path = proc_dir / f"{sym}_engineered_features.csv"
        g.drop(columns=['symbol']).to_csv(out_path)
        logger.info(f"Wrote {out_path} ({g.shape})")
        count += 1
    logger.info(f"Completed feature engineering for {count} symbols")
    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='*', help='Optional subset of symbols to process')
    args = p.parse_args()
    raise SystemExit(main(args.symbols))
