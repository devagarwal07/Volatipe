#!/usr/bin/env python
"""
This script builds features for the model training and evaluation pipeline.
It uses the feature engineering logic from src/features/build_features.py
to generate a feature set from the raw data.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to allow imports from src
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.features.build_features import build_feature_frame
from src.utils.data_utils import load_all_data, ensure_dir
from src.utils.logging import get_logger
from src.utils.config_loader import config_loader

logger = get_logger(__name__)

def main():
    """
    Main function to run the feature engineering pipeline.
    """
    logger.info("Starting feature engineering process...")

    # Load configuration (project uses config/config.yaml)
    try:
        config = config_loader.load('config.yaml')
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    feature_cfg = config.get('features', {})
    if not feature_cfg:
        logger.error("Feature configuration not found in config/config.yaml")
        return

    # Load raw data
    data_root = Path(config['data'].get('root_dir', 'data'))
    data_path = data_root / 'raw'
    logger.info(f"Loading raw data from {data_path}...")
    all_data = load_all_data(data_path)

    if all_data.empty:
        logger.error("No data loaded, exiting.")
        return

    logger.info(f"Loaded data for {len(all_data['symbol'].unique())} symbols.")

    # Build features
    logger.info("Building feature frame...")
    feature_df = build_feature_frame(
        all_data,
        config,
        min_symbol_history=30,
        max_nan_col_frac=0.6,
        min_valid_cols_frac=0.7,  # derive min_valid_cols dynamically
        fill=True,
    )

    if feature_df.empty:
        logger.warning("Primary feature build produced 0 rows (likely insufficient history). Retrying with relaxed thresholds (min_symbol_history=5, no row validity constraint).")
        feature_df = build_feature_frame(
            all_data,
            config,
            min_symbol_history=5,
            max_nan_col_frac=0.8,
            min_valid_cols=None,
            min_valid_cols_frac=None,
            fill=True,
        )
        if feature_df.empty:
            logger.error("Relaxed feature build still produced 0 rows. Check raw data sufficiency.")
    
    # Save features
    features_dir = data_root / 'features'
    ensure_dir(features_dir)
    output_path = features_dir / 'features.parquet'
    logger.info(f"Saving features to {output_path}...")
    feature_df.to_parquet(output_path)

    logger.info(f"Successfully built and saved {len(feature_df)} feature rows to {output_path}")
    logger.info("Feature engineering process completed.")

if __name__ == '__main__':
    main()
