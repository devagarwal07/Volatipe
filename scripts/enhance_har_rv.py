#!/usr/bin/env python
"""
Enhances the HAR-RV model implementation to use historical volatility data
for predictions rather than relying on GARCH as a proxy.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
import sys

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Constants
CACHE_DIR = Path('models/cache')
HIST_DATA_FILE = CACHE_DIR / 'historical_volatility.parquet'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_cache_historical_data():
    """Extract historical volatility data from raw files and cache for HAR-RV predictions"""
    try:
        # Load features file with historical volatility
        features_file = Path('data/processed/engineered_features.parquet')
        if not features_file.exists():
            logger.error("Features file not found, cannot extract historical volatility")
            return False
            
        features = pd.read_parquet(features_file)
        
        # Extract relevant columns for HAR-RV
        vol_cols = [col for col in features.columns if col.startswith('rv_') or col == 'ret']
        needed_cols = ['symbol', 'date'] + vol_cols
        available_cols = [col for col in needed_cols if col in features.columns]
        
        # Check if we have necessary data
        if 'symbol' not in available_cols or len(vol_cols) == 0:
            logger.error("Required volatility columns not found in features file")
            return False
            
        # Extract and save historical data
        hist_data = features[available_cols].copy()
        hist_data.to_parquet(HIST_DATA_FILE)
        
        logger.info(f"Successfully cached historical volatility data for {len(hist_data['symbol'].unique())} symbols")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract historical data: {e}")
        return False

def enhance_har_rv_model():
    """Update the HAR-RV model to use historical data from cache"""
    
    # First, make sure we have historical data cached
    if not HIST_DATA_FILE.exists():
        logger.info("Historical data not cached, extracting now...")
        if not extract_and_cache_historical_data():
            logger.error("Could not extract historical data, HAR-RV enhancement failed")
            return False
    
    # Load the HAR-RV model
    models = load_models()
    if 'har' not in models:
        logger.error("HAR-RV model not loaded, cannot enhance")
        return False
    
    har_model = models['har']
    
    # Add historical data to the model
    try:
        hist_data = pd.read_parquet(HIST_DATA_FILE)
        logger.info(f"Loaded historical data with {len(hist_data)} rows")
        
        # Add historical data attribute to model
        har_model.historical_data = hist_data
        
        # Save enhanced model
        with open(Path('models/saved/har_ensemble_enhanced.pkl'), 'wb') as f:
            pickle.dump(har_model, f)
        
        # Also update the main model path
        with open(Path('models/saved/har_ensemble.pkl'), 'wb') as f:
            pickle.dump(har_model, f)
        
        logger.info("Successfully enhanced HAR-RV model with historical data")
        return True
        
    except Exception as e:
        logger.error(f"Failed to enhance HAR-RV model: {e}")
        return False

def patch_predict_har_function():
    """Create an updated version of predict_har that uses historical data"""
    template = """
def predict_har(models: Dict[str, Any], horizon: int = 1, symbol: str = 'RELIANCE') -> dict:
    \"\"\"Generate forecasts using HAR-RV model if available\"\"\"
    har = models.get('har')
    if not har:
        raise ValueError("HAR-RV model not loaded")
    
    # Check if we have the enhanced HAR model with historical data
    if hasattr(har, 'historical_data'):
        try:
            # Get historical data for the requested symbol
            hist_data = har.historical_data
            symbol_data = hist_data[hist_data['symbol'] == symbol].sort_index()
            
            if len(symbol_data) == 0:
                # Fall back to first available symbol if requested one not found
                available_symbols = hist_data['symbol'].unique()
                if len(available_symbols) > 0:
                    symbol = available_symbols[0]
                    symbol_data = hist_data[hist_data['symbol'] == symbol].sort_index()
                    logger.warning(f"Symbol {symbol} not found in historical data, using {symbol} instead")
            
            if len(symbol_data) > 0:
                # Extract required data for HAR prediction
                rv_cols = [col for col in symbol_data.columns if col.startswith('rv_')]
                if len(rv_cols) > 0:
                    target_col = rv_cols[0]  # Use first RV column
                    rv_series = symbol_data[target_col].dropna()
                    
                    # If returns available, use them
                    returns = symbol_data['ret'] if 'ret' in symbol_data.columns else None
                    
                    # Get HAR prediction using historical data
                    har_preds = har.predict(rv_series, returns=returns, horizon=horizon)
                    
                    # Extract ensemble prediction
                    if isinstance(har_preds, dict) and 'ensemble' in har_preds:
                        result = har_preds
                        result['model_type'] = "HAR-RV"
                        result['using_historical_data'] = True
                        result['data_length'] = len(rv_series)
                        return result
        except Exception as e:
            logger.warning(f"HAR-RV prediction with historical data failed: {e}")
    
    # If we reach here, fall back to the proxy method
    try:
        # Create dummy forecast since we can't run real HAR without historical data
        result = {
            "ensemble": 0.0,
            "model_type": "HAR-RV",
            "note": "Placeholder - HAR-RV requires historical data unavailable in API context"
        }
        
        # If GARCH is available, use its forecast as placeholder
        if 'garch' in models:
            try:
                garch_out = models['garch'].forecast(horizon=1)
                if isinstance(garch_out, dict) and 'ensemble' in garch_out:
                    result['ensemble'] = float(garch_out['ensemble'])
                    result['note'] += " (using GARCH forecast as proxy)"
            except Exception:
                pass
                
        return result
    except Exception as e:
        logger.warning(f"HAR prediction failed: {e}")
        return {"error": str(e), "ensemble": 0.0}
"""
    
    # Save the new function template to a file
    template_path = Path('src/models/predict_har_enhanced.py')
    with open(template_path, 'w') as f:
        f.write("from __future__ import annotations\n")
        f.write("from typing import Any, Dict\n")
        f.write("import pandas as pd\n")
        f.write("from ..utils.logging import get_logger\n\n")
        f.write("logger = get_logger(__name__)\n\n")
        f.write(template)
    
    logger.info(f"Saved enhanced predict_har function to {template_path}")
    return True

if __name__ == "__main__":
    print("Enhancing HAR-RV model implementation...")
    
    # Extract and cache historical volatility data
    if not HIST_DATA_FILE.exists():
        print("Extracting and caching historical volatility data...")
        if extract_and_cache_historical_data():
            print("✓ Historical data cached successfully")
        else:
            print("✗ Failed to cache historical data")
    else:
        print("✓ Historical data already cached")
    
    # Enhance HAR-RV model with historical data
    if enhance_har_rv_model():
        print("✓ HAR-RV model enhanced successfully")
    else:
        print("✗ Failed to enhance HAR-RV model")
    
    # Create patched predict_har function
    if patch_predict_har_function():
        print("✓ Created enhanced predict_har function")
    else:
        print("✗ Failed to create enhanced predict_har function")
        
    print("\nTo complete the enhancement:")
    print("1. Replace the predict_har function in src/models/predict.py with the enhanced version")
    print("2. Restart the API to use the enhanced HAR-RV model")
    print("3. Run the benchmarks again to see the difference between GARCH and HAR-RV")
