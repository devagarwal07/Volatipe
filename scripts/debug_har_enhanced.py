#!/usr/bin/env python
"""
Debug script to test the enhanced HAR-RV model with historical data.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models, predict_har, predict_garch
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Test the enhanced HAR-RV model"""
    print("Loading models...")
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")
    
    # Check HAR model for historical data
    har_model = models.get('har')
    if har_model is None:
        print("ERROR: HAR model not loaded")
        return
        
    print("Checking HAR model for historical data...")
    if hasattr(har_model, 'historical_data'):
        print(f"✓ Historical data found with {len(har_model.historical_data)} records")
        symbols = har_model.historical_data['symbol'].unique()
        print(f"  Symbols available: {', '.join(symbols)}")
    else:
        print("✗ No historical data attached to HAR model")
        
        # Try loading historical data
        print("Attempting to load historical data from cache...")
        try:
            cache_file = Path('models/cache/historical_volatility.parquet')
            if cache_file.exists():
                hist_data = pd.read_parquet(cache_file)
                print(f"✓ Loaded {len(hist_data)} records from cache file")
                
                # Attach to model
                har_model.historical_data = hist_data
                print("✓ Attached historical data to HAR model")
                
                # Save model
                print("Saving enhanced model...")
                import pickle
                with open(Path('models/saved/har_ensemble.pkl'), 'wb') as f:
                    pickle.dump(har_model, f)
                print("✓ Saved enhanced HAR model")
            else:
                print("✗ Cache file not found")
        except Exception as e:
            print(f"✗ Failed to load cache: {str(e)}")
    
    # Test predictions
    symbols_to_test = ['RELIANCE', 'INFY', 'SBIN', '^NSEI']
    
    print("\nGARCH predictions:")
    for symbol in symbols_to_test:
        try:
            pred = predict_garch(models, horizon=5)
            print(f"  {symbol}: {pred['ensemble']:.4f}")
        except Exception as e:
            print(f"  {symbol}: Error - {str(e)}")
    
    print("\nHAR predictions:")
    for symbol in symbols_to_test:
        try:
            pred = predict_har(models, horizon=5, symbol=symbol)
            if isinstance(pred, dict) and 'ensemble' in pred:
                ensemble = pred['ensemble']
                if hasattr(ensemble, 'iloc'):
                    val = ensemble.iloc[0]
                else:
                    val = ensemble
                print(f"  {symbol}: {val:.4f}")
                if 'using_historical_data' in pred:
                    print(f"    Using historical data: {pred['using_historical_data']}")
                if 'note' in pred:
                    print(f"    Note: {pred['note']}")
        except Exception as e:
            print(f"  {symbol}: Error - {str(e)}")

if __name__ == "__main__":
    main()
