#!/usr/bin/env python
"""
Interactive test script for model predictions.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# IMPORTANT: Set environment variable to enable debug logging
os.environ['DEBUG'] = '1'

def load_models_directly():
    """Load models directly from file"""
    models = {}
    
    # Load GARCH model
    garch_path = Path('models/garch_ensemble.pkl')
    if garch_path.exists():
        print(f"Loading GARCH model from {garch_path}...")
        with open(garch_path, 'rb') as f:
            models['garch'] = pickle.load(f)
        print("✓ GARCH model loaded")
    
    # Load HAR model
    har_path = Path('models/saved/har_ensemble.pkl')
    if har_path.exists():
        print(f"Loading HAR model from {har_path}...")
        with open(har_path, 'rb') as f:
            har_model = pickle.load(f)
            models['har'] = har_model
        print("✓ HAR model loaded")
        
        # Check for historical data
        if hasattr(har_model, 'historical_data'):
            print(f"✓ HAR model has historical data with {len(har_model.historical_data)} records")
        else:
            print("✗ HAR model missing historical data")
            
            # Try loading historical data
            cache_file = Path('models/cache/historical_volatility.parquet')
            if cache_file.exists():
                print(f"Loading historical data from {cache_file}...")
                try:
                    hist_data = pd.read_parquet(cache_file)
                    har_model.historical_data = hist_data
                    print(f"✓ Added historical data with {len(hist_data)} records")
                    
                    # Save enhanced model
                    with open(har_path, 'wb') as f:
                        pickle.dump(har_model, f)
                    print(f"✓ Saved enhanced model to {har_path}")
                    
                    # Also save to API path
                    api_path = Path('models/har_ensemble.pkl')
                    with open(api_path, 'wb') as f:
                        pickle.dump(har_model, f)
                    print(f"✓ Saved enhanced model to {api_path}")
                except Exception as e:
                    print(f"✗ Error loading historical data: {e}")
    
    return models

def predict_har_custom(models, horizon=1, symbol='RELIANCE'):
    """Custom implementation of predict_har for testing"""
    har = models.get('har')
    if not har:
        print("✗ HAR model not loaded")
        return None
    
    print(f"Making HAR prediction for {symbol} with horizon {horizon}")
    
    # Check for historical data
    if hasattr(har, 'historical_data'):
        print("✓ HAR model has historical data")
        hist_data = har.historical_data
        
        # Filter for symbol
        symbol_data = hist_data[hist_data['symbol'] == symbol].sort_index()
        print(f"Found {len(symbol_data)} records for {symbol}")
        
        if len(symbol_data) > 0:
            # Extract volatility series
            rv_cols = [col for col in symbol_data.columns if col.startswith('rv_')]
            if rv_cols:
                target_col = rv_cols[0]
                print(f"Using column {target_col} for prediction")
                rv_series = symbol_data[target_col].dropna()
                print(f"Series length: {len(rv_series)}")
                
                # Get returns if available
                returns = symbol_data['ret'] if 'ret' in symbol_data.columns else None
                if returns is not None:
                    print("✓ Returns data available")
                
                # Make prediction
                try:
                    print("Making HAR prediction...")
                    har_preds = har.predict(rv_series, returns=returns, horizon=horizon)
                    print(f"Prediction type: {type(har_preds)}")
                    if isinstance(har_preds, dict):
                        print(f"Keys: {list(har_preds.keys())}")
                        if 'ensemble' in har_preds:
                            print(f"Ensemble value: {har_preds['ensemble']}")
                            return har_preds
                    else:
                        print(f"Prediction: {har_preds}")
                        return {'ensemble': har_preds, 'using_historical_data': True}
                except Exception as e:
                    print(f"✗ Error making prediction: {e}")
        else:
            print(f"✗ No data found for symbol {symbol}")
    else:
        print("✗ HAR model missing historical data")
    
    # Fall back to GARCH if needed
    print("Falling back to GARCH...")
    if 'garch' in models:
        try:
            garch_out = models['garch'].forecast(horizon=1)
            print(f"GARCH fallback: {garch_out['ensemble']}")
            return {
                'ensemble': garch_out['ensemble'],
                'note': 'Falling back to GARCH (HAR-RV prediction failed)'
            }
        except Exception as e:
            print(f"✗ GARCH fallback failed: {e}")
    
    return {'error': 'Failed to make HAR-RV prediction'}

def main():
    """Main test function"""
    print("=== Model Prediction Test ===")
    
    # Load models directly
    models = load_models_directly()
    
    # Test prediction
    symbols = ['RELIANCE', 'INFY', 'SBIN', '^NSEI']
    horizons = [1, 5, 10]
    
    for symbol in symbols:
        print(f"\n=== Testing {symbol} ===")
        for horizon in horizons:
            print(f"\nHorizon: {horizon}")
            result = predict_har_custom(models, horizon=horizon, symbol=symbol)
            if result and 'ensemble' in result:
                print(f"Final prediction: {result['ensemble']}")
            else:
                print("✗ Prediction failed")

if __name__ == "__main__":
    main()
