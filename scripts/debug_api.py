#!/usr/bin/env python
"""
Debug HAR-RV API issue.
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models, predict_har, predict_garch
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Test HAR-RV prediction function"""
    print("Loading models...")
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")
    
    # Check if HAR model has historical data
    har_model = models.get('har')
    if har_model is not None:
        if hasattr(har_model, 'historical_data'):
            print(f"HAR model has historical data with {len(har_model.historical_data)} records")
        else:
            print("HAR model does not have historical data attached")
            
            # Try loading and attaching historical data
            try:
                print("Loading historical data from cache...")
                cache_file = Path('models/cache/historical_volatility.parquet')
                if cache_file.exists():
                    import pandas as pd
                    hist_data = pd.read_parquet(cache_file)
                    har_model.historical_data = hist_data
                    print(f"Attached historical data with {len(hist_data)} records to HAR model")
                    
                    # Save model back for API to use
                    import pickle
                    with open('models/har_ensemble.pkl', 'wb') as f:
                        pickle.dump(har_model, f)
                    print("Saved updated model for API use")
                else:
                    print("Historical data cache not found")
            except Exception as e:
                print(f"Error loading historical data: {e}")
    
    # Test GARCH prediction as baseline
    try:
        print("\nGARCH prediction:")
        garch_out = predict_garch(models, horizon=5)
        print(f"Result: {garch_out}")
    except Exception as e:
        print(f"GARCH Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test HAR-RV prediction
    try:
        print("\nHAR-RV prediction:")
        har_out = predict_har(models, horizon=5, symbol='RELIANCE')
        print(f"Result type: {type(har_out)}")
        print(f"Result keys: {list(har_out.keys()) if isinstance(har_out, dict) else 'N/A'}")
        
        if isinstance(har_out, dict) and 'ensemble' in har_out:
            ensemble = har_out['ensemble']
            print(f"Ensemble type: {type(ensemble)}")
            print(f"Ensemble value: {ensemble}")
            
            # Extract as float for API
            if hasattr(ensemble, 'iloc'):
                print(f"Converting pandas Series to float...")
                if len(ensemble) > 0:
                    # For multi-step horizon, take average of available forecasts
                    available_steps = min(len(ensemble), 5)
                    float_val = float(ensemble.iloc[:available_steps].mean())
                    print(f"Float value: {float_val}")
                else:
                    print("Empty Series!")
            elif isinstance(ensemble, (int, float)):
                print(f"Already a float/int: {ensemble}")
            else:
                print(f"Unknown ensemble type: {type(ensemble)}")
    except Exception as e:
        print(f"HAR-RV Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
