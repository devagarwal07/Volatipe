#!/usr/bin/env python
"""Simple debug script to test model loading and forecasting outside API context"""
import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models, predict_garch
from src.utils.data_utils import load_pickle

def main():
    """Test model loading and forecasting"""
    print("=== Testing Model Loading ===")
    models = {}
    
    # Try direct pickle load
    try:
        garch_path = Path('models/garch_ensemble.pkl')
        if garch_path.exists():
            with open(garch_path, 'rb') as f:
                garch = pickle.load(f)
            models['garch_direct'] = garch
            print(f"Direct GARCH load: {type(garch)}")
        else:
            print(f"GARCH model not found at {garch_path}")
    except Exception as e:
        print(f"Error loading GARCH directly: {e}")
    
    # Try using load_models
    try:
        loaded_models = load_models()
        print(f"Models loaded via load_models(): {list(loaded_models.keys())}")
        for name, model in loaded_models.items():
            models[name] = model
            print(f"Model {name}: {type(model)}")
    except Exception as e:
        print(f"Error in load_models(): {e}")
    
    # Try forecasting
    if 'garch' in models:
        try:
            print("\n=== Testing Forecast ===")
            forecast = predict_garch(models, horizon=3)
            print(f"Forecast result: {forecast}")
        except Exception as e:
            print(f"Error forecasting: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
