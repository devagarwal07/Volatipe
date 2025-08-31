#!/usr/bin/env python
"""Test script to debug HAR-RV prediction"""
import os
import sys
from pathlib import Path

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models, predict_har
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Test HAR model prediction"""
    print("=== Testing HAR-RV Model Prediction ===")
    
    models = load_models()
    if 'har' not in models:
        print("HAR-RV model not loaded")
        return
    
    print(f"HAR-RV model type: {type(models['har'])}")
    
    try:
        print("\nTrying prediction with horizon=3...")
        preds = predict_har(models, horizon=3)
        print(f"Prediction result keys: {preds.keys() if isinstance(preds, dict) else 'not a dict'}")
        
        if isinstance(preds, dict) and 'ensemble' in preds:
            print(f"Ensemble prediction: {preds['ensemble']}")
            print(f"Ensemble type: {type(preds['ensemble'])}")
            
            if hasattr(preds['ensemble'], 'iloc'):
                print(f"First value: {preds['ensemble'].iloc[0]}")
        else:
            print(f"Full prediction output: {preds}")
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
