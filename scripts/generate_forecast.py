#!/usr/bin/env python
"""
Debug script to test creating a new volatility forecast.
This combines GARCH and HAR predictions to generate a more robust forecast.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for importing
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.predict import load_models, predict_garch, predict_har
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_forecast_report(symbol='RELIANCE', horizon=5, output_file=None):
    """
    Create and save a volatility forecast report
    
    Args:
        symbol: Ticker symbol to forecast
        horizon: Forecast horizon (days)
        output_file: Optional path to save JSON report
        
    Returns:
        Dictionary with forecast data
    """
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")
    
    # Dictionary to store forecasts
    report = {
        'symbol': symbol,
        'timestamp': datetime.utcnow().isoformat(),
        'horizon': horizon,
        'forecasts': {}
    }
    
    # Get GARCH forecast
    try:
        garch_out = predict_garch(models, horizon=horizon)
        garch_value = float(garch_out['ensemble'])
        report['forecasts']['garch'] = {
            'value': garch_value,
            'components': {k: v for k, v in garch_out.items() if k != 'weights'}
        }
        if 'weights' in garch_out:
            report['forecasts']['garch']['weights'] = garch_out['weights']
    except Exception as e:
        print(f"GARCH forecast error: {e}")
        report['forecasts']['garch'] = {'error': str(e)}
    
    # Get HAR forecast (if available)
    try:
        har_out = predict_har(models, horizon=horizon)
        har_value = float(har_out['ensemble']) if 'ensemble' in har_out else None
        report['forecasts']['har'] = {
            'value': har_value,
            'components': har_out
        }
    except Exception as e:
        print(f"HAR forecast error: {e}")
        report['forecasts']['har'] = {'error': str(e)}
    
    # Create combined forecast
    valid_forecasts = []
    if 'garch' in report['forecasts'] and 'value' in report['forecasts']['garch']:
        valid_forecasts.append(report['forecasts']['garch']['value'])
    
    if 'har' in report['forecasts'] and 'value' in report['forecasts']['har']:
        valid_forecasts.append(report['forecasts']['har']['value'])
    
    if valid_forecasts:
        combined_value = sum(valid_forecasts) / len(valid_forecasts)
        report['forecasts']['combined'] = {
            'value': combined_value,
            'method': 'equal_weight',
            'count': len(valid_forecasts)
        }
    else:
        report['forecasts']['combined'] = {'error': 'No valid forecasts available'}
    
    # Generate forecast dates
    start_date = datetime.now().date()
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(1, horizon+1)]
    report['forecast_dates'] = dates
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    
    return report


def print_report(report):
    """Print a formatted forecast report"""
    print("\n==== VOLATILITY FORECAST REPORT ====")
    print(f"Symbol: {report['symbol']}")
    print(f"Generated: {report['timestamp']}")
    print(f"Horizon: {report['horizon']} days")
    print("\nForecasts:")
    
    for model, data in report['forecasts'].items():
        if 'value' in data:
            print(f"  {model.upper()}: {data['value']:.4f}")
        else:
            print(f"  {model.upper()}: Error - {data.get('error', 'Unknown error')}")
    
    print("\nForecast Dates:")
    for date in report['forecast_dates']:
        print(f"  {date}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate volatility forecast")
    parser.add_argument("--symbol", "-s", default="RELIANCE", help="Ticker symbol")
    parser.add_argument("--horizon", "-d", type=int, default=5, help="Forecast horizon (days)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    report = create_forecast_report(
        symbol=args.symbol,
        horizon=args.horizon,
        output_file=args.output
    )
    
    print_report(report)
