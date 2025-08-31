# train_all_stocks.ps1 - Train and evaluate models for all available stocks
param(
    [string]$FeaturesFile = 'data/processed/engineered_features.parquet',
    [int]$HarHorizon = 5,
    [int]$LstmEpochs = 15,
    [int]$TransformerEpochs = 15,
    [switch]$SkipDeep,
    [switch]$Save,
    [switch]$Verbose
)

Write-Host '=== Training Models for All Stocks ===' -ForegroundColor Green
if (Test-Path "venv\Scripts\Activate.ps1") { & "venv\Scripts\Activate.ps1" }

# Step 1: Check for features file, create if it doesn't exist
if (-not (Test-Path $FeaturesFile)) {
    Write-Host "Features file not found: $FeaturesFile" -ForegroundColor Yellow
    Write-Host "Running feature engineering to create it..." -ForegroundColor Cyan
    & .\scripts\feature_engineering.ps1
    if (-not (Test-Path $FeaturesFile)) {
        Write-Error "Failed to create features file"
        exit 1
    }
}

# Step 2: Get list of all available stocks
Write-Host "Getting list of available stocks..." -ForegroundColor Cyan

$py_get_symbols = @"
import pandas as pd
try:
    features = pd.read_parquet(r'$FeaturesFile')
    symbols = sorted(features['symbol'].unique())
    print(','.join(symbols))
except Exception as e:
    print(f"Error: {str(e)}")
    exit(1)
"@

$stocks = $py_get_symbols | python
if (-not $stocks) {
    Write-Error "Failed to get stock symbols from features file"
    exit 1
}

$stocksList = $stocks -split ','
Write-Host "Found $($stocksList.Count) stocks: $stocks" -ForegroundColor Green

# Step 3: Create results tracking directory
$resultsDir = "results/benchmarks"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

# Step 4: Train and evaluate for each stock
$summary = @{}
foreach ($stock in $stocksList) {
    Write-Host "`n====== Training and Evaluating for $stock ======" -ForegroundColor Magenta
    
    # Train models for this stock
    Write-Host "Training models for $stock..." -ForegroundColor Cyan
    & .\scripts\train_models.ps1 -FeaturesFile $FeaturesFile -Symbol $stock -HarHorizon $HarHorizon `
        -LstmEpochs $LstmEpochs -TransformerEpochs $TransformerEpochs `
        -SkipDeep:$SkipDeep -Save:$Save -Verbose:$Verbose
    
    # Now evaluate the models
    Write-Host "Evaluating models for $stock..." -ForegroundColor Cyan
    
    $py_evaluate = @"
import os
import pandas as pd
import numpy as np
import mlflow
import pickle
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.models.predict import load_models, predict_garch, predict_har
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Load features and filter for current stock
features = pd.read_parquet(r'$FeaturesFile')
symbol = '$stock'
stock_data = features[features['symbol'] == symbol].sort_index()

# Create train/test split
if 'rv_20d' in stock_data.columns:
    target = stock_data['rv_20d']
else:
    # fallback: realized volatility proxy via rolling std of returns
    target = stock_data['ret'].rolling(20).std() * (252**0.5)

# Use the last 20% for testing
test_size = int(len(target) * 0.2)
test_data = target.iloc[-test_size:].dropna()

# Load trained models
models = load_models()
results = {'symbol': symbol, 'test_size': len(test_data)}

# Evaluate GARCH
try:
    if 'garch' in models:
        # Get predictions (one step at a time for proper evaluation)
        garch_preds = []
        for i in range(min(len(test_data), 30)):  # Limit to 30 days to avoid long runtimes
            idx = test_data.index[i]
            pred = predict_garch(models, horizon=1)
            if isinstance(pred, dict) and 'ensemble' in pred:
                garch_preds.append(pred['ensemble'])
            
        if garch_preds:
            garch_series = pd.Series(garch_preds, index=test_data.index[:len(garch_preds)])
            results['garch_mse'] = mean_squared_error(test_data[:len(garch_preds)], garch_series)
            results['garch_mae'] = mean_absolute_error(test_data[:len(garch_preds)], garch_series)
            results['garch_mape'] = mean_absolute_percentage_error(test_data[:len(garch_preds)], garch_series)
except Exception as e:
    logger.error(f"GARCH evaluation failed: {str(e)}")
    results['garch_error'] = str(e)

# Evaluate HAR-RV
try:
    if 'har' in models:
        returns = stock_data['ret'] if 'ret' in stock_data.columns else None
        har_preds = []
        for i in range(min(len(test_data), 30)):  # Limit to 30 days
            pred = predict_har(models, horizon=1)
            if isinstance(pred, dict) and 'ensemble' in pred:
                har_preds.append(pred['ensemble'])
            
        if har_preds:
            har_series = pd.Series(har_preds, index=test_data.index[:len(har_preds)])
            results['har_mse'] = mean_squared_error(test_data[:len(har_preds)], har_series)
            results['har_mae'] = mean_absolute_error(test_data[:len(har_preds)], har_series)
            results['har_mape'] = mean_absolute_percentage_error(test_data[:len(har_preds)], har_series)
except Exception as e:
    logger.error(f"HAR-RV evaluation failed: {str(e)}")
    results['har_error'] = str(e)

# Save results for this stock
results_file = f"results/benchmarks/{symbol}_evaluation.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print results summary
print(f"Symbol: {symbol}")
print(f"Test Size: {results['test_size']} days")
if 'garch_mse' in results:
    print(f"GARCH MSE: {results['garch_mse']:.6f}")
    print(f"GARCH MAE: {results['garch_mae']:.6f}")
    print(f"GARCH MAPE: {results['garch_mape']:.6f}")
if 'har_mse' in results:
    print(f"HAR-RV MSE: {results['har_mse']:.6f}")
    print(f"HAR-RV MAE: {results['har_mae']:.6f}")
    print(f"HAR-RV MAPE: {results['har_mape']:.6f}")
"@

    $py_evaluate | python

    # Track completion
    $summary[$stock] = $true
}

# Step 5: Generate benchmark comparison report
Write-Host "`n====== Generating Benchmark Report ======" -ForegroundColor Green

$py_benchmark = @"
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get all evaluation files
results_dir = Path("results/benchmarks")
eval_files = list(results_dir.glob("*_evaluation.json"))

if not eval_files:
    print("No evaluation files found!")
    exit(1)

# Combine all results
all_results = []
for file in eval_files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_results.append(data)

# Create DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('symbol')

# Calculate average metrics
metrics = ['garch_mse', 'garch_mae', 'garch_mape', 'har_mse', 'har_mae', 'har_mape']
averages = {}
for metric in metrics:
    if metric in results_df.columns:
        averages[metric] = results_df[metric].mean()

# Generate summary report
report_file = results_dir / "benchmark_summary.md"
with open(report_file, 'w') as f:
    f.write("# Volatility Model Benchmark Report\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Average Metrics\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    for metric, value in averages.items():
        f.write(f"| {metric} | {value:.6f} |\n")
    
    f.write("\n\n## Results by Stock\n\n")
    f.write("| Symbol | GARCH MSE | GARCH MAE | GARCH MAPE | HAR-RV MSE | HAR-RV MAE | HAR-RV MAPE |\n")
    f.write("|--------|-----------|-----------|------------|------------|------------|------------|\n")
    
    for symbol in sorted(results_df.index):
        row = results_df.loc[symbol]
        garch_mse = f"{row.get('garch_mse', 'N/A'):.6f}" if 'garch_mse' in row else 'N/A'
        garch_mae = f"{row.get('garch_mae', 'N/A'):.6f}" if 'garch_mae' in row else 'N/A'
        garch_mape = f"{row.get('garch_mape', 'N/A'):.6f}" if 'garch_mape' in row else 'N/A'
        har_mse = f"{row.get('har_mse', 'N/A'):.6f}" if 'har_mse' in row else 'N/A'
        har_mae = f"{row.get('har_mae', 'N/A'):.6f}" if 'har_mae' in row else 'N/A'
        har_mape = f"{row.get('har_mape', 'N/A'):.6f}" if 'har_mape' in row else 'N/A'
        
        f.write(f"| {symbol} | {garch_mse} | {garch_mae} | {garch_mape} | {har_mse} | {har_mae} | {har_mape} |\n")

    f.write("\n\n## Model Comparison\n\n")
    f.write("Summary of models that performed best for each stock:\n\n")
    
    best_models = {}
    for symbol in results_df.index:
        row = results_df.loc[symbol]
        if 'garch_mse' in row and 'har_mse' in row:
            best_model = 'GARCH' if row['garch_mse'] < row['har_mse'] else 'HAR-RV'
            best_models[symbol] = best_model
    
    f.write("| Symbol | Best Model |\n")
    f.write("|--------|------------|\n")
    for symbol, model in best_models.items():
        f.write(f"| {symbol} | {model} |\n")

    # Count model wins
    model_counts = {'GARCH': 0, 'HAR-RV': 0}
    for model in best_models.values():
        model_counts[model] = model_counts.get(model, 0) + 1
    
    f.write("\n\n## Overall Winner\n\n")
    f.write(f"GARCH wins: {model_counts.get('GARCH', 0)}\n")
    f.write(f"HAR-RV wins: {model_counts.get('HAR-RV', 0)}\n")

# Create visualizations
if len(results_df) > 0:
    # MSE comparison
    plt.figure(figsize=(12, 6))
    if 'garch_mse' in results_df.columns and 'har_mse' in results_df.columns:
        comparison = results_df[['garch_mse', 'har_mse']].copy()
        comparison.columns = ['GARCH', 'HAR-RV']
        comparison.plot(kind='bar', title='MSE Comparison by Stock')
        plt.ylabel('Mean Squared Error')
        plt.tight_layout()
        plt.savefig(results_dir / 'mse_comparison.png')
    
    # MAPE comparison
    plt.figure(figsize=(12, 6))
    if 'garch_mape' in results_df.columns and 'har_mape' in results_df.columns:
        comparison = results_df[['garch_mape', 'har_mape']].copy()
        comparison.columns = ['GARCH', 'HAR-RV']
        comparison.plot(kind='bar', title='MAPE Comparison by Stock')
        plt.ylabel('Mean Absolute Percentage Error')
        plt.tight_layout()
        plt.savefig(results_dir / 'mape_comparison.png')

print(f"Benchmark report generated: {report_file}")
print("Visualizations saved to results/benchmarks/")
"@

$py_benchmark | python

Write-Host '=== All Stock Training and Evaluation Complete ===' -ForegroundColor Green
Write-Host "Check the benchmark report in results/benchmarks/benchmark_summary.md" -ForegroundColor Cyan
