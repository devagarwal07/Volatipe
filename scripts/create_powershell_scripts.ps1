# PowerShell Scripts for Windows Environment
# Scripts for project setup, data download, and model training

# Create setup script
@'
# setup.ps1 - Project Setup Script for Windows
param(
    [switch]$CreateVenv,
    [switch]$InstallDeps,
    [switch]$SetupDirs,
    [string]$PythonPath = "python"
)

Write-Host "=== Volatility Prediction System Setup ===" -ForegroundColor Green

# Function to check if command exists
function Test-Command {
    param($CommandName)
    try {
        Get-Command $CommandName -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Check Python installation
if (-not (Test-Command $PythonPath)) {
    Write-Error "Python not found. Please install Python or specify correct path with -PythonPath"
    exit 1
}

$pythonVersion = & $PythonPath --version
Write-Host "Found Python: $pythonVersion" -ForegroundColor Yellow

# Create virtual environment
if ($CreateVenv) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        Write-Host "Virtual environment already exists. Removing..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv"
    }
    
    & $PythonPath -m venv venv
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created successfully" -ForegroundColor Green
    } else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Warning "Virtual environment not found. Run with -CreateVenv to create one."
}

# Install dependencies
if ($InstallDeps) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install core dependencies
    $coreDeps = @(
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0"
    )
    
    foreach ($dep in $coreDeps) {
        Write-Host "Installing $dep..." -ForegroundColor Cyan
        python -m pip install $dep
    }
    
    # Install optional dependencies (with error handling)
    $optionalDeps = @(
        "yfinance>=0.2.0",
        "arch>=5.3.0",
        "ta>=0.10.0",
        "torch>=1.13.0",
        "hmmlearn>=0.2.7",
        "mlflow>=2.0.0",
        "redis>=4.5.0",
        "pytest>=7.0.0",
        "jupyter>=1.0.0",
        "plotly>=5.0.0"
    )
    
    foreach ($dep in $optionalDeps) {
        Write-Host "Installing $dep..." -ForegroundColor Cyan
        python -m pip install $dep
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to install $dep - continuing with other packages"
        }
    }
    
    Write-Host "✓ Dependencies installation completed" -ForegroundColor Green
}

# Setup directory structure
if ($SetupDirs) {
    Write-Host "Setting up directory structure..." -ForegroundColor Yellow
    
    $dirs = @(
        "data\raw",
        "data\processed", 
        "data\external",
        "models\saved",
        "models\checkpoints",
        "logs",
        "reports\figures",
        "reports\tables",
        "config\environments"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force
            Write-Host "Created directory: $dir" -ForegroundColor Cyan
        }
    }
    
    Write-Host "✓ Directory structure created" -ForegroundColor Green
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    
    $envContent = @"
# Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data Configuration
DATA_PATH=./data
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed

# Model Configuration
MODEL_PATH=./models
CHECKPOINT_PATH=./models/checkpoints

# API Configuration
API_HOST=localhost
API_PORT=8000
API_WORKERS=1

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=volatility-prediction

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
"@
    
    Set-Content -Path ".env" -Value $envContent
    Write-Host "✓ .env file created" -ForegroundColor Green
}

Write-Host "=== Setup completed successfully! ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review .env file and adjust settings" -ForegroundColor White
Write-Host "2. Run: python -m pytest tests/ to test installation" -ForegroundColor White
Write-Host "3. Run: python scripts/download_data.py to download data" -ForegroundColor White
Write-Host "4. Start Jupyter: jupyter notebook notebooks/" -ForegroundColor White
'@ | Out-File -FilePath "setup.ps1" -Encoding UTF8

# Create data download script
@'
# download_data.ps1 - Data Download Script
param(
    [string[]]$Symbols = @("NIFTY50.NS", "NIFTYBANK.NS", "RELIANCE.NS", "TCS.NS"),
    [int]$Days = 1095,  # 3 years
    [switch]$IncludeVIX,
    [switch]$Verbose
)

Write-Host "=== Stock Data Download Script ===" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    if ($Verbose) { Write-Host "✓ Virtual environment activated" -ForegroundColor Yellow }
}

# Create data directories
$dataDirs = @("data\raw", "data\processed", "data\external")
foreach ($dir in $dataDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        if ($Verbose) { Write-Host "Created directory: $dir" -ForegroundColor Cyan }
    }
}

# Download stock data
Write-Host "Downloading stock data..." -ForegroundColor Yellow
Write-Host "Symbols: $($Symbols -join ', ')" -ForegroundColor Cyan
Write-Host "Period: $Days days" -ForegroundColor Cyan

$pythonScript = @"
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Configuration
symbols = '$($Symbols -join "','"')'.split(',')
end_date = datetime.now()
start_date = end_date - timedelta(days=$Days)
raw_data_dir = 'data/raw'

print(f'Downloading data from {start_date.date()} to {end_date.date()}')
print(f'Symbols: {symbols}')

# Download each symbol
success_count = 0
failed_symbols = []

for symbol in symbols:
    try:
        print(f'Downloading {symbol}...', end=' ')
        
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if not data.empty:
            # Save to CSV
            filename = f'{symbol.replace(".NS", "")}_stock_data.csv'
            filepath = os.path.join(raw_data_dir, filename)
            data.to_csv(filepath)
            
            print(f'✓ ({len(data)} records)')
            success_count += 1
        else:
            print('✗ (no data)')
            failed_symbols.append(symbol)
            
    except Exception as e:
        print(f'✗ (error: {e})')
        failed_symbols.append(symbol)

print(f'\nDownload completed:')
print(f'  Successful: {success_count}/{len(symbols)}')
if failed_symbols:
    print(f'  Failed: {failed_symbols}')
"@

# Execute Python script
$pythonScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Stock data download completed" -ForegroundColor Green
} else {
    Write-Error "Stock data download failed"
}

# Download VIX data if requested
if ($IncludeVIX) {
    Write-Host "Downloading VIX data..." -ForegroundColor Yellow
    
    $vixScript = @"
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

try:
    # Download India VIX
    end_date = datetime.now()
    start_date = end_date - timedelta(days=$Days)
    
    print(f'Downloading India VIX from {start_date.date()} to {end_date.date()}')
    
    # Try different VIX symbols
    vix_symbols = ['^INDIAVIX', 'INDIA VIX']
    vix_data = None
    
    for symbol in vix_symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if not data.empty:
                vix_data = data
                print(f'✓ Downloaded VIX data using symbol: {symbol}')
                break
        except:
            continue
    
    if vix_data is not None:
        filepath = os.path.join('data/raw', 'india_vix_data.csv')
        vix_data.to_csv(filepath)
        print(f'✓ VIX data saved ({len(vix_data)} records)')
    else:
        print('⚠ VIX data not available from yfinance')
        
except Exception as e:
    print(f'✗ VIX download failed: {e}')
"@
    
    $vixScript | python
}

Write-Host "=== Data download completed ===" -ForegroundColor Green
'@ | Out-File -FilePath "scripts\download_data.ps1" -Encoding UTF8

# Create training script
@'
# train_models.ps1 - Model Training Script
param(
    [string]$Model = "all",  # garch, lstm, ensemble, all
    [string]$Symbol = "NIFTY50",
    [int]$Epochs = 100,
    [switch]$UseGPU,
    [switch]$SaveModel,
    [switch]$Verbose
)

Write-Host "=== Model Training Script ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    if ($Verbose) { Write-Host "✓ Virtual environment activated" -ForegroundColor Yellow }
}

# Check for required data
$dataFile = "data\processed\${Symbol}_engineered_features.csv"
if (-not (Test-Path $dataFile)) {
    Write-Error "Features file not found: $dataFile"
    Write-Host "Please run feature engineering first (02_feature_engineering.ipynb)" -ForegroundColor Yellow
    exit 1
}

Write-Host "Training configuration:" -ForegroundColor Yellow
Write-Host "  Model: $Model" -ForegroundColor Cyan
Write-Host "  Symbol: $Symbol" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "  Use GPU: $UseGPU" -ForegroundColor Cyan
Write-Host "  Save Model: $SaveModel" -ForegroundColor Cyan

# Python training script
$trainingScript = @"
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.models.garch_ensemble import GARCHEnsemble
from src.models.lstm_model import LSTMVolatilityModel
from src.models.ensemble_model import EnsembleVolatilityModel
from src.config.settings import get_config
from src.utils.logging import get_logger

# Configuration
config = get_config()
logger = get_logger(__name__)

print('=== Model Training Started ===')
print(f'Training model: $Model')
print(f'Target symbol: $Symbol')

# Load data
print('Loading data...')
features_file = 'data/processed/${Symbol}_engineered_features.csv'
features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)
print(f'✓ Loaded features: {features_df.shape}')

# Prepare target variable (next day volatility)
target_col = 'vol_simple_20d'  # Use 20-day volatility as target
if target_col not in features_df.columns:
    print(f'⚠ Target column {target_col} not found, using first volatility column')
    vol_cols = [col for col in features_df.columns if 'vol' in col.lower()]
    if vol_cols:
        target_col = vol_cols[0]
    else:
        print('✗ No volatility columns found in features')
        sys.exit(1)

# Prepare data for training
target = features_df[target_col].shift(-1).dropna()  # Next day volatility
features = features_df.iloc[:-1]  # Align features

# Remove target from features if present
feature_cols = [col for col in features.columns if col != target_col]
X = features[feature_cols]

# Split data (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

print(f'Training data: {X_train.shape}')
print(f'Test data: {X_test.shape}')

# Device configuration
device = torch.device('cuda' if $UseGPU.ToString().ToLower() -eq 'true' and torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Training results
results = {}

# Train models based on selection
if '$Model' in ['garch', 'all']:
    print('\n--- Training GARCH Ensemble ---')
    try:
        garch_model = GARCHEnsemble()
        garch_model.fit(y_train)
        
        # Predict
        garch_pred = garch_model.predict(steps=len(y_test))
        
        # Calculate metrics
        mse = np.mean((y_test.values - garch_pred) ** 2)
        mae = np.mean(np.abs(y_test.values - garch_pred))
        
        results['garch'] = {'mse': mse, 'mae': mae}
        print(f'✓ GARCH trained - MSE: {mse:.6f}, MAE: {mae:.6f}')
        
        if $SaveModel.ToString().ToLower() -eq 'true':
            import joblib
            joblib.dump(garch_model, 'models/saved/${Symbol}_garch_model.pkl')
            print('✓ GARCH model saved')
            
    except Exception as e:
        print(f'✗ GARCH training failed: {e}')

if '$Model' in ['lstm', 'all']:
    print('\n--- Training LSTM Model ---')
    try:
        from src.models.lstm_model import LSTMConfig
        
        # LSTM configuration
        lstm_config = LSTMConfig(
            input_size=min(50, X_train.shape[1]),  # Limit features for LSTM
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            sequence_length=30
        )
        
        lstm_model = LSTMVolatilityModel(lstm_config)
        
        # Prepare sequence data for LSTM
        X_train_seq = X_train.iloc[:, :lstm_config.input_size].values
        X_test_seq = X_test.iloc[:, :lstm_config.input_size].values
        
        # Train LSTM
        lstm_model.fit(X_train_seq, y_train.values, epochs=$Epochs, device=device)
        
        # Predict
        lstm_pred = lstm_model.predict(X_test_seq, device=device)
        
        # Calculate metrics
        mse = np.mean((y_test.values - lstm_pred) ** 2)
        mae = np.mean(np.abs(y_test.values - lstm_pred))
        
        results['lstm'] = {'mse': mse, 'mae': mae}
        print(f'✓ LSTM trained - MSE: {mse:.6f}, MAE: {mae:.6f}')
        
        if $SaveModel.ToString().ToLower() -eq 'true':
            torch.save(lstm_model.model.state_dict(), 'models/saved/${Symbol}_lstm_model.pth')
            print('✓ LSTM model saved')
            
    except Exception as e:
        print(f'✗ LSTM training failed: {e}')

if '$Model' in ['ensemble', 'all']:
    print('\n--- Training Ensemble Model ---')
    try:
        ensemble_model = EnsembleVolatilityModel()
        
        # For ensemble, we need trained base models
        # This is a simplified version
        print('⚠ Ensemble training requires pre-trained base models')
        print('Please train individual models first')
        
    except Exception as e:
        print(f'✗ Ensemble training failed: {e}')

# Print results summary
print('\n=== Training Results ===')
for model_name, metrics in results.items():
    print(f'{model_name.upper()}:')
    for metric_name, value in metrics.items():
        print(f'  {metric_name}: {value:.6f}')

# Save results
if results:
    import json
    results_file = f'reports/training_results_{Symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('reports', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'✓ Results saved to {results_file}')

print('=== Training Completed ===')
"@

# Execute training
$trainingScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Model training completed successfully" -ForegroundColor Green
} else {
    Write-Error "Model training failed"
}
'@ | Out-File -FilePath "scripts\train_models.ps1" -Encoding UTF8

# Create API server script
@'
# run_api.ps1 - API Server Script
param(
    [string]$Host = "localhost",
    [int]$Port = 8000,
    [int]$Workers = 1,
    [switch]$Reload,
    [switch]$Debug
)

Write-Host "=== Starting Volatility Prediction API ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Yellow
}

# Check if FastAPI is installed
python -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "FastAPI not installed. Please run setup.ps1 -InstallDeps"
    exit 1
}

Write-Host "Starting API server..." -ForegroundColor Yellow
Write-Host "  Host: $Host" -ForegroundColor Cyan
Write-Host "  Port: $Port" -ForegroundColor Cyan
Write-Host "  Workers: $Workers" -ForegroundColor Cyan
Write-Host "  Reload: $Reload" -ForegroundColor Cyan
Write-Host "  Debug: $Debug" -ForegroundColor Cyan

# Build uvicorn command
$uvicornArgs = @(
    "src.serving.app:app",
    "--host", $Host,
    "--port", $Port.ToString(),
    "--workers", $Workers.ToString()
)

if ($Reload) {
    $uvicornArgs += "--reload"
}

if ($Debug) {
    $uvicornArgs += "--log-level", "debug"
}

# Start server
Write-Host "API will be available at: http://${Host}:${Port}" -ForegroundColor Green
Write-Host "API documentation: http://${Host}:${Port}/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow

python -m uvicorn @uvicornArgs
'@ | Out-File -FilePath "scripts\run_api.ps1" -Encoding UTF8

# Create test script
@'
# run_tests.ps1 - Test Execution Script
param(
    [string]$TestPath = "tests/",
    [switch]$Coverage,
    [switch]$Verbose,
    [string]$Pattern = "*test*.py"
)

Write-Host "=== Running Tests ===" -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    if ($Verbose) { Write-Host "✓ Virtual environment activated" -ForegroundColor Yellow }
}

# Check if pytest is installed
python -c "import pytest" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pytest not installed. Installing..."
    python -m pip install pytest pytest-cov
}

# Build pytest command
$pytestArgs = @($TestPath)

if ($Verbose) {
    $pytestArgs += "-v"
}

if ($Coverage) {
    $pytestArgs += "--cov=src", "--cov-report=html", "--cov-report=term"
}

# Add pattern
$pytestArgs += "-k", $Pattern

Write-Host "Running tests with pattern: $Pattern" -ForegroundColor Yellow
Write-Host "Test path: $TestPath" -ForegroundColor Cyan

# Run tests
python -m pytest @pytestArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All tests passed" -ForegroundColor Green
    
    if ($Coverage) {
        Write-Host "Coverage report generated in htmlcov/" -ForegroundColor Cyan
    }
} else {
    Write-Error "Some tests failed"
}
'@ | Out-File -FilePath "scripts\run_tests.ps1" -Encoding UTF8

Write-Host "✓ PowerShell scripts created successfully" -ForegroundColor Green
Write-Host "Available scripts:" -ForegroundColor Yellow
Write-Host "  setup.ps1 - Project setup and environment configuration" -ForegroundColor Cyan
Write-Host "  scripts\download_data.ps1 - Download stock market data" -ForegroundColor Cyan  
Write-Host "  scripts\train_models.ps1 - Train volatility prediction models" -ForegroundColor Cyan
Write-Host "  scripts\run_api.ps1 - Start the FastAPI server" -ForegroundColor Cyan
Write-Host "  scripts\run_tests.ps1 - Run test suite" -ForegroundColor Cyan
Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Yellow
Write-Host "  .\setup.ps1 -CreateVenv -InstallDeps -SetupDirs" -ForegroundColor White
Write-Host "  .\scripts\download_data.ps1 -IncludeVIX -Verbose" -ForegroundColor White
Write-Host "  .\scripts\train_models.ps1 -Model lstm -Epochs 50 -SaveModel" -ForegroundColor White
Write-Host "  .\scripts\run_api.ps1 -Port 8080 -Reload" -ForegroundColor White
