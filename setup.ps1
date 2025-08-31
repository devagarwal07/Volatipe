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
    Write-Host "[OK] Virtual environment created successfully" -ForegroundColor Green
    } else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Warning "Virtual environment not found. Run with -CreateVenv to create one."
}

# Install dependencies
if ($InstallDeps) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    
    # Upgrade pip first
    python -m pip install --upgrade pip

    if (Test-Path "requirements.txt") {
        Write-Host "requirements.txt found. Installing from file..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "requirements.txt installation encountered errors. Falling back to manual list."
        } else {
            Write-Host "[OK] Dependencies installed from requirements.txt" -ForegroundColor Green
        }
    }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "requirements.txt")) {
        # Fallback manual installation (reduced list)
        $deps = @(
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            "yfinance>=0.2.0",
            "arch>=6.2.0",
            "statsmodels>=0.14.0",
            "torch>=2.0.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "pydantic>=1.10.0",
            "redis>=4.5.0",
            "mlflow>=2.2.0",
            "ta>=0.10.2",
            "requests>=2.28.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.13.0",
            "pytest>=7.0.0",
            "jupyter>=1.0.0"
        )
        foreach ($dep in $deps) {
            Write-Host "Installing $dep..." -ForegroundColor Cyan
            python -m pip install $dep
        }
    }
    Write-Host "[OK] Dependencies installation completed" -ForegroundColor Green
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
    
    Write-Host "[OK] Directory structure created" -ForegroundColor Green
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
    Write-Host "[OK] .env file created" -ForegroundColor Green
}

Write-Host "=== Setup completed successfully! ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review .env file and adjust settings" -ForegroundColor White
Write-Host "2. Run: python -m pytest tests/ to test installation" -ForegroundColor White
Write-Host "3. Run: python scripts/download_data.py to download data" -ForegroundColor White
Write-Host "4. Start Jupyter: jupyter notebook notebooks/" -ForegroundColor White
