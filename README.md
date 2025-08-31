# Multi-Scale Adaptive Ensemble (MSAE) Volatility Prediction System

A comprehensive volatility prediction system for Indian stock markets (NSE/BSE) using advanced machine learning techniques.

## 🎯 Project Overview

This system implements state-of-the-art volatility forecasting for Indian equity markets using:

- **GARCH Family Models**: GARCH(1,1), EGARCH, GJR-GARCH with ensemble weighting
- **Deep Learning Models**: LSTM with attention mechanism, Transformer architecture
- **HAR-RV Models**: Heterogeneous Autoregressive Realized Volatility
- **Ensemble Framework**: Adaptive model combination with dynamic weights
- **Indian Market Features**: VIX integration, FII/DII flows, calendar effects
- **Production API**: FastAPI-based serving with real-time predictions

## 🏗️ Architecture

```
MSAE-India/
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data ingestion and processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model implementations
│   ├── serving/         # API and serving layer
│   ├── backtest/        # Backtesting framework
│   ├── monitoring/      # Model monitoring and drift detection
│   └── utils/           # Utilities and helpers
├── notebooks/           # Jupyter notebooks for analysis
├── scripts/             # Automation scripts
├── tests/              # Test suite
├── data/               # Data storage
├── models/             # Saved models
├── reports/            # Analysis reports
└── docker/             # Containerization
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Windows PowerShell (for Windows users)
- 8GB+ RAM recommended
- GPU support optional (for deep learning models)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MSAE-India
```

2. **Windows Setup (PowerShell)**
```powershell
# Setup environment
.etup.ps1 -CreateVenv -InstallDeps -SetupDirs

# Download data
.criptsownload_data.ps1 -IncludeVIX -Verbose

# Train models
.cripts	rain_models.ps1 -Model all -SaveModel

# Start API
.cripts
un_api.ps1 -Reload
```

3. **Linux/Mac Setup**
```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebooks or scripts
jupyter notebook notebooks/
```

### Data Sources

The system supports multiple data sources:

- **yfinance**: NSE/BSE stock data via Yahoo Finance
- **NSE API**: Direct NSE data feeds (when available)
- **Custom CSV**: Options chain and VIX data
- **External APIs**: FII/DII flow data

## 📊 Features

### Core Features

1. **Advanced Volatility Estimators**
   - Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
   - Realized volatility from high-frequency data
   - Implied volatility from options

2. **Technical Indicators**
   - RSI, MACD, Bollinger Bands, Stochastic
   - Volume-based indicators
   - Momentum and trend features

3. **Market Microstructure**
   - Bid-ask spreads, market impact
   - Order flow imbalance
   - High-frequency patterns

4. **Indian Market Specifics**
   - VIX integration and derivatives
   - FII/DII flow analysis
   - Calendar effects (festivals, budget, monsoon)
   - Sector rotation patterns

### Model Portfolio

1. **GARCH Family**
   - GARCH(1,1), EGARCH, GJR-GARCH
   - Information criteria based weighting
   - Regime-dependent parameters

2. **Deep Learning**
   - LSTM with attention mechanism
   - Transformer architecture
   - Sequence-to-sequence modeling

3. **HAR-RV Models**
   - Daily, weekly, monthly volatility components
   - Jump detection and leverage effects
   - Multi-horizon forecasting

4. **Ensemble Methods**
   - Dynamic weight allocation
   - Performance-based model selection
   - Regime-aware combinations

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
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

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=volatility-prediction
```

### Model Configuration

Models are configured via YAML files in `config/`:

```yaml
# config/model_config.yaml
garch:
  models: ['GARCH', 'EGARCH', 'GJR-GARCH']
  max_lag_ar: 3
  max_lag_ma: 3
  
lstm:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  sequence_length: 30
  
ensemble:
  update_frequency: 'weekly'
  min_history: 252
  performance_window: 60
```

## 📚 Notebooks

The system includes comprehensive Jupyter notebooks:

1. **01_data_download.ipynb**: Data acquisition and quality checks
2. **02_feature_engineering.ipynb**: Feature creation and analysis
3. **03_garch_baseline.ipynb**: GARCH model implementation
4. **04_lstm_model.ipynb**: Deep learning model development
5. **05_ensemble_model.ipynb**: Ensemble framework
6. **06_backtest_eval.ipynb**: Performance evaluation

## 🌐 API Usage

### Start the API Server

```powershell
.cripts
un_api.ps1 -Port 8000 -Reload
```

### API Endpoints

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")

# Predict volatility
payload = {
    "symbol": "NIFTY50",
    "horizon": 5,
    "model_type": "ensemble"
}
response = requests.post("http://localhost:8000/predict", json=payload)

# Get regime information
response = requests.post("http://localhost:8000/regime", json={"symbol": "NIFTY50"})
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🧪 Testing

### Run Tests

```powershell
# All tests
.cripts
un_tests.ps1

# With coverage
.cripts
un_tests.ps1 -Coverage

# Specific test pattern
.cripts
un_tests.ps1 -Pattern "*garch*"
```

### Test Coverage

- Unit tests for all model components
- Integration tests for API endpoints
- Data validation tests
- Performance benchmarks

## 📈 Performance

### Backtesting Results

The system provides comprehensive backtesting with:

- Walk-forward validation
- Out-of-sample testing
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Transaction cost modeling for Indian markets

### Model Performance Metrics

- **Accuracy**: RMSE, MAE, MAPE
- **Direction**: Hit rate, directional accuracy
- **Risk**: VaR, CVaR, maximum drawdown
- **Stability**: Model consistency, drift detection

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t MSAE-India .

# Run container
docker run -p 8000:8000 MSAE-India

# Run with GPU support
docker run --gpus all -p 8000:8000 MSAE-India
```

### Docker Compose

```bash
# Start full stack (API + Redis + MLflow)
docker-compose up -d

# Scale API workers
docker-compose up --scale api=3
```

## 🔍 Monitoring

The system includes comprehensive monitoring:

- **Model Drift Detection**: Statistical tests for feature and target drift
- **Performance Tracking**: Real-time model performance metrics
- **Data Quality**: Automated data validation and anomaly detection
- **System Health**: API performance, resource utilization

### MLflow Integration

- Experiment tracking for all model runs
- Model registry for version management
- Performance comparison across models
- Automated model deployment

## 📊 Data Schema

### Input Data Format

```python
# Stock data (OHLCV)
{
    "Date": "2023-01-01",
    "Open": 18500.0,
    "High": 18650.0,
    "Low": 18450.0,
    "Close": 18600.0,
    "Volume": 150000000
}

# VIX data
{
    "Date": "2023-01-01",
    "VIX": 15.25,
    "VIX_Change": 0.5
}
```

### Feature Schema

Features are organized by category:

- **Basic**: Returns, momentum, volatility
- **Technical**: RSI, MACD, Bollinger Bands
- **Advanced**: Regime indicators, cross-asset correlations
- **Market**: VIX derivatives, calendar effects

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards

- Python 3.8+ with type hints
- Black code formatting
- Pytest for testing
- Comprehensive documentation

### Issue Reporting

Use GitHub issues for:
- Bug reports
- Feature requests
- Performance issues
- Documentation improvements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NSE/BSE** for market data access
- **Yahoo Finance** for historical data API
- **PyTorch/TensorFlow** for deep learning frameworks
- **FastAPI** for modern API development
- **MLflow** for experiment tracking

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation in `docs/`
- Review example notebooks in `notebooks/`

---

**Note**: This system is for educational and research purposes. Always validate models thoroughly before using for actual trading decisions. Volatility Prediction System

Comprehensive multi-scale adaptive ensemble (MSAE) volatility prediction framework specialized for Indian equity markets (NSE/BSE). Combines classical econometric GARCH family models, realized volatility (HAR-RV), and deep learning (LSTM + Attention, Transformer) with adaptive regime-aware ensembling.

## Key Features
- Multi-model volatility forecasting: GARCH / EGARCH / GJR-GARCH / TGARCH / HAR-RV
- Deep sequence models: Bidirectional Attention LSTM & Transformer
- Adaptive regime detection (HMM + GMM + market heuristics)
- Indian market specific signals: India VIX, FII/DII flows, policy events, monsoon & festival seasonality
- Rich volatility & technical feature engineering (OHLCV derived, cross-asset, interaction features)
- MLflow tracking & experiment management
- FastAPI microservice with Redis caching & optional API key auth
- Backtesting engine with walk-forward evaluation & transaction cost modeling
- Dockerized deployment & reproducible environment

## Quick Start
```bash
# (Linux/macOS) create environment
bash scripts/setup_environment.sh

# Fetch data
bash scripts/fetch_data.sh

# Train models & run backtest
bash scripts/train_model.sh

# Run API
bash scripts/run_server.sh
```
Windows users can adapt the bash scripts (or use WSL).

## Project Structure
(See repository tree for full breakdown.)

## Configuration
Edit `config/config.yaml` (data sources, symbols, paths) and `config/model.yaml` (hyperparameters & ensemble settings). Dynamic loading via `src.utils.config_loader`.

## API Endpoints (Planned)
- `GET /health`
- `GET /model/status`
- `POST /predict` (supports horizon 1-30)
- `GET /regime`
- `POST /update/data`

## MLflow
Experiments logged under `mlruns/`. Set `MLFLOW_TRACKING_URI` to remote server for production.

## Tests
Run `pytest -q` after implementing additional logic. Target coverage >80%.

## Roadmap (Initial Implementation Stage)
Phase 1: Data ingestion & feature stubs
Phase 2: Core GARCH + HAR-RV models
Phase 3: LSTM/Transformer & ensemble logic
Phase 4: Serving layer & caching, monitoring
Phase 5: Optimization, documentation, hardening

## Disclaimer
This project uses publicly available market data. Ensure compliance with NSE/BSE data usage policies for production deployments.

---
© 2025 MSAE-India Project
