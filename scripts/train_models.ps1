# train_models.ps1 - Train GARCH + HAR-RV + (optional) LSTM & Transformer models with MLflow logging
param(
    [string]$FeaturesFile = 'data/processed/engineered_features.parquet',
    [string]$Symbol = '',
    [int]$HarHorizon = 5,
    [int]$LstmEpochs = 15,
    [int]$TransformerEpochs = 15,
    [switch]$SkipDeep,
    [switch]$Save,
    [switch]$Verbose
)

Write-Host '=== Training Models ===' -ForegroundColor Green
if (Test-Path "venv\Scripts\Activate.ps1") { & "venv\Scripts\Activate.ps1" }

if (-not (Test-Path $FeaturesFile)) { Write-Error "Features file not found: $FeaturesFile"; exit 1 }

$VerboseFlag = if ($Verbose) { 1 } else { 0 }

$py = @"
import os, pandas as pd, numpy as np, math
from datetime import datetime
import mlflow, torch
from sklearn.preprocessing import StandardScaler
from src.models.garch_model import GARCHEnsemble, GARCHSpec
from src.models.har_rv_model import HARRVEnsemble, create_default_har_configs
from src.models.lstm_model import LSTMVolatilityModel, LSTMConfig
from src.models.transformer_model import TransformerVolatilityModel, TransformerConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

features = pd.read_parquet(r'$FeaturesFile')
if 'symbol' not in features.columns:
    raise SystemExit('symbol column missing in features')

symbol_override = '$Symbol'.strip().upper()
if symbol_override and symbol_override in features['symbol'].unique():
    primary_symbol = symbol_override
else:
    primary_symbol = features['symbol'].unique()[0]
sub = features[features['symbol']==primary_symbol].sort_index()
if 'rv_20d' in sub.columns:
    target = sub['rv_20d']
else:
    # fallback: realized volatility proxy via rolling std of returns
    if 'ret' not in sub.columns:
        raise SystemExit('No volatility proxy or returns present')
    target = sub['ret'].rolling(20).std() * (252**0.5)

returns = sub['ret'] if 'ret' in sub.columns else None
rv_series = target.dropna()

if len(rv_series) < 100:
    raise SystemExit('Insufficient data for training')

train_cut = int(len(rv_series)*0.8)
train_rv = rv_series.iloc[:train_cut]

results = {}

# GARCH ensemble (minimal spec)
garch_specs = [GARCHSpec(type='garch', p=1, q=1), GARCHSpec(type='egarch', p=1, o=1, q=1)]
try:
    garch_model = GARCHEnsemble(garch_specs, selection='aic').fit(train_rv)
    garch_forecast = garch_model.forecast(horizon=1)['ensemble']
    try:
        # If garch_forecast is a Series/array-like
        last_val = garch_forecast[-1]
    except Exception:
        # Scalar fallback
        last_val = garch_forecast if isinstance(garch_forecast, (int, float)) else math.nan
    results['garch_last_forecast'] = float(last_val) if not math.isnan(last_val) else math.nan
except Exception as e:
    logger.error(f'GARCH training failed: {e}')
    garch_model = None
    garch_forecast = []

# HAR-RV ensemble
har_configs = create_default_har_configs()
try:
    har_ens = HARRVEnsemble(har_configs).fit(train_rv, returns=returns.iloc[:train_cut] if returns is not None else None)
    har_preds = har_ens.predict(train_rv, returns=returns.iloc[:train_cut] if returns is not None else None, horizon=$HarHorizon)
    if isinstance(har_preds, dict):
        ens_series = har_preds.get('ensemble')
    else:
        ens_series = har_preds  # Backwards compatibility if single series returned
    if ens_series is not None and hasattr(ens_series, '__len__') and len(ens_series)>0:
        results['har_pred_first'] = float(ens_series.iloc[0])
except Exception as e:
    logger.error(f'HAR-RV training failed: {e}')
    har_ens = None
    har_preds = None

# Deep models
skip_deep = bool($SkipDeep)
lstm_model = None
transformer_model = None
if not skip_deep:
    try:
        df_sub = sub.copy().dropna()
        vol_col = 'rv_20d' if 'rv_20d' in df_sub.columns else target.name
        feat_cols = [c for c in df_sub.columns if c not in ['symbol'] and not c.startswith('rv_')]
        if not feat_cols:
            raise ValueError('No feature columns for deep models')
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df_sub[feat_cols])
        y_all = df_sub[vol_col].shift(-1).dropna()
        X_all = X_all[:len(y_all)]
        split = int(len(X_all)*0.8)
        X_train, X_test = X_all[:split], X_all[split:]
        y_train, y_test = y_all.values[:split], y_all.values[split:]

        # LSTM
        lstm_cfg = LSTMConfig(input_size=min(32, X_train.shape[1]), hidden_size=64, num_layers=1, dropout=0.1, bidirectional=False, attention=False)
        lstm_model = LSTMVolatilityModel(lstm_cfg)
        lstm_model.fit(X_train[:, :lstm_cfg.input_size], y_train, epochs=$LstmEpochs)
        lstm_pred = lstm_model.predict(X_test[:, :lstm_cfg.input_size])
        if len(lstm_pred)==len(y_test):
            results['lstm_mse'] = float(np.mean((y_test - lstm_pred)**2))
    except Exception as e:
        logger.error(f'LSTM training failed: {e}')
        lstm_model = None

    try:
        trans_cfg = TransformerConfig(input_dim=min(32, X_train.shape[1]), d_model=64, n_head=4, num_layers=2, d_ff=128, dropout=0.1, max_seq_length=128, output_dim=1)
        transformer_model = TransformerVolatilityModel(trans_cfg, sequence_length=20)
        transformer_model.fit(X_train[:, :trans_cfg.input_dim], y_train, epochs=$TransformerEpochs)
        tr_pred = transformer_model.predict(X_test[:, :trans_cfg.input_dim])
        if len(tr_pred)==len(y_test):
            results['transformer_mse'] = float(np.mean((y_test - tr_pred)**2))
    except Exception as e:
        logger.error(f'Transformer training failed: {e}')
        transformer_model = None

mlflow.set_experiment('volatility_india')
with mlflow.start_run(run_name='full_pipeline'):
    mlflow.log_param('primary_symbol', primary_symbol)
    mlflow.log_param('garch_models', len(garch_specs))
    mlflow.log_param('har_models', len(har_configs))
    mlflow.log_param('deep_enabled', not skip_deep)
    for k,v in results.items():
        if isinstance(v, (int,float)) and not math.isnan(v):
            mlflow.log_metric(k, v)

    os.makedirs('models/saved', exist_ok=True)
    import pickle
    if garch_model:
        with open('models/saved/garch_ensemble.pkl','wb') as f: pickle.dump(garch_model, f)
        mlflow.log_artifact('models/saved/garch_ensemble.pkl')
        # Copy to API expected path
        import shutil
        try:
            shutil.copy('models/saved/garch_ensemble.pkl','models/garch_ensemble.pkl')
        except Exception:
            pass
    if har_ens:
        with open('models/saved/har_ensemble.pkl','wb') as f: pickle.dump(har_ens, f)
        mlflow.log_artifact('models/saved/har_ensemble.pkl')
    if lstm_model:
        torch.save(lstm_model.model.state_dict(), 'models/saved/lstm_model.pt')
        mlflow.log_artifact('models/saved/lstm_model.pt')
    if transformer_model:
        torch.save(transformer_model.model.state_dict(), 'models/saved/transformer_model.pt')
        mlflow.log_artifact('models/saved/transformer_model.pt')

print('Training complete.')
"@

$py | python
Write-Host '=== Training Complete ===' -ForegroundColor Green
