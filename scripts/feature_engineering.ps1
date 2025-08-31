# feature_engineering.ps1 - Build engineered features from downloaded raw CSVs
param(
  [string]$RawDir = 'data/raw',
  [string]$OutDir = 'data/processed',
  [switch]$Verbose
)

Write-Host '=== Feature Engineering ===' -ForegroundColor Green
if (Test-Path "venv\Scripts\Activate.ps1") { & "venv\Scripts\Activate.ps1" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$py = @"
import os, pandas as pd, numpy as np, glob, sys
from datetime import datetime

raw_dir = r'$RawDir'
out_dir = r'$OutDir'
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(raw_dir, '*_stock_data.csv')))
if not files:
    print('No raw stock data CSVs found.')
    sys.exit(1)

frames = []
for fp in files:
    sym = os.path.basename(fp).replace('_stock_data.csv','')
    df = pd.read_csv(fp)
    if 'Date' in df.columns:
        df.rename(columns={'Date':'date'}, inplace=True)
    if 'date' not in df.columns:
        print(f'Skipping {fp}: missing date column')
        continue
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # normalize column names
    cols_map = {c:c.lower() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)
    keep = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    if len(keep) < 4:
        print(f'Skipping {sym}: insufficient OHLC columns')
        continue
    df = df[keep]
    df['symbol'] = sym
    frames.append(df)

if not frames:
    print('No usable data.')
    sys.exit(1)

full = pd.concat(frames)
# Ensure deterministic order by symbol then date
full.sort_values(['symbol','date' if 'date' in full.index.names else full.index.name], inplace=True)

# Simple realized vol features
full['ret'] = full.groupby('symbol')['close'].pct_change()
for w in [5,10,20,60]:
    full[f'rv_{w}d'] = full.groupby('symbol')['ret'].transform(lambda s: s.rolling(w).std() * (252**0.5))

# Lag features
for l in [1,2,3,5]:
    full[f'ret_lag{l}'] = full.groupby('symbol')['ret'].shift(l)

full = full.dropna()

out_file = os.path.join(out_dir, 'engineered_features.parquet')
full.to_parquet(out_file)
print(f'Saved features: {full.shape} -> {out_file}')
"@

$py | python
Write-Host '=== Feature Engineering Complete ===' -ForegroundColor Green
