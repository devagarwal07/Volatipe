# download_data.ps1 - Download raw market data
param(
    [string[]]$Symbols = @("NIFTY50.NS"),
    [string]$SymbolsFile,
    [int]$Days = 365,
    [string]$StartDate,
    [string]$EndDate,
    [int]$MaxRetries = 2,
    [switch]$IncludeVIX,
    [switch]$Verbose
)

Write-Host "=== Downloading Market Data ===" -ForegroundColor Green

if (Test-Path "venv\Scripts\Activate.ps1") { & "venv\Scripts\Activate.ps1" }

# Resolve symbols (file overrides inline list)
$resolvedSymbols = @()
if ($SymbolsFile -and (Test-Path $SymbolsFile)) {
    if ($Verbose) { Write-Host "Loading symbols from file: $SymbolsFile" -ForegroundColor Yellow }
    $resolvedSymbols = Get-Content $SymbolsFile | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object { $_.Trim() }
} else {
    $resolvedSymbols = $Symbols
}
if (-not $resolvedSymbols -or $resolvedSymbols.Count -eq 0) { Write-Error "No symbols provided"; exit 1 }

# Ensure raw data directory
if (-not (Test-Path "data/raw")) { New-Item -ItemType Directory -Path "data/raw" | Out-Null }

$symbolsCsv = ($resolvedSymbols -join ',')
if ($Verbose) { Write-Host "Symbols: $symbolsCsv" -ForegroundColor Cyan }

$startParam = $StartDate
$endParam = $EndDate
if ($startParam -and -not $endParam) { $endParam = (Get-Date).ToString('yyyy-MM-dd') }

$py = @"
import yfinance as yf, os, sys, pandas as pd, time
from datetime import datetime, timedelta

symbols = "${symbolsCsv}".split(',')
start_override = "${startParam}"
end_override = "${endParam}"
days_fallback = ${Days}
max_retries = ${MaxRetries}

end = datetime.now()
if end_override:
    try:
        end = datetime.fromisoformat(end_override)
    except Exception:
        print(f"Invalid EndDate '{end_override}', using now()")

if start_override:
    try:
        start = datetime.fromisoformat(start_override)
    except Exception:
        print(f"Invalid StartDate '{start_override}', using end - {days_fallback} days")
        start = end - timedelta(days=days_fallback)
else:
    start = end - timedelta(days=days_fallback)

os.makedirs('data/raw', exist_ok=True)
print(f"Date range: {start.date()} -> {end.date()}")
print(f"Total symbols: {len(symbols)}")

def download_with_retry(ticker, retries):
    for attempt in range(retries+1):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end)
            return df, None
        except Exception as e:
            err = str(e)
            time.sleep(1 + attempt)
    return None, err

summary = []
for s in symbols:
    root = s.replace('.NS','')
    print(f"Downloading {s} ...", end=' ')
    df, err = download_with_retry(s, max_retries)
    if df is None or df.empty:
        print('failed' + (f" ({err})" if err else ''))
        summary.append((s, 0))
        continue
    out = f"data/raw/{root}_stock_data.csv"
    try:
        df.to_csv(out)
        print(f"saved {len(df)} rows -> {out}")
        summary.append((s, len(df)))
    except Exception as e:
        print(f"save error: {e}")
        summary.append((s, -1))

print("\nSummary:")
ok = [s for s,c in summary if c>0]
fail = [s for s,c in summary if c<=0]
print(f"  Success: {len(ok)}; Failed: {len(fail)}")
if fail:
    print(f"  Failed symbols: {fail}")

 # VIX block appended later
"@

# Inject VIX flag and append VIX download block separately to avoid PowerShell parsing issues
$includeVixFlag = if ($IncludeVIX) { 1 } else { 0 }
$vixBlock = @"
include_vix = $includeVixFlag
if include_vix:
    try:
        vix = yf.Ticker('^INDIAVIX').history(start=start, end=end)
        if not vix.empty:
            vix.to_csv('data/raw/india_vix_data.csv')
            print(f"Saved VIX {len(vix)} rows")
    except Exception as e:
        print('VIX error', e)
"@
$py = $py + "`n" + $vixBlock
$py | python
Write-Host "=== Download complete ===" -ForegroundColor Green
