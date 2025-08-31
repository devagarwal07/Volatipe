param(
  [string]$ApiHost = '127.0.0.1',
  [int]$Port = 8013,
  [int]$Horizon = 3,
  [string]$Symbol = 'RELIANCE'
)

$base = 'http://{0}:{1}' -f $ApiHost, $Port
Write-Host "Testing health endpoint at $base/health" -ForegroundColor Cyan
try {
  $health = Invoke-RestMethod -Uri "$base/health" -Method Get -ErrorAction Stop
  Write-Host "Health: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
  Write-Error "Health request failed: $_"; exit 1
}

$bodyObj = @{ horizon = $Horizon; symbol = $Symbol }
$bodyJson = $bodyObj | ConvertTo-Json -Compress
Write-Host "POST /predict with body: $bodyJson" -ForegroundColor Cyan
try {
  $resp = Invoke-RestMethod -Uri "$base/predict" -Method Post -Body $bodyJson -ContentType 'application/json' -ErrorAction Stop
  Write-Host "Predict response:" -ForegroundColor Green
  ($resp | ConvertTo-Json -Depth 5) | Write-Output
} catch {
  Write-Error "Predict request failed: $_"; exit 1
}
