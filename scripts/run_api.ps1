# run_api.ps1 - Launch FastAPI service
param(
  [string]$ApiHost = '0.0.0.0',
  [int]$Port = 8000,
  [switch]$Reload,
  [switch]$Detach
)
Write-Host '=== Starting API ===' -ForegroundColor Green
if (Test-Path "venv\Scripts\Activate.ps1") { & "venv\Scripts\Activate.ps1" }
Write-Host "Starting uvicorn via python -m" -ForegroundColor Cyan
$argsList = @('uvicorn','src.serving.app:app','--host', $ApiHost, '--port', $Port)
if ($Reload) { $argsList += '--reload' }
if ($Detach) {
  Start-Process -FilePath python -ArgumentList ('-m ' + ($argsList -join ' ')) -WindowStyle Normal
  Write-Host "API started in background (PID shown in new window)" -ForegroundColor Yellow
} else {
  python -m @argsList
}
