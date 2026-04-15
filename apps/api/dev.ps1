# DeepShield — FastAPI gateway local dev (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "[api] Creating venv..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "[api] Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }

$env:RUNPOD_INFERENCE_URL = "http://localhost:8000"

Write-Host "[api] http://localhost:8080  (Ctrl+C to stop)" -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
