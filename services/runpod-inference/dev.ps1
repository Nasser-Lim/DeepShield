# DeepShield — inference local dev (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "[inference] Creating venv..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "[inference] Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }

$env:DEVICE = "cpu"
$env:UPLOAD_DIR = "$env:TEMP\deepshield\uploads"

Write-Host "[inference] http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
