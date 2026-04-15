# DeepShield — Next.js local dev (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[web] Installing dependencies..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) { Write-Error "npm install failed"; exit 1 }

$env:NEXT_PUBLIC_API_URL = "http://localhost:8080"

Write-Host "[web] http://localhost:3000  (Ctrl+C to stop)" -ForegroundColor Green
npm run dev
