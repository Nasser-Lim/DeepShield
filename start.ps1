# DeepShield — 전체 스택 로컬 실행 (Windows PowerShell)
# 실행: .\start.ps1
# 종료: Ctrl+C 한 번 (모든 job 자동 정리)

$root = $PSScriptRoot

Write-Host ""
Write-Host "=== DeepShield Local Stack ===" -ForegroundColor Cyan
Write-Host "  inference  ->  http://localhost:8000/docs"
Write-Host "  api        ->  http://localhost:8080/docs"
Write-Host "  web        ->  http://localhost:3000"
Write-Host ""
Write-Host "Ctrl+C 로 전체 종료" -ForegroundColor DarkGray
Write-Host ""

# ── 각 서비스를 Background Job 으로 실행 ─────────────────────────────────

$inferJob = Start-Job -Name "inference" -ScriptBlock {
    param($root)
    Set-Location "$root\services\runpod-inference"
    $env:DEVICE     = "cpu"
    $env:UPLOAD_DIR = "$env:TEMP\deepshield\uploads"
    .\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
} -ArgumentList $root

# inference 가 먼저 바인딩될 시간을 줌
Start-Sleep -Seconds 3

$apiJob = Start-Job -Name "api" -ScriptBlock {
    param($root)
    Set-Location "$root\apps\api"
    $env:RUNPOD_INFERENCE_URL = "http://localhost:8000"
    .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
} -ArgumentList $root

Start-Sleep -Seconds 2

$webJob = Start-Job -Name "web" -ScriptBlock {
    param($root)
    Set-Location "$root\apps\web"
    $env:NEXT_PUBLIC_API_URL = "http://localhost:8080"
    npm run dev
} -ArgumentList $root

$jobs = @($inferJob, $apiJob, $webJob)

# ── 로그 스트리밍 루프 ────────────────────────────────────────────────────
Write-Host "[start] 서비스 기동 중..." -ForegroundColor Yellow

try {
    while ($true) {
        foreach ($job in $jobs) {
            $lines = Receive-Job -Job $job 2>&1
            foreach ($line in $lines) {
                $tag = "[{0,-9}]" -f $job.Name
                switch ($job.Name) {
                    "inference" { Write-Host "$tag $line" -ForegroundColor Magenta }
                    "api"       { Write-Host "$tag $line" -ForegroundColor Cyan    }
                    "web"       { Write-Host "$tag $line" -ForegroundColor Green   }
                }
            }
        }
        # 죽은 job 감지
        foreach ($job in $jobs) {
            if ($job.State -eq "Failed") {
                Write-Host "[$($job.Name)] FAILED — 재시작하려면 start.ps1 을 다시 실행하세요." -ForegroundColor Red
            }
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    # Ctrl+C 또는 오류 시 모든 job 정리
    Write-Host ""
    Write-Host "[start] 모든 서비스 종료 중..." -ForegroundColor Yellow
    $jobs | Stop-Job  -ErrorAction SilentlyContinue
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    Write-Host "[start] 종료 완료." -ForegroundColor DarkGray
}
