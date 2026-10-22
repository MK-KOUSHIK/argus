# Run Backend Local Development Server
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RootDir "backend"

Set-Location $BackendDir

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating Python virtual environment (.venv)..." -ForegroundColor Cyan
    & ".\.venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating Python virtual environment (venv)..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "Notice: No local virtual environment found. Using system python." -ForegroundColor Yellow
}

Write-Host "Starting Argus FastAPI backend on http://localhost:8000..." -ForegroundColor Green
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
