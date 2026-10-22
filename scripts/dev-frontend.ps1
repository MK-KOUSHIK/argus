# Run Frontend Local Development Server
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $RootDir "frontend"

Set-Location $FrontendDir

Write-Host "Starting Argus React + Vite frontend on http://localhost:5173..." -ForegroundColor Cyan
npm run dev
