<#
.SYNOPSIS
    Managed Ollama Server Lifecycle & Runtime Configuration Script for Argus (Windows).

.DESCRIPTION
    Launches, stops, restarts, and verifies local Ollama inference server processes
    with explicit KV-cache quantization (F16 / Q8_0) and Flash Attention settings.
    Records managed PID, runtime parameters, and verification evidence in a machine-readable
    manifest (backend/data/ollama_runtime.json).
#>

[CmdletBinding()]
param(
    [ValidateSet("start", "stop", "restart", "status")]
    [string]$Action = "start",

    [ValidateSet("f16", "q8_0", "q4_0")]
    [string]$KvType = "q8_0",

    $FlashAttention = $true,

    [int]$Port = 11434,

    [string]$HostAddress = "127.0.0.1",

    [string]$ManifestPath = "backend/data/ollama_runtime.json",

    [string]$LogPath = "backend/data/ollama_server.log",

    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

if ($FlashAttention -isnot [bool]) {
    $strFa = $FlashAttention.ToString().ToLower().Trim()
    $FlashAttention = ($strFa -eq "1" -or $strFa -eq "true" -or $strFa -eq "$true")
}

# Resolve absolute paths
$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$absoluteManifestPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ManifestPath))
$absoluteLogPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $LogPath))
$absoluteErrLogPath = $absoluteLogPath -replace '\.log$', '.err.log'

# Ensure data directory exists
$manifestDir = [System.IO.Path]::GetDirectoryName($absoluteManifestPath)
if (-not (Test-Path $manifestDir)) {
    New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
}

function Get-PortProcessId([int]$targetPort) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $targetPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) {
            return $conn.OwningProcess
        }
    } catch {
        # Fallback to netstat if Get-NetTCPConnection is unavailable
        $netstat = netstat -ano | Select-String ":$targetPort\s+.*LISTENING\s+(\d+)"
        if ($netstat -match ":$targetPort\s+.*LISTENING\s+(\d+)") {
            return [int]$matches[1]
        }
    }
    return $null
}

function Read-RuntimeManifest() {
    if (Test-Path $absoluteManifestPath) {
        try {
            $raw = Get-Content -Path $absoluteManifestPath -Raw -Encoding UTF8
            return ($raw | ConvertFrom-Json)
        } catch {
            return $null
        }
    }
    return $null
}

function Write-RuntimeManifest($manifestData) {
    $json = $manifestData | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($absoluteManifestPath, $json, [System.Text.Encoding]::UTF8)
}

function Test-OllamaHealth([string]$endpoint) {
    try {
        $res = Invoke-RestMethod -Uri "$endpoint/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
        $models = @()
        if ($res.models) {
            $models = $res.models | ForEach-Object { $_.name }
        }
        return @{
            Healthy = $true
            Models = $models
        }
    } catch {
        return @{
            Healthy = $false
            Models = @()
        }
    }
}

function Scan-ServerLogEvidence([string]$stdLog, [string]$errLog, [string]$targetKv, [bool]$targetFa) {
    $allLines = @()
    if (Test-Path $stdLog) {
        $allLines += Get-Content -Path $stdLog -ErrorAction SilentlyContinue
    }
    if (Test-Path $errLog) {
        $allLines += Get-Content -Path $errLog -ErrorAction SilentlyContinue
    }

    if ($allLines.Count -eq 0) {
        return @{
            KvConfirmed = $false
            FaConfirmed = $false
            Confirmed = $false
            Evidence = "No log output recorded yet"
            RawSnippet = $null
        }
    }

    $kvEvidence = $null
    $faEvidence = $null

    # 1. Independent KV-Cache Check
    $serverConfigLine = $allLines | Select-String "OLLAMA_KV_CACHE_TYPE:$targetKv" | Select-Object -Last 1
    $kvBufferLine = $allLines | Select-String "llama_kv_cache:\s*size\s*=\s*.*K\s*\($targetKv\)" | Select-Object -Last 1
    if ($serverConfigLine) {
        $kvEvidence = "Server config: OLLAMA_KV_CACHE_TYPE=$targetKv"
    } elseif ($kvBufferLine) {
        $kvEvidence = $kvBufferLine.Line.Trim()
    }

    # 2. Independent Flash Attention Check
    $flashLine = $allLines | Select-String "flash_attn\s*=\s*enabled" | Select-Object -Last 1
    $faConfigLine = $allLines | Select-String "OLLAMA_FLASH_ATTENTION:true|OLLAMA_FLASH_ATTENTION:1" | Select-Object -Last 1
    if ($flashLine) {
        $faEvidence = "llama_context: flash_attn = enabled"
    } elseif ($faConfigLine) {
        $faEvidence = "Server config: OLLAMA_FLASH_ATTENTION=1"
    }

    $kvConfirmed = [bool]$kvEvidence
    $faConfirmed = [bool]$faEvidence
    $allConfirmed = ($kvConfirmed -and $faConfirmed)

    $evidenceItems = @()
    if ($kvEvidence) { $evidenceItems += $kvEvidence }
    if ($faEvidence) { $evidenceItems += $faEvidence }

    return @{
        KvConfirmed = $kvConfirmed
        FaConfirmed = $faConfirmed
        Confirmed = $allConfirmed
        Evidence = if ($evidenceItems) { ($evidenceItems -join " | ") } else { "Runtime settings not yet observed in logs" }
        RawSnippet = if ($evidenceItems) { $evidenceItems[0] } else { $null }
    }
}

# -------------------------------------------------------------
# ACTION: STATUS
# -------------------------------------------------------------
if ($Action -eq "status") {
    $manifest = Read-RuntimeManifest
    $portPid = Get-PortProcessId -targetPort $Port
    $endpoint = "http://$HostAddress`:$Port"
    $health = Test-OllamaHealth -endpoint $endpoint

    Write-Host "=== Argus Ollama Runtime Status ===" -ForegroundColor Cyan
    Write-Host "Port: $Port | Endpoint: $endpoint"

    if ($manifest) {
        Write-Host "Manifest PID: $($manifest.pid)"
        Write-Host "Configured KV Cache: $($manifest.kv_cache_type)"
        Write-Host "Configured Flash Attention: $($manifest.flash_attention)"
        Write-Host "Launch Timestamp: $($manifest.start_time)"
        Write-Host "KV Cache Confirmed: $($manifest.kv_cache_confirmed)"
        Write-Host "Flash Attention Confirmed: $($manifest.flash_attention_confirmed)"
        Write-Host "Confirmation: $($manifest.confirmation_evidence)"
    } else {
        Write-Host "Manifest: No active manifest found" -ForegroundColor DarkGray
    }

    if ($portPid) {
        $proc = Get-Process -Id $portPid -ErrorAction SilentlyContinue
        Write-Host "Active Listening Process: PID $portPid ($($proc.ProcessName))" -ForegroundColor Green
        Write-Host "Ollama API Responding: $($health.Healthy)"
        Write-Host "Installed Models: $($health.Models -join ', ')"
    } else {
        Write-Host "Active Listening Process: None (Port $Port is free)" -ForegroundColor Yellow
    }

    exit 0
}

# -------------------------------------------------------------
# ACTION: STOP
# -------------------------------------------------------------
if ($Action -eq "stop" -or $Action -eq "restart") {
    Write-Host "Stopping Ollama process on port $Port..." -ForegroundColor Cyan
    $manifest = Read-RuntimeManifest
    $portPid = Get-PortProcessId -targetPort $Port

    # Safety check: Only stop managed process if it is actually ollama!
    if ($manifest -and $manifest.pid) {
        $proc = Get-Process -Id $manifest.pid -ErrorAction SilentlyContinue
        if ($proc) {
            if ($proc.ProcessName -like "*ollama*") {
                Write-Host "Stopping managed Ollama process PID $($proc.Id)..."
                Stop-Process -Id $proc.Id -Force
                Start-Sleep -Seconds 1
            } else {
                Write-Warning "Manifest PID $($manifest.pid) belongs to '$($proc.ProcessName)', not Ollama. Will not terminate unrelated process."
            }
        }
    }

    # Stop listening process on target port only if it is Ollama
    if ($portPid) {
        $otherProc = Get-Process -Id $portPid -ErrorAction SilentlyContinue
        if ($otherProc -and $otherProc.ProcessName -like "*ollama*") {
            Write-Host "Stopping Ollama process PID $portPid on port $Port..."
            Stop-Process -Id $portPid -Force
            Start-Sleep -Seconds 1
        } elseif ($otherProc) {
            if ($Force) {
                Write-Host "Force stopping unrelated process PID $portPid ($($otherProc.ProcessName))..." -ForegroundColor Red
                Stop-Process -Id $portPid -Force
            } else {
                throw "Port $Port is occupied by unrelated process PID $portPid ($($otherProc.ProcessName)). Refusing to terminate automatically. Use -Force to override."
            }
        }
    }

    # Verify stopped
    $remainingPid = Get-PortProcessId -targetPort $Port
    if ($remainingPid) {
        throw "Failed to stop process on port $Port (PID $remainingPid still listening)."
    }

    if ($manifest) {
        $manifest.status = "stopped"
        Write-RuntimeManifest $manifest
    }

    Write-Host "Ollama process successfully stopped." -ForegroundColor Green
    if ($Action -eq "stop") {
        exit 0
    }
}

# -------------------------------------------------------------
# ACTION: START (or continuation of RESTART)
# -------------------------------------------------------------
Write-Host "Starting Ollama server process..." -ForegroundColor Cyan
Write-Host "  KV-Cache Type: $KvType"
Write-Host "  Flash Attention: $FlashAttention"
Write-Host "  Binding: $HostAddress`:$Port"

# Check if port is already in use
$existingPid = Get-PortProcessId -targetPort $Port
if ($existingPid) {
    $existingProc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    $manifest = Read-RuntimeManifest

    if ($manifest -and $manifest.pid -eq $existingPid -and $manifest.kv_cache_type -eq $KvType -and $manifest.flash_attention -eq $FlashAttention) {
        Write-Host "Ollama is already running with requested configuration (PID $existingPid)." -ForegroundColor Green
        exit 0
    } else {
        if ($Force -and $existingProc -and $existingProc.ProcessName -like "*ollama*") {
            Write-Host "Terminating existing Ollama process PID $existingPid to apply new configuration..." -ForegroundColor Yellow
            Stop-Process -Id $existingPid -Force
            Start-Sleep -Seconds 1
        } else {
            $msg = "Port $Port is in use by PID $existingPid ($($existingProc.ProcessName)). " +
                   "Current server configuration may not match requested settings (KV: $KvType, FlashAttn: $FlashAttention). " +
                   "Run with -Action restart or -Force to restart with the new configuration."
            throw $msg
        }
    }
}

# Set environment variables for the Ollama server process
$env:OLLAMA_FLASH_ATTENTION = if ($FlashAttention) { "1" } else { "0" }
$env:OLLAMA_KV_CACHE_TYPE = $KvType
$env:OLLAMA_HOST = "$HostAddress`:$Port"
$env:OLLAMA_DEBUG = "1"

# Prepare clean log files and unique launch session id
$sessionId = [Guid]::NewGuid().ToString()
if (Test-Path $absoluteLogPath) {
    Remove-Item $absoluteLogPath -Force -ErrorAction SilentlyContinue
}
if (Test-Path $absoluteErrLogPath) {
    Remove-Item $absoluteErrLogPath -Force -ErrorAction SilentlyContinue
}

# Launch Ollama serve in background with separate stdout/stderr log files
$proc = Start-Process -FilePath "ollama" -ArgumentList "serve" `
    -RedirectStandardOutput $absoluteLogPath `
    -RedirectStandardError $absoluteErrLogPath `
    -NoNewWindow -PassThru

if (-not $proc -or $proc.HasExited) {
    throw "Failed to start Ollama process. Check logs at $absoluteErrLogPath"
}

Write-Host "Ollama launched with PID $($proc.Id) (Session: $sessionId). Waiting for readiness..."

# Poll health check
$endpoint = "http://$HostAddress`:$Port"
$ready = $false
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Milliseconds 500
    $attempt++

    if ($proc.HasExited) {
        $logContent = if (Test-Path $absoluteErrLogPath) { Get-Content $absoluteErrLogPath -Tail 20 | Out-String } else { "No log file" }
        throw "Ollama process PID $($proc.Id) terminated unexpectedly. ExitCode: $($proc.ExitCode)`nLogs:`n$logContent"
    }

    $health = Test-OllamaHealth -endpoint $endpoint
    if ($health.Healthy) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for Ollama API at $endpoint after 15 seconds."
}

Write-Host "Ollama is responsive at $endpoint." -ForegroundColor Green

# Check for model availability
$targetModel = "qwen2.5:1.5b"
$modelInstalled = $false
if ($health.Models) {
    $modelInstalled = ($health.Models | Where-Object { $_ -like "*$targetModel*" }).Count -gt 0
}

Write-Host "Target model '$targetModel' installed: $modelInstalled"
if (-not $modelInstalled) {
    Write-Warning "Model '$targetModel' not found in local Ollama repository. Available models: $($health.Models -join ', ')"
}

# Check initial log evidence
$evidence = Scan-ServerLogEvidence -stdLog $absoluteLogPath -errLog $absoluteErrLogPath -targetKv $KvType -targetFa $FlashAttention

# Save manifest with session tracking
$manifestData = [PSCustomObject]@{
    session_id = $sessionId
    pid = $proc.Id
    status = "running"
    kv_cache_type = $KvType
    flash_attention = $FlashAttention
    host = $HostAddress
    port = $Port
    endpoint = $endpoint
    start_time = [DateTime]::UtcNow.ToString("o")
    log_path = $absoluteLogPath
    err_log_path = $absoluteErrLogPath
    default_model = $targetModel
    model_installed = $modelInstalled
    available_models = $health.Models
    kv_cache_confirmed = $evidence.KvConfirmed
    flash_attention_confirmed = $evidence.FaConfirmed
    independently_confirmed = $evidence.Confirmed
    confirmation_evidence = $evidence.Evidence
}

Write-RuntimeManifest $manifestData
Write-Host "Runtime manifest written to $absoluteManifestPath" -ForegroundColor Green
Write-Host "=== Ollama Startup Complete ===" -ForegroundColor Cyan
Write-Host "PID: $($proc.Id) | KV: $KvType | Flash Attention: $FlashAttention"
Write-Host "Confirmation: $($evidence.Evidence)"
