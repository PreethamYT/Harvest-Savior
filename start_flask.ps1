# =============================================================================
#  start_flask.ps1 — Harvest Savior AI Microservice Launcher
# =============================================================================
#
#  PURPOSE
#  -------
#  Reliably starts the Python/Flask AI microservice using the venv that
#  has TensorFlow properly installed.
#
#  IMPORTANT — WHY THIS SCRIPT EXISTS
#  ------------------------------------
#  On Windows, typing plain "python app.py" can accidentally launch the
#  Windows Store Python stub (PythonSoftwareFoundation.Python.*), which has
#  a broken TensorFlow installation due to the Windows 260-char Long Path
#  limit. When that happens, Flask starts in DEMO MODE and returns fake
#  predictions. This script hard-codes the correct interpreter path to
#  prevent that issue permanently.
#
#  HOW TO RUN
#  ----------
#  From any PowerShell terminal in the project folder:
#      .\start_flask.ps1
#
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
$PYTHON_EXE   = "C:\Users\Preetham Rao\Desktop\Harvest_Savior\Harvest-Savior\ai_engine\venv\Scripts\python.exe"
$FLASK_DIR    = Join-Path $PSScriptRoot "harvest-savior-ai"
$FLASK_PORT   = 5000

# ── Step 1: Verify the Python executable exists ───────────────────────────────
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "[ERROR] Python executable not found at:" -ForegroundColor Red
    Write-Host "        $PYTHON_EXE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure the original Harvest-Savior project venv is present." -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Using Python: $PYTHON_EXE" -ForegroundColor Cyan

# ── Step 2: Kill ANY process currently holding port 5000 ─────────────────────
#  netstat shows the PID but Windows Store Python can hide from Get-Process.
#  We therefore also kill any python.exe process with "app.py" in its command.
Write-Host "[INFO] Checking for processes on port $FLASK_PORT..." -ForegroundColor Cyan

$portPid = (netstat -ano | Select-String ":$FLASK_PORT\s.*LISTENING") -replace '.*LISTENING\s+', '' | Select-Object -First 1
if ($portPid) {
    Write-Host "[INFO] Killing PID $portPid (holding port $FLASK_PORT)..." -ForegroundColor Yellow
    Stop-Process -Id ([int]$portPid) -Force -ErrorAction SilentlyContinue
}

# Also kill any python process with app.py in its command line (catches reloader children)
Get-WmiObject Win32_Process | Where-Object {
    $_.CommandLine -match "app\.py" -and $_.Name -match "python"
} | ForEach-Object {
    Write-Host "[INFO] Killing stale Flask process PID $($_.ProcessId) ($($_.CommandLine))" -ForegroundColor Yellow
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

# ── Step 3: Confirm port is free ──────────────────────────────────────────────
$stillInUse = netstat -ano | Select-String ":$FLASK_PORT\s.*LISTENING"
if ($stillInUse) {
    Write-Host "[ERROR] Port $FLASK_PORT is still in use. Cannot start Flask." -ForegroundColor Red
    Write-Host "        $stillInUse" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Port $FLASK_PORT is free." -ForegroundColor Green

# ── Step 4: Start Flask ───────────────────────────────────────────────────────
Write-Host "[INFO] Starting Flask from: $FLASK_DIR" -ForegroundColor Cyan
Write-Host "[INFO] TensorFlow model load takes ~60 seconds on first start." -ForegroundColor Yellow
Write-Host "[INFO] Watch for '[Predictor] Model ready. Classes: 15' in the output." -ForegroundColor Yellow
Write-Host ""

Set-Location $FLASK_DIR

# Run Flask directly in this terminal so all logs are visible.
# Press Ctrl+C to stop Flask when done.
& $PYTHON_EXE app.py
