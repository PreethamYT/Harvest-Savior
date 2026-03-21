# ==============================================================
#  Harvest Savior — Phase 2 Environment Setup Script
#  Run this ONCE from any PowerShell window.
#
#  WHAT THIS DOES:
#    1. Creates a Python virtual environment at C:\hs_venv
#       (short path avoids Windows' 260-char limit that breaks TF)
#    2. Installs TensorFlow, Pillow, matplotlib, scikit-learn
#    3. Installs Flask (so the AI server also runs in this env)
#    4. Prints the path to the Python executable so you know
#       which interpreter to use to run app.py and train_model.py
#
#  USAGE (run from any directory):
#    powershell -ExecutionPolicy Bypass -File setup_training_env.ps1
# ==============================================================

$envPath = "C:\hs_venv"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Harvest Savior — Phase 2 Training Environment Setup" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

# ── Step 1: Create the virtual environment ────────────────────
if (Test-Path "$envPath\Scripts\python.exe") {
    Write-Host "[1/3] Virtual environment already exists at $envPath" -ForegroundColor Yellow
} else {
    Write-Host "[1/3] Creating virtual environment at $envPath ..." -ForegroundColor Cyan
    python -m venv $envPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Could not create venv. Make sure Python 3.9+ is installed." -ForegroundColor Red
        exit 1
    }
    Write-Host "      Done." -ForegroundColor Green
}

$pip    = "$envPath\Scripts\pip.exe"
$python = "$envPath\Scripts\python.exe"

# ── Step 2: Upgrade pip inside the venv ───────────────────────
Write-Host ""
Write-Host "[2/3] Upgrading pip ..." -ForegroundColor Cyan
& $pip install --upgrade pip --quiet

# ── Step 3: Install all required packages ─────────────────────
Write-Host ""
Write-Host "[3/3] Installing packages (TensorFlow, Flask, Pillow, matplotlib, scikit-learn) ..." -ForegroundColor Cyan
Write-Host "      This may take 3-5 minutes on first run." -ForegroundColor Gray

& $pip install `
    tensorflow `
    flask `
    Pillow `
    numpy `
    matplotlib `
    scikit-learn `
    kaggle

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Package installation failed." -ForegroundColor Red
    exit 1
}

# ── Done ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Python executable : $python" -ForegroundColor Cyan
Write-Host ""
Write-Host "TO START THE FLASK SERVER (Phase 1):" -ForegroundColor Yellow
Write-Host "  cd harvest-savior-ai"
Write-Host "  $python app.py"
Write-Host ""
Write-Host "TO TRAIN THE CNN MODEL (Phase 2):" -ForegroundColor Yellow
Write-Host "  cd harvest-savior-ai"
Write-Host "  $python train_model.py"
Write-Host ""
