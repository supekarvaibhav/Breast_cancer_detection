@echo off
title BreastGuard AI – Setup

echo ============================================================
echo   BreastGuard AI – First-Time Setup
echo ============================================================
echo.

REM Step 1: Create venv
if not exist venv (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

REM Step 2: Activate and install
echo [2/4] Installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

REM Step 3: Init DB
echo [3/4] Initializing database...
python -c "from run import app; from app import db; app.app_context().__enter__(); db.create_all(); print('DB ready')"

echo [4/4] Setup complete!
echo.
echo ============================================================
echo   To train the AI model:
echo   python train.py --dataset C:\path\to\BreaKHis_v1
echo.
echo   To start the web server:
echo   python run.py
echo.
echo   Then open: http://localhost:5000
echo ============================================================
pause
