@echo off
title Road Traffic Sign Detection & Recognition System
echo ========================================================
echo   Road / Traffic Sign Detection & Recognition System
echo   Self-Hosted Production Launcher (Member D)
echo ========================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/3] Checking environment & dependencies...
pip install -r requirements.txt --quiet

echo [2/3] Checking sample datasets...
if not exist "data\samples\00000.png" (
    echo Extracting sample test dataset...
    python scripts\prepare_sample_data.py
)

if not exist "data\samples\simulated_driving.mp4" (
    echo Generating simulated driving video...
    python scripts\generate_driving_simulation.py
)

echo [3/3] Launching Streamlit Web Dashboard...
echo Dashboard will open in your default web browser at http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the application.
echo.
streamlit run app.py

pause
