@echo off
TITLE Road Traffic Sign Detection & Recognition System
color 0A
cls

echo =====================================================================
echo    ROAD TRAFFIC SIGN DETECTION AND RECOGNITION SYSTEM
echo    GTSRB 43 Classes -- Fully Integrated Vision Platform
echo =====================================================================
echo.
echo Select an option to launch:
echo.
echo   [1] Launch Interactive Web Dashboard (Streamlit)
echo   [2] Launch REST API Service (FastAPI / Uvicorn)
echo   [3] Run Automated Pytest Test Suite
echo   [4] Process Sample Image (CLI)
echo   [5] Process Sample Video (CLI)
echo   [6] Run Member A Complete Data Pipeline
echo   [7] Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo Launching Streamlit Dashboard...
    python -m streamlit run app.py
    pause
    goto end
)

if "%choice%"=="2" (
    echo Launching FastAPI Server at http://127.0.0.1:8000 ...
    python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
    pause
    goto end
)

if "%choice%"=="3" (
    echo Running Pytest Suite...
    python -m pytest tests/ -v
    pause
    goto end
)

if "%choice%"=="4" (
    echo Processing Sample Image...
    python cli.py --source image --input data/samples/class_14_sample_1.png --show
    pause
    goto end
)

if "%choice%"=="5" (
    echo Processing Sample Video...
    python cli.py --source video --input data/samples/simulated_driving.mp4 --show
    pause
    goto end
)

if "%choice%"=="6" (
    echo Running Complete Data Pipeline...
    python run_pipeline.py --skip-download --full
    pause
    goto end
)

:end
