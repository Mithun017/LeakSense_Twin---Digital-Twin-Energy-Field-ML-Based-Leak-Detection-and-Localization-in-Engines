@echo off
title LeakSense Twin - Leak Detection and Localization System
echo.
echo ================================================================
echo  LeakSense Twin - Digital Twin ^& Energy Field ^& ML Based
echo  Leak Detection and Localization in Engines
echo ================================================================
echo.
echo Starting the LeakSense Twin system...
echo.

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Change to the leak_sense_twin directory
cd leak_sense_twin

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.7+ and add it to your PATH.
    echo.
    pause
    exit /b 1
)

REM Check if requirements are installed, install if missing
python -c "import numpy, pandas, sklearn, torch, joblib" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required Python packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install required packages.
        echo.
        pause
        exit /b 1
    )
)

REM Run the main leak detection system
echo.
echo Running LeakSense Twin system...
echo.
python main_leak_detection_system.py

REM Pause so user can see the results
echo.
echo ================================================================
echo  System execution completed. Press any key to exit...
echo ================================================================
pause >nul