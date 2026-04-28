@echo off
echo ============================================================
echo   LeakSense Twin - Cat C18 Engine Diagnostics
echo   Starting Backend + Frontend Servers
echo ============================================================
echo.

:: Get the directory of this script
set "SCRIPT_DIR=%~dp0"

:: Start Backend
echo [1/2] Starting Backend Server (FastAPI on port 8000)...
start "LeakSense Backend" cmd /k "cd /d "%SCRIPT_DIR%backend" && python main.py"
timeout /t 5 /nobreak > nul

:: Start Frontend
echo [2/2] Starting Frontend Dev Server (Vite on port 5173)...
start "LeakSense Frontend" cmd /k "cd /d "%SCRIPT_DIR%frontend" && node node_modules\vite\bin\vite.js --port 5173"
timeout /t 3 /nobreak > nul

echo.
echo ============================================================
echo   LeakSense Twin is running!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ============================================================
echo.
echo Opening dashboard in browser...
start http://localhost:5173
