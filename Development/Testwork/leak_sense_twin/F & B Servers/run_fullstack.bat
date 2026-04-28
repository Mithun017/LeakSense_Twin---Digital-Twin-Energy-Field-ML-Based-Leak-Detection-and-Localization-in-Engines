@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   LeakSense - Full Stack Unified Dashboard
echo ==========================================

:: Step 1: Kill existing zombie processes on ports 8001 and 3001
echo [1/4] Cleaning up ports 8001 and 3001...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p8001 = (Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue).OwningProcess; if ($p8001) { Stop-Process -Id $p8001 -Force }; $p3001 = (Get-NetTCPConnection -LocalPort 3001 -ErrorAction SilentlyContinue).OwningProcess; if ($p3001) { Stop-Process -Id $p3001 -Force }"

:: Step 2: Start Backend
echo [2/4] Starting Advanced API (Port 8001)...
:: Using /D flag to avoid path parsing bugs in the command string itself
start "LeakSense Backend" /D "%~dp0backend" cmd /c uvicorn main:app --port 8001

:: Step 3: Start Frontend
echo [3/4] Starting Advanced Dashboard (Port 3001)...
:: We call vite.js directly via node to bypass the broken CMD path parsing inside 'npm run dev'
start "LeakSense Frontend" /D "%~dp0frontend" cmd /c "node node_modules\vite\bin\vite.js --port 3001"

:: Step 4: Launch Browser
echo [4/4] Initializing Dashboard (5 seconds)...
timeout /t 5 /nobreak > nul
start http://localhost:3001

echo.
echo ==========================================
echo   System is live! Keep this window open.
echo ==========================================
pause
