@echo off
echo ================================================
echo   Elder Trading System - Starting...
echo ================================================
echo.

cd /d "%~dp0backend"

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting server...
echo Open browser at: http://localhost:5001
echo.
python app.py

pause
