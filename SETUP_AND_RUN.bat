@echo off
echo ============================================
echo  AgriBot - Setup and Run
echo ============================================
echo.

echo [1/3] Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Setting up database...
python manage.py migrate
if errorlevel 1 (
    echo ERROR: migrate failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting AgriBot server...
echo.
echo  Open your browser and go to: http://127.0.0.1:8000
echo  Press Ctrl+C to stop the server.
echo.
python manage.py runserver
pause
