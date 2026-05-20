@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

set "SMS_PREFER_TRANSFORMER=false"
set "EMAIL_PREFER_TRANSFORMER=false"

echo [INFO] SMS model priority: Logistic Regression first
echo [INFO] Email model priority: Logistic Regression first
echo [INFO] Starting Flask app...
.venv\Scripts\python.exe app.py

if errorlevel 1 (
  echo [ERROR] App exited with an error.
)

pause
exit /b 0
