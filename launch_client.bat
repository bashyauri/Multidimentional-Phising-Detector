@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Smart launcher started...

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not in PATH.
  echo Install Python 3.13+ from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] First-time setup detected. Running setup_client.bat...
  call setup_client.bat
  if errorlevel 1 (
    echo [ERROR] Setup failed. Cannot continue.
    pause
    exit /b 1
  )
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if exist ".venv-gpu\Scripts\python.exe" (
  set "PYTHON_EXE=.venv-gpu\Scripts\python.exe"
  echo [INFO] GPU environment detected. Using .venv-gpu for model inference.
)

set "SMS_PREFER_TRANSFORMER=false"
set "EMAIL_PREFER_TRANSFORMER=false"

echo [INFO] SMS model priority: Logistic Regression first
echo [INFO] Email model priority: Logistic Regression first
echo [INFO] Launching web application...
start "" http://127.0.0.1:5000
%PYTHON_EXE% app.py

if errorlevel 1 (
  echo [ERROR] App exited with an error.
)

pause
exit /b 0
