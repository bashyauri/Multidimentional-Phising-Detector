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

echo [INFO] Launching web application...
start "" http://127.0.0.1:5000
.venv\Scripts\python.exe app.py

if errorlevel 1 (
  echo [ERROR] App exited with an error.
)

pause
exit /b 0
