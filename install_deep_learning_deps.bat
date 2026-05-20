@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

echo [INFO] Installing optional deep learning dependencies.
echo [INFO] If Python 3.13 wheels are unavailable, use Python 3.11 or 3.12 for this environment.
.venv\Scripts\python.exe -m pip install -r requirements-deep-learning.txt
if errorlevel 1 goto :install_error

echo [SUCCESS] Deep learning dependencies installed.
pause
exit /b 0

:install_error
echo [ERROR] Deep learning dependency install failed.
echo PyTorch may not provide wheels for your current Python version. Try Python 3.11 or 3.12.
pause
exit /b 1
