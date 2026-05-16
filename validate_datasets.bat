@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

echo [INFO] Validating datasets...
.venv\Scripts\python.exe -m ml_training.validate_datasets --datasets-dir datasets
if errorlevel 1 (
  echo [ERROR] Dataset validation failed.
  pause
  exit /b 1
)

echo [SUCCESS] Datasets are valid.
pause
exit /b 0
