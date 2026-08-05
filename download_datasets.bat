@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo DATASET DOWNLOAD SCRIPT
echo Phishing Detection Research App
echo ========================================
echo.

echo This script will attempt to download all required datasets.
echo Some datasets require manual download from Kaggle.
echo.

pause

echo [INFO] Checking Python installation...
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not in PATH.
  echo Install Python 3.9+ from https://www.python.org/downloads/
  pause
  exit /b 1
)

echo [INFO] Python found. Starting dataset download...
echo.

python download_datasets.py

if errorlevel 1 (
  echo [ERROR] Dataset download script failed.
  pause
  exit /b 1
)

echo.
echo ========================================
echo DOWNLOAD COMPLETE
echo ========================================
echo.
echo Please check the datasets/ directory for downloaded files.
echo Some datasets may require manual download (see instructions above).
echo.
echo After downloading datasets:
echo 1. Run validate_datasets.bat to verify datasets
echo 2. Run train_models.bat to train models
echo.

pause
exit /b 0
