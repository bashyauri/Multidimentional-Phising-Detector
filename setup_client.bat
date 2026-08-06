@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not in PATH.
  echo Install Python 3.13+ from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo [INFO] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  pause
  exit /b 1
)

echo [INFO] Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.
  echo.
  echo COMMON ISSUE: Windows path too long error
  echo ========================================
  echo If you see a "filename or extension is too long" error,
  echo this is because your project path is too long for Windows.
  echo.
  echo AUTOMATIC FIX AVAILABLE!
  echo.
  set /p "FIX=Would you like to automatically fix this issue? (Y/N): "
  if /i "%FIX%"=="Y" (
    echo.
    echo [INFO] Running automatic path fix tool...
    call fix_path_issue.bat
    echo.
    echo [INFO] Please run launch_client.bat again from the new location.
    pause
    exit /b 1
  )
  echo.
  echo MANUAL SOLUTION: Move the project folder to a shorter path:
  echo   - Current path: %CD%
  echo   - Recommended: C:\Phising or C:\Projects\Phising
  echo   - Avoid: OneDrive, Desktop with long names, nested folders
  echo.
  echo After moving to a shorter path:
  echo   1. Delete the .venv folder
  echo   2. Double-click launch_client.bat again
  echo.
  pause
  exit /b 1
)

echo [INFO] Downloading required NLTK resources...
.venv\Scripts\python.exe -m nltk.downloader punkt punkt_tab stopwords
if errorlevel 1 (
  echo [WARN] NLTK download failed. The app may try downloading at runtime.
)

echo [SUCCESS] Setup complete.
echo Next step: double-click run_client.bat
pause
exit /b 0
