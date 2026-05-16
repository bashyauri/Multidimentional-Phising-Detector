@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Creating desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_desktop_shortcut.ps1"
if errorlevel 1 (
  echo [ERROR] Failed to create desktop shortcut.
  pause
  exit /b 1
)

echo [SUCCESS] Desktop shortcut created.
pause
exit /b 0
