@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo [INFO] Generating 10 legitimate-style and 10 fake-style voice WAV samples...
%PYTHON% ml_training\generate_voice_test_samples.py
if errorlevel 1 goto :error

echo [SUCCESS] Voice test samples created in datasets\voice\test_samples
pause
exit /b 0

:error
echo [ERROR] Voice sample generation failed.
pause
exit /b 1
