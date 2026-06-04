@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo [INFO] Generating URL, SMS, Email, and QR evaluation samples...
%PYTHON% ml_training\generate_text_qr_eval_samples.py
if errorlevel 1 goto :error

echo [SUCCESS] Evaluation samples created in datasets\evaluation_samples
pause
exit /b 0

:error
echo [ERROR] Evaluation sample generation failed.
echo If QR generation failed, install qrcode with: pip install qrcode[pil]
pause
exit /b 1
