@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

if not exist "datasets\faceforensics\original" (
  echo [ERROR] Missing datasets\faceforensics\original
  pause
  exit /b 1
)

if not exist "datasets\faceforensics\manipulated" (
  echo [ERROR] Missing datasets\faceforensics\manipulated
  pause
  exit /b 1
)

echo [INFO] Training Deepfake model from datasets\faceforensics...
.venv\Scripts\python.exe -m ml_training.train_deepfake --dataset-dir datasets/faceforensics
if errorlevel 1 goto :train_error

echo [SUCCESS] Deepfake model training complete.
echo Restart the app or call POST /api/reload-models before testing uploads.
pause
exit /b 0

:train_error
echo [ERROR] Deepfake training failed.
echo Make sure original\ and manipulated\ contain at least two readable images/videos each.
pause
exit /b 1
