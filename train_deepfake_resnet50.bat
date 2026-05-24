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

echo [INFO] Training ResNet-50 Deepfake model...
echo [INFO] Profile:
echo          Frames/video: 16
echo          Image size: 224
echo          Epochs: 15
echo          Batch size: 8

.venv\Scripts\python.exe -m ml_training.train_deepfake_resnet50 ^
  --dataset-dir datasets/faceforensics ^
  --batch-size 8 ^
  --epochs 15 ^
  --frames-per-video 16 ^
  --image-size 224 ^
  --unfreeze-last-blocks 4 ^
  --val-split 0.2 ^
  --early-stop-patience 3 ^
  --pretrained ^
  --device auto

if errorlevel 1 goto :train_error

echo.
echo [SUCCESS] ResNet-50 deepfake model training complete.
echo Restart the app or reload the model before testing uploads.

pause
exit /b 0

:train_error
echo.
echo [ERROR] Deepfake training failed.
echo Make sure datasets contain readable media files.
pause
exit /b 1