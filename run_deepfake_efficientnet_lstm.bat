@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv-gpu\Scripts\python.exe" (
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

echo [INFO] Running EfficientNet-B0 + LSTM deepfake training (CPU by default)...
.venv-gpu\Scripts\python.exe -m ml_training.train_deepfake_efficientnet_lstm --dataset-dir datasets/faceforensics --batch-size 1 --epochs 10 --frames-per-video 4 --image-size 160 --pretrained --device cpu %*

if errorlevel 1 goto :train_error

echo [SUCCESS] Training finished.
pause
exit /b 0

:train_error
echo [ERROR] Training failed.
pause
exit /b 1
