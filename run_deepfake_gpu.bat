@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv-gpu\Scripts\python.exe" (
  echo [ERROR] GPU virtual environment not found at .venv-gpu\Scripts\python.exe
  echo Install/setup it first, then run again.
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

echo [INFO] Running EfficientNet-B0 deepfake training on DirectML GPU...
echo [INFO] Default settings: face crop, batch=1 epochs=20 frames=8 image=160 unfreeze=2 val=0.2 patience=5
echo [INFO] Any command-line options you pass after this file override those defaults.
echo [INFO] This profile is tuned for low-VRAM DirectML GPUs.

echo [INFO] MSc-level settings: batch=2 epochs=15 frames=8 image=192 unfreeze=4 val=0.2 patience=3
REM If you get an out-of-memory error, switch to CPU by changing --device dml to --device cpu and reduce batch/image size if needed.
REM Example for CPU (safer, but slower):
REM .venv-gpu\Scripts\python.exe -m ml_training.train_deepfake_efficientnet --dataset-dir datasets/faceforensics --batch-size 1 --epochs 15 --frames-per-video 4 --image-size 160 --unfreeze-last-blocks 2 --val-split 0.2 --early-stop-patience 3 --pretrained --device cpu %*

.venv-gpu\Scripts\python.exe -m ml_training.train_deepfake_efficientnet --dataset-dir datasets/faceforensics --batch-size 1 --epochs 15 --frames-per-video 4 --image-size 160 --unfreeze-last-blocks 2 --val-split 0.2 --early-stop-patience 3 --pretrained --device dml %*
if errorlevel 1 goto :train_error

echo [SUCCESS] GPU training finished.
echo Model: models\deepfake_efficientnet_b0.pt
echo Metrics: models\deepfake_efficientnet_b0_metrics.json
pause
exit /b 0

:train_error
echo [ERROR] GPU training failed.
echo If this is a package error, verify .venv-gpu has torch-directml and dependencies installed.
pause
exit /b 1
