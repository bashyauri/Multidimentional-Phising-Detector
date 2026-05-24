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

echo [INFO] Running ConvNeXt-Tiny deepfake training on DirectML GPU...
echo [INFO] Default settings: face crop, batch=1 epochs=8 frames=4 image=128 unfreeze=2 val=0.2 patience=2
echo [INFO] Any command-line options you pass after this file override those defaults.
echo [INFO] This profile is tuned for low-VRAM DirectML GPUs.

.venv-gpu\Scripts\python.exe -m ml_training.train_deepfake_convnext --dataset-dir datasets/faceforensics --batch-size 1 --epochs 8 --frames-per-video 4 --image-size 128 --unfreeze-last-blocks 2 --val-split 0.2 --early-stop-patience 2 --pretrained --device dml %*
if errorlevel 1 goto :train_error

echo [SUCCESS] GPU training finished.
echo Model: models\deepfake_convnext_tiny.pt
echo Metrics: models\deepfake_convnext_tiny_metrics.json
pause
exit /b 0

:train_error
echo [ERROR] GPU training failed.
echo If this is a package error, verify .venv-gpu has torch-directml and dependencies installed.
pause
exit /b 1
