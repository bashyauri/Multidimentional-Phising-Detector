@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

echo [INFO] This downloads a FaceForensics++ subset into datasets\faceforensics.
echo [INFO] Target: 5000 original videos and 5000 Deepfakes videos, c23 compression.
echo [INFO] You must accept the FaceForensics++ terms shown by the script.

.venv\Scripts\python.exe -m ml_training.download_faceforensics_subset --output-dir datasets/faceforensics --compression c23 --num-videos 5000 --server AUTO
if errorlevel 1 goto :download_error

echo [SUCCESS] Download complete. Now run train_deepfake.bat
pause
exit /b 0

:download_error
echo [ERROR] FaceForensics++ download failed.
echo Try again later, try another network/VPN, or manually download using the official FaceForensics++ script.
pause
exit /b 1
