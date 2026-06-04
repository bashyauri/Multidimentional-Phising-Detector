@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

if not exist "datasets\url_phishing.csv" (
  echo [ERROR] Missing datasets\url_phishing.csv
  pause
  exit /b 1
)
if not exist "datasets\email_phishing.csv" (
  echo [ERROR] Missing datasets\email_phishing.csv
  pause
  exit /b 1
)
if not exist "datasets\sms_spam.csv" (
  echo [ERROR] Missing datasets\sms_spam.csv
  pause
  exit /b 1
)

echo [INFO] Running dataset validation...
.venv\Scripts\python.exe -m ml_training.validate_datasets --datasets-dir datasets
if errorlevel 1 goto :train_error

echo [INFO] Training URL model...
.venv\Scripts\python.exe -m ml_training.train_url --dataset datasets/url_phishing.csv
if errorlevel 1 goto :train_error

if exist "datasets\qr" (
  echo [INFO] Training QR model from datasets\qr using XGBoost when available...
  .venv\Scripts\python.exe -m ml_training.train_qr --dataset-dir "datasets/qr"
  if errorlevel 1 (
    echo [WARN] QR training failed; continuing with URL/Email/SMS models.
  )
) else (
  echo [WARN] Skipping QR training: datasets\qr not found.
)

echo [INFO] Training Email model...
.venv\Scripts\python.exe -m ml_training.train_email --dataset datasets/email_phishing.csv
if errorlevel 1 goto :train_error

echo [INFO] Training SMS model...
.venv\Scripts\python.exe -m ml_training.train_sms --dataset datasets/sms_spam.csv
if errorlevel 1 goto :train_error

if exist "datasets\voice\DATASET-balanced.csv" (
  echo [INFO] Training Voice Deepfake model from datasets\voice\DATASET-balanced.csv...
  .venv\Scripts\python.exe ml_training\train_voice_deepfake_balanced.py
  if errorlevel 1 (
    echo [WARN] Voice deepfake training failed; continuing with other models.
  )
) else (
  echo [WARN] Skipping voice deepfake training: datasets\voice\DATASET-balanced.csv not found.
)

echo [INFO] Checking deep learning dependencies for deepfake training...
.venv\Scripts\python.exe -c "import torch, torchvision" >nul 2>nul
if errorlevel 1 (
  echo [WARN] Skipping deepfake training: torch/torchvision not available in this environment.
  echo [WARN] To enable deepfake CNN training, use Python 3.11/3.12 and run install_deep_learning_deps.bat
  goto :train_success
)

echo [INFO] Training EfficientNet-B0 Deepfake model if FaceForensics samples are available...
echo [INFO] Friendly profile: face crop, 8 frames/video, image size 160, max 15 epochs, early stopping enabled.
.venv\Scripts\python.exe -m ml_training.train_deepfake --dataset-dir datasets/faceforensics --batch-size 2 --epochs 15 --frames-per-video 8 --image-size 160 --unfreeze-last-blocks 2 --val-split 0.2 --early-stop-patience 4 --pretrained --skip-if-empty
if errorlevel 1 (
  echo [WARN] Deepfake training failed; continuing with URL/Email/SMS models.
)

:train_success
echo [SUCCESS] Model training complete.
echo You can now run run_client.bat
pause
exit /b 0

:train_error
echo [ERROR] Training failed.
pause
exit /b 1
