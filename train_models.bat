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

echo [INFO] Training Email model...
.venv\Scripts\python.exe -m ml_training.train_email --dataset datasets/email_phishing.csv
if errorlevel 1 goto :train_error

echo [INFO] Training SMS model...
.venv\Scripts\python.exe -m ml_training.train_sms --dataset datasets/sms_spam.csv
if errorlevel 1 goto :train_error

echo [INFO] Training Deepfake model if FaceForensics samples are available...
.venv\Scripts\python.exe -m ml_training.train_deepfake --dataset-dir datasets/faceforensics --skip-if-empty
if errorlevel 1 goto :train_error

echo [SUCCESS] Model training complete.
echo You can now run run_client.bat
pause
exit /b 0

:train_error
echo [ERROR] Training failed.
pause
exit /b 1
