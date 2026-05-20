@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Please run setup_client.bat first.
  pause
  exit /b 1
)

echo [INFO] Training SMS DistilBERT model.
.venv\Scripts\python.exe -m ml_training.train_text_transformer --task sms --dataset datasets/sms_spam.csv --base-model distilbert-base-uncased --epochs 2 --batch-size 4 --max-length 160
if errorlevel 1 goto :train_error

echo [SUCCESS] SMS DistilBERT training complete.
echo Restart the app or call POST /api/reload-models before testing.
pause
exit /b 0

:train_error
echo [ERROR] SMS DistilBERT training failed.
echo Make sure torch and transformers are installed.
pause
exit /b 1
