@echo off
REM Train and evaluate on the balanced voice dataset using RandomForest
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe ml_training\train_voice_deepfake_balanced.py
) else (
  python ml_training\train_voice_deepfake_balanced.py
)
pause
