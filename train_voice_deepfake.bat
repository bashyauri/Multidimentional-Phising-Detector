@echo off
REM Train Voice Deepfake Detection (Windows Batch Script)
REM Assumes you have Python, torch, torchaudio, and dependencies installed

set DATASET_DIR=datasets/voice
set EPOCHS=10
set BATCH_SIZE=16
set DEVICE=cpu

python -m ml_training.train_voice_deepfake --dataset-dir %DATASET_DIR% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --device %DEVICE%

pause