@echo off
REM Train EfficientNet+LSTM Deepfake Detection (CPU)
set DATASET_DIR=datasets/faceforensics
set EPOCHS=10
set BATCH_SIZE=16
set DEVICE=cpu

python -m ml_training.train_deepfake_efficientnet_lstm --dataset-dir %DATASET_DIR% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --device %DEVICE%
pause