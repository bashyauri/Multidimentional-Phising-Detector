@echo off
REM Train EfficientNet + ViT Hybrid Deepfake Detection (CPU)
set DATASET_DIR=datasets/faceforensics
set EPOCHS=10
set BATCH_SIZE=16
set DEVICE=cpu

python -m ml_training.train_deepfake_efficientnet_vit --dataset-dir %DATASET_DIR% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --device-preference %DEVICE%
pause