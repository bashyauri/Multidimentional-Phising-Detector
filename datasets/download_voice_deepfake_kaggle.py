"""
Download and organize a Kaggle voice deepfake dataset for training.
- Requires: pip install kaggle
- Place kaggle.json (API key) in ~/.kaggle or /content/.kaggle for Colab
"""
import os
import zipfile
import shutil

# Kaggle dataset name (update as needed)
KAGGLE_DATASET = "ejlok1/real-and-fake-voice-detection"
DOWNLOAD_DIR = "voice_kaggle_raw"
TARGET_DIR = "voice"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

# Download dataset
os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DOWNLOAD_DIR} --unzip")

# Organize files
real_dir = os.path.join(TARGET_DIR, "real")
fake_dir = os.path.join(TARGET_DIR, "fake")
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Move files (dataset-specific logic)
for root, dirs, files in os.walk(DOWNLOAD_DIR):
    for file in files:
        fpath = os.path.join(root, file)
        if file.lower().startswith("real"):  # e.g., real_1.wav
            shutil.move(fpath, os.path.join(real_dir, file))
        elif file.lower().startswith("fake"):  # e.g., fake_1.wav
            shutil.move(fpath, os.path.join(fake_dir, file))

print("Download and organization complete. Files in:", TARGET_DIR)
