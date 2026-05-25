"""
Automate model/metrics update, plot regeneration, and model reload for Flask app.
- Copy new model and metrics files into models/
- Regenerate confusion matrix plot for EfficientNet-B0 (extend as needed)
- Reload models via Flask API
"""
import shutil
import requests
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# --- CONFIG ---
MODEL_SRC = "deepfake_efficientnet_b0.pt"  # Update as needed
METRICS_SRC = "deepfake_efficientnet_b0_metrics.json"  # Update as needed
MODEL_DST = os.path.join("models", MODEL_SRC)
METRICS_DST = os.path.join("models", METRICS_SRC)
PLOT_PATH = os.path.join("static", "plots", "deepfake_efficientnet_b0_confusion_matrix.png")
FLASK_RELOAD_URL = "http://127.0.0.1:5000/api/reload-models"

# --- COPY FILES ---
def copy_file(src, dst):
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"[WARN] {src} not found, skipping.")

copy_file(MODEL_SRC, MODEL_DST)
copy_file(METRICS_SRC, METRICS_DST)

# --- REGENERATE PLOT ---
def regenerate_confusion_matrix(metrics_path, plot_path):
    if not os.path.exists(metrics_path):
        print(f"[WARN] Metrics file {metrics_path} not found, skipping plot.")
        return
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    cm = np.array(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Deepfake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("EfficientNet-B0 Confusion Matrix (Latest)")
    plt.savefig(plot_path)
    plt.close()
    print(f"Updated plot: {plot_path}")

regenerate_confusion_matrix(METRICS_DST, PLOT_PATH)

# --- RELOAD MODELS VIA API ---
def reload_models():
    try:
        resp = requests.post(FLASK_RELOAD_URL)
        if resp.ok:
            print("Models reloaded successfully.")
        else:
            print(f"[ERROR] Model reload failed: {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Could not reload models: {e}")

reload_models()

print("All steps completed.")
