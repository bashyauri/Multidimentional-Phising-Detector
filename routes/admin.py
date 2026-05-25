"""
Flask Admin Model Management UI (prototype)
- Allows admin to upload new model/metrics files
- Trigger plot regeneration and model reload
- For research/demo use only (not for public deployment)
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import requests

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

MODEL_FOLDER = "models"
PLOT_FOLDER = os.path.join("static", "plots")
FLASK_RELOAD_URL = "http://127.0.0.1:5000/api/reload-models"

@admin_bp.route("/", methods=["GET", "POST"])
def admin_panel():
    if request.method == "POST":
        # Handle file uploads
        model_file = request.files.get("model_file")
        metrics_file = request.files.get("metrics_file")
        messages = []
        if model_file and model_file.filename:
            dst = os.path.join(MODEL_FOLDER, model_file.filename)
            model_file.save(dst)
            messages.append(f"Uploaded model: {model_file.filename}")
        if metrics_file and metrics_file.filename:
            dst = os.path.join(MODEL_FOLDER, metrics_file.filename)
            metrics_file.save(dst)
            messages.append(f"Uploaded metrics: {metrics_file.filename}")
        for msg in messages:
            flash(msg)
        return redirect(url_for("admin.admin_panel"))
    # List models and metrics
    files = os.listdir(MODEL_FOLDER)
    return render_template("admin_panel.html", files=files)

@admin_bp.route("/regenerate-plot", methods=["POST"])
def regenerate_plot():
    metrics_name = request.form.get("metrics_name")
    plot_name = request.form.get("plot_name")
    if not metrics_name or not plot_name:
        flash("Missing metrics or plot name.")
        return redirect(url_for("admin.admin_panel"))
    metrics_path = os.path.join(MODEL_FOLDER, metrics_name)
    plot_path = os.path.join(PLOT_FOLDER, plot_name)
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        cm = np.array(metrics["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Deepfake"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(plot_name)
        plt.savefig(plot_path)
        plt.close()
        flash(f"Plot {plot_name} updated.")
    except Exception as e:
        flash(f"Error: {e}")
    return redirect(url_for("admin.admin_panel"))

@admin_bp.route("/reload-models", methods=["POST"])
def reload_models():
    try:
        resp = requests.post(FLASK_RELOAD_URL)
        if resp.ok:
            flash("Models reloaded successfully.")
        else:
            flash(f"Model reload failed: {resp.status_code}")
    except Exception as e:
        flash(f"Could not reload models: {e}")
    return redirect(url_for("admin.admin_panel"))
