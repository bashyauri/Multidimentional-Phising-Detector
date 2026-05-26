import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "static" / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(value):
    text = str(value).strip().lower()
    phishing_tokens = {"phishing", "malicious", "spam", "smishing", "bad", "1", "true", "yes"}
    legitimate_tokens = {"legitimate", "ham", "safe", "good", "0", "false", "no", "benign"}

    if text in phishing_tokens:
        return 1
    if text in legitimate_tokens:
        return 0

    try:
        num = float(text)
        return 1 if num >= 1 else 0
    except ValueError:
        return 0


def find_column(df, candidates):
    columns_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None


def evaluate_model(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def save_confusion_plot(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_metrics(model_name, metrics):
    single_path = MODELS_DIR / f"{model_name}_metrics.json"
    with open(single_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_path = MODELS_DIR / "metrics_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary[model_name] = metrics

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
