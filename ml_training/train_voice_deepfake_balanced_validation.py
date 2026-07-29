import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_training.common import save_confusion_plot, write_metrics

# Load the balanced dataset
csv_path = r"datasets/voice/DATASET-balanced.csv"
df = pd.read_csv(csv_path)

# Prepare features and labels
X = df.drop('LABEL', axis=1)
y = df['LABEL'].map({'FAKE': 1, 'REAL': 0})  # Adjust if 'REAL' exists

# Train/validation/test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Train a simple classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "models/voice_model_balanced_validation.pkl")

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "confusion_matrix": cm.tolist(),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "model": "RandomForestClassifier (70/15/15 split)",
    "dataset": csv_path,
    "split_method": "three-way (70% train, 15% validation, 15% test)",
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "test_size": 0.15,
    "n_estimators": 100
}

# Save metrics
with open("models/voice_deepfake_metrics_balanced_validation.json", "w") as f:
    json.dump(metrics, f, indent=2)

save_confusion_plot(metrics["confusion_matrix"], "voice_validation")
write_metrics("voice_validation", metrics)

print("Model saved to models/voice_model_balanced_validation.pkl")
print("Metrics saved to models/voice_deepfake_metrics_balanced_validation.json")
print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
print(metrics)
