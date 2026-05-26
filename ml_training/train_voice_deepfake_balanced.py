import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json

# Load the balanced dataset
csv_path = r"datasets/voice/DATASET-balanced.csv"
df = pd.read_csv(csv_path)

# Prepare features and labels
X = df.drop('LABEL', axis=1)
y = df['LABEL'].map({'FAKE': 1, 'REAL': 0})  # Adjust if 'REAL' exists

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a simple classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "model": "RandomForestClassifier",
    "dataset": csv_path,
    "test_size": 0.2,
    "n_estimators": 100
}

# Save metrics
with open("models/voice_deepfake_metrics_balanced.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to models/voice_deepfake_metrics_balanced.json")
print(metrics)
