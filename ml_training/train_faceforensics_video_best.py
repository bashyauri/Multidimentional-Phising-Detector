import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json

# Load the original, larger video features CSV
csv_path = "datasets/faceforensics/features_video_combined.csv"
df = pd.read_csv(csv_path)

# Prepare features and labels
X = df.drop(["path", "label"], axis=1)
y = df["label"].astype(int)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model candidates
models = {
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
}

# Hyperparameter grids
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", "log2", None]
    },
    "HistGradientBoosting": {
        "max_iter": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_leaf": [10, 20, 30],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_leaf_nodes": [31, 63, 127]
    },
    "MLP": {
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01]
    }
}

results = {}
for name, model in models.items():
    print(f"\nTuning and training {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "best_params": grid.best_params_,
        "model": name
    }
    results[name] = metrics
    print(f"{name} metrics: {metrics}")

# Save all results
with open("models/faceforensics_video_metrics_best.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll model results saved to models/faceforensics_video_metrics_best.json")
