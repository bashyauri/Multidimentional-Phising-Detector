import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ml_training.common import evaluate_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _source_group_from_path(path_value: str) -> str:
    """Map manipulated/original files to a shared source identity.

    Examples:
    - original/000.mp4 -> 000
    - manipulated/000_123.mp4 -> 000
    """
    stem = Path(str(path_value)).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _pick_grouped_split(X, y, groups, test_size: float, random_state: int):
    # Convert test_size into a reasonable number of folds for one holdout split.
    n_splits = int(round(1.0 / max(0.05, min(0.5, test_size))))
    n_splits = max(2, min(10, n_splits))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    return train_idx, test_idx, n_splits


def train_grouped(
    csv_path: Path,
    model_type: str,
    test_size: float,
    random_state: int,
    max_rows: int | None,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"path", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain 'path' and 'label' columns.")

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["group_id"] = df["path"].map(_source_group_from_path)

    X = df.drop(columns=["path", "label", "group_id"])
    y = df["label"]
    groups = df["group_id"]

    train_idx, test_idx, n_splits = _pick_grouped_split(X, y, groups, test_size, random_state)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        scaler = None
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # HistGradientBoosting benefits from scaled mixed-range features.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = HistGradientBoostingClassifier(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    metrics["model"] = "RandomForestClassifier" if model_type == "rf" else "HistGradientBoostingClassifier"
    metrics["dataset"] = str(csv_path).replace("\\", "/")
    metrics["split_strategy"] = "StratifiedGroupKFold(one-fold holdout)"
    metrics["n_splits"] = int(n_splits)
    metrics["test_size_target"] = float(test_size)
    metrics["rows"] = int(len(df))
    metrics["group_count"] = int(df["group_id"].nunique())
    metrics["label_counts"] = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}

    metrics_path = MODELS_DIR / "faceforensics_video_metrics_grouped.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "grouping": "group_id from path stem (prefix before underscore)",
        "threshold": 0.5,
    }
    model_path = MODELS_DIR / "faceforensics_video_grouped_model.pkl"
    joblib.dump(model_bundle, model_path)

    print("Grouped training complete")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FaceForensics deepfake model with grouped split (leakage-safe)")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(BASE_DIR / "datasets" / "faceforensics" / "features_video_classical_cnn_balanced.csv"),
        help="Path to features CSV (must include path,label).",
    )
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb", help="Classifier type.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Approximate holdout fraction.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick experiments.")
    args = parser.parse_args()

    return train_grouped(
        csv_path=Path(args.csv_path),
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
