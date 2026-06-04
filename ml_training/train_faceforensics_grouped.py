import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from ml_training.common import evaluate_model, write_metrics


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def extract_ids_from_path(path_value: str) -> list[str]:
    stem = Path(str(path_value)).stem
    ids = re.findall(r"\d+", stem)
    return ids if ids else [stem]


def build_groups(paths: pd.Series) -> np.ndarray:
    # Group by source ID (first token) to keep originals and their manipulations together.
    # Example: original 123.mp4 and manipulated 123_456.mp4 both map to group 123.
    groups = []
    for p in paths:
        ids = extract_ids_from_path(p)
        groups.append(ids[0])
    return np.array(groups, dtype=object)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return 0.7 * float(bal_acc) + 0.3 * float(macro_f1)


def _find_valid_group_split(X, y, groups, test_size: float, random_state: int):
    for offset in range(100):
        seed = random_state + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
            return train_idx, test_idx, seed
    raise RuntimeError(
        "Could not find a valid grouped split with both classes in train and test after 100 attempts. "
        "Use a different CSV or adjust test-size."
    )


def train_grouped(csv_path: Path, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    y = df["label"].astype(int).to_numpy()
    groups = build_groups(df["path"]) 
    X = df.drop(columns=["path", "label"]).to_numpy(dtype=np.float32)

    train_idx, test_idx, split_seed = _find_valid_group_split(X, y, groups, test_size, random_state)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
    rf_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, rf_pred))
    rf_metrics["macro_f1"] = float(f1_score(y_test, rf_pred, average="macro", zero_division=0))
    rf_metrics["objective"] = score_predictions(y_test, rf_pred)
    rf_metrics["model"] = "RandomForestClassifier (grouped split)"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=random_state)
    hgb.fit(X_train_s, y_train)
    hgb_pred = hgb.predict(X_test_s)
    hgb_prob = hgb.predict_proba(X_test_s)[:, 1]
    hgb_metrics = evaluate_model(y_test, hgb_pred, hgb_prob)
    hgb_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, hgb_pred))
    hgb_metrics["macro_f1"] = float(f1_score(y_test, hgb_pred, average="macro", zero_division=0))
    hgb_metrics["objective"] = score_predictions(y_test, hgb_pred)
    hgb_metrics["model"] = "HistGradientBoostingClassifier (grouped split)"

    comparison = {
        "dataset": str(csv_path).replace("\\", "/"),
        "rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_class_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "test_class_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()},
        "unique_groups": int(len(np.unique(groups))),
        "train_groups": int(len(np.unique(groups[train_idx]))),
        "test_groups": int(len(np.unique(groups[test_idx]))),
        "split_random_state_used": int(split_seed),
        "rf": rf_metrics,
        "hgb": hgb_metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = MODELS_DIR / "faceforensics_grouped_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    if hgb_metrics["objective"] >= rf_metrics["objective"]:
        best_name = "deepfake_grouped_hgb"
        joblib.dump({"model": hgb, "scaler": scaler}, MODELS_DIR / f"{best_name}.pkl")
        best_metrics = hgb_metrics
    else:
        best_name = "deepfake_grouped_rf"
        joblib.dump(rf, MODELS_DIR / f"{best_name}.pkl")
        best_metrics = rf_metrics

    best_metrics["dataset"] = str(csv_path).replace("\\", "/")
    best_metrics["split_strategy"] = "GroupShuffleSplit by source ID (first token of filename)"
    best_metrics["groups_total"] = int(len(np.unique(groups)))
    best_metrics["train_groups"] = int(len(np.unique(groups[train_idx])))
    best_metrics["test_groups"] = int(len(np.unique(groups[test_idx])))
    best_metrics["split_random_state_used"] = int(split_seed)
    best_metrics["selected_artifact"] = f"models/{best_name}.pkl"
    write_metrics("deepfake_grouped", best_metrics)

    print("Saved:")
    print(f"- {comparison_path}")
    print(f"- {MODELS_DIR / (best_name + '.pkl')}")
    print(f"- {MODELS_DIR / 'deepfake_grouped_metrics.json'}")
    print("Selected model objective:", round(best_metrics["objective"], 6))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FaceForensics models with leakage-safe grouped split")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="datasets/faceforensics/features_video_classical_cnn_balanced.csv",
        help="Feature CSV path with columns: path, features..., label",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_grouped(Path(args.csv_path), test_size=args.test_size, random_state=args.random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
