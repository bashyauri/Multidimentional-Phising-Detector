import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.deepfake_features import extract_media_features, iter_media_files


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "faceforensics"
MODELS_DIR = BASE_DIR / "models"


def _load_split_samples(dataset_dir: Path, max_samples_per_class: int | None, max_frames: int):
    class_dirs = [
        (dataset_dir / "original", 0),
        (dataset_dir / "manipulated", 1),
    ]

    x_rows = []
    y_rows = []
    skipped = []

    for class_dir, label in class_dirs:
        files = list(iter_media_files(class_dir))
        if max_samples_per_class:
            files = files[:max_samples_per_class]

        for path in files:
            try:
                x_rows.append(extract_media_features(path, max_frames=max_frames))
                y_rows.append(label)
            except Exception as exc:
                skipped.append(f"{path}: {exc}")

    if not x_rows:
        return np.empty((0, 0)), np.array([], dtype=int), skipped

    return np.vstack(x_rows), np.array(y_rows, dtype=int), skipped


def train_deepfake_model(
    dataset_dir: Path,
    max_samples_per_class: int | None = None,
    max_frames: int = 12,
    skip_if_empty: bool = False,
) -> int:
    x, y, skipped = _load_split_samples(dataset_dir, max_samples_per_class, max_frames)

    if skipped:
        print("[WARN] Some media files could not be processed:")
        for item in skipped[:20]:
            print(f" - {item}")
        if len(skipped) > 20:
            print(f" - ... {len(skipped) - 20} more")

    class_counts = {int(label): int((y == label).sum()) for label in np.unique(y)}
    if x.size == 0 or len(class_counts) < 2 or min(class_counts.values(), default=0) < 2:
        message = (
            "Deepfake dataset needs at least 2 usable original samples and "
            "2 usable manipulated samples."
        )
        if skip_if_empty:
            print(f"[SKIP] {message}")
            return 0
        raise ValueError(message)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = "RandomForestClassifier"
    metrics["dataset"] = str(dataset_dir)
    metrics["class_counts"] = class_counts
    metrics["max_frames_per_video"] = max_frames
    metrics["notes"] = "Class 0=original/real, class 1=manipulated/deepfake"

    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "deepfake_model.pkl"
    joblib.dump(model, model_out)

    save_confusion_plot(metrics["confusion_matrix"], "deepfake")
    write_metrics("deepfake", metrics)

    print("Deepfake model trained successfully")
    print(f"Model saved to: {model_out}")
    print(metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train deepfake detector from FaceForensics-style folders")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--skip-if-empty", action="store_true")
    args = parser.parse_args()

    return train_deepfake_model(
        Path(args.dataset_dir),
        max_samples_per_class=args.max_samples_per_class,
        max_frames=args.max_frames,
        skip_if_empty=args.skip_if_empty,
    )


if __name__ == "__main__":
    raise SystemExit(main())
