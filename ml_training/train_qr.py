import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.qr_features import build_qr_feature_frame, decode_qr_text_from_bytes


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "qr"
MODELS_DIR = BASE_DIR / "models"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _resolve_qr_layout(dataset_dir: Path) -> tuple[Path, Path]:
    candidates = [dataset_dir]
    candidates.extend(path for path in dataset_dir.iterdir() if path.is_dir())

    for base in candidates:
        benign = base / "benign_qr_images_500"
        malicious = base / "malicious_qr_images_500"
        if benign.exists() and malicious.exists():
            return benign, malicious

    raise FileNotFoundError(
        "Could not find benign_qr_images_500 and malicious_qr_images_500 under datasets/qr"
    )


def _iter_images(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def train_qr_model(dataset_dir: Path, max_samples_per_class: int | None = None) -> None:
    benign_dir, malicious_dir = _resolve_qr_layout(dataset_dir)

    url_model = None
    url_model_path = MODELS_DIR / "url_model.pkl"
    if url_model_path.exists():
        url_model = joblib.load(url_model_path)

    rows = []
    labels = []
    decoded_count = 0

    for class_dir, label in ((benign_dir, 0), (malicious_dir, 1)):
        files = list(_iter_images(class_dir))
        if max_samples_per_class:
            files = files[:max_samples_per_class]

        for image_path in files:
            file_bytes = image_path.read_bytes()
            decoded_text = decode_qr_text_from_bytes(file_bytes)
            if decoded_text:
                decoded_count += 1

            features = build_qr_feature_frame(
                file_bytes=file_bytes,
                url_model=url_model,
                decoded_text=decoded_text,
            )
            rows.append(features.iloc[0].to_dict())
            labels.append(label)

    if not rows or len(set(labels)) < 2:
        raise ValueError("QR dataset must contain both benign and malicious images")

    x = pd.DataFrame(rows).fillna(0)
    y = np.array(labels, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    class_counts = np.bincount(y_fit, minlength=2)
    scale_pos_weight = float(class_counts[0] / max(class_counts[1], 1)) if class_counts[1] else 1.0

    model_name = "XGBoost + QR decode/image features"
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            gamma=0.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=2,
        )
        try:
            model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except TypeError:
            model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(x_fit, y_fit)
        model_name = "RandomForest + QR decode/image features"

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = model_name
    metrics["dataset"] = str(dataset_dir)
    metrics["decoded_rate"] = round(decoded_count / max(len(rows), 1), 6)
    metrics["samples"] = int(len(rows))
    metrics["validation_samples"] = int(len(x_val))

    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "qr_model.pkl"
    joblib.dump(model, model_out)

    save_confusion_plot(metrics["confusion_matrix"], "qr")
    write_metrics("qr", metrics)

    print("QR model trained successfully")
    print(f"Model saved to: {model_out}")
    print(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train QR phishing detector")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    args = parser.parse_args()

    train_qr_model(
        dataset_dir=Path(args.dataset_dir),
        max_samples_per_class=args.max_samples_per_class,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
