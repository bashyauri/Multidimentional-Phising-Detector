import argparse
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, find_column, normalize_label, save_confusion_plot, write_metrics
from utils.preprocessing import URL_MODEL_FEATURE_COLUMNS, extract_url_features


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"


def train_url_model(dataset_path: Path):
    df = pd.read_csv(dataset_path)

    label_col = find_column(df, ["label", "result", "class", "target", "status"])
    if not label_col:
        raise ValueError("Could not find label column. Expected one of: label, result, class, target, status")

    y = df[label_col].apply(normalize_label)

    url_col = find_column(df, ["url", "domain", "link"])
    if url_col:
        feature_rows = df[url_col].astype(str).apply(extract_url_features)
        x = pd.concat(feature_rows.to_list(), ignore_index=True)
    else:
        drop_cols = [label_col]
        x = df.drop(columns=drop_cols)

    if not url_col:
        available = [col for col in URL_MODEL_FEATURE_COLUMNS if col in x.columns]
        if available:
            x = x.reindex(columns=URL_MODEL_FEATURE_COLUMNS, fill_value=0)

    # Drop text/metadata columns, keep only numeric features matching our schema
    TEXT_COLS = {"URL", "Domain", "TLD", "Title"}
    drop_cols = [c for c in df.columns if c == label_col or c in TEXT_COLS]
    x_all = df.drop(columns=drop_cols, errors="ignore")

    # Use only columns present in our feature schema (ensures training/inference consistency)
    available = [c for c in URL_MODEL_FEATURE_COLUMNS if c in x_all.columns]
    if available:
        x = x_all[available].reindex(columns=URL_MODEL_FEATURE_COLUMNS, fill_value=0)
    else:
        # Fallback: use all numeric precomputed columns
        x = x_all.select_dtypes(include="number")

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y)

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        )
        model_name = "XGBoost + URL feature engineering (70/15/15 split)"
    except ImportError:
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        )
        model_name = "HistGradientBoosting fallback + URL feature engineering (70/15/15 split)"

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = model_name
    metrics["dataset"] = str(dataset_path)
    metrics["split_method"] = "three-way (70% train, 15% validation, 15% test)"
    metrics["train_samples"] = len(x_train)
    metrics["validation_samples"] = len(x_val)
    metrics["test_samples"] = len(x_test)

    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "url_model_validation.pkl"
    joblib.dump(model, model_out)

    save_confusion_plot(metrics["confusion_matrix"], "url_validation")
    write_metrics("url_validation", metrics)

    print("URL model trained successfully with three-way split")
    print(f"Model saved to: {model_out}")
    print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}, Test samples: {len(x_test)}")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train URL phishing detection model")
    parser.add_argument("--dataset", type=str, default=str(DATASETS_DIR / "qr" / "PhiUSIIL_Phishing_URL_Dataset.csv"))
    args = parser.parse_args()
    train_url_model(Path(args.dataset))
