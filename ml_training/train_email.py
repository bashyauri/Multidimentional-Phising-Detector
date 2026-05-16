import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_training.common import evaluate_model, find_column, normalize_label, save_confusion_plot, write_metrics
from utils.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"


def train_email_model(dataset_path: Path):
    df = pd.read_csv(dataset_path)

    text_col = find_column(df, ["text", "email", "content", "message", "body"])
    label_col = find_column(df, ["label", "class", "target", "result"])

    if not text_col or not label_col:
        raise ValueError("Dataset must contain text/email column and label column")

    x = df[text_col].astype(str).apply(clean_text)
    y = df[label_col].apply(normalize_label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = "LogisticRegression + TF-IDF"
    metrics["dataset"] = str(dataset_path)

    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "email_model.pkl"
    joblib.dump(pipeline, model_out)

    save_confusion_plot(metrics["confusion_matrix"], "email")
    write_metrics("email", metrics)

    print("Email model trained successfully")
    print(f"Model saved to: {model_out}")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train email phishing detection model")
    parser.add_argument("--dataset", type=str, default=str(DATASETS_DIR / "email_phishing.csv"))
    args = parser.parse_args()
    train_email_model(Path(args.dataset))
