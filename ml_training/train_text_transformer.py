import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, find_column, normalize_label, save_confusion_plot, write_metrics


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"


def _load_text_dataset(dataset_path: Path, task: str) -> tuple[list[str], np.ndarray]:
    if task == "sms":
        try:
            df = pd.read_csv(dataset_path)
        except Exception:
            df = pd.read_csv(dataset_path, sep="\t", names=["label", "text"], header=None)
        text_candidates = ["text", "sms", "message", "content"]
    else:
        df = pd.read_csv(dataset_path)
        text_candidates = ["text", "email", "content", "message", "body"]

    text_col = find_column(df, text_candidates)
    label_col = find_column(df, ["label", "class", "target", "result"])

    if (not text_col or not label_col) and task == "sms" and df.shape[1] >= 2:
        df = pd.read_csv(dataset_path, sep="\t", names=["label", "text"], header=None)
        text_col = "text"
        label_col = "label"

    if not text_col or not label_col:
        raise ValueError(f"{task} dataset must contain text and label columns")

    x = df[text_col].astype(str).tolist()
    y = df[label_col].apply(normalize_label).to_numpy(dtype=np.int64)
    return x, y


def train_text_transformer(
    task: str,
    dataset_path: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    max_samples: int | None,
) -> None:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Install torch and transformers before training DistilBERT/BERT models."
        ) from exc

    x, y = _load_text_dataset(dataset_path, task)
    if max_samples:
        x = x[:max_samples]
        y = y[:max_samples]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        TextDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TextDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.item())
        print(f"epoch={epoch + 1}/{epochs} loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend((probs >= 0.5).astype(int).tolist())

    metrics = evaluate_model(y_test, np.array(y_pred), np.array(y_prob))
    metrics["model"] = f"{base_model} fine-tuned"
    metrics["dataset"] = str(dataset_path)
    metrics["max_length"] = max_length
    metrics["epochs"] = epochs

    out_dir = MODELS_DIR / f"{task}_distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))

    save_confusion_plot(metrics["confusion_matrix"], task)
    write_metrics(task, metrics)

    print(f"{task.upper()} transformer model trained successfully")
    print(f"Model saved to: {out_dir}")
    print(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DistilBERT/BERT text phishing detector")
    parser.add_argument("--task", choices=["email", "sms"], required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    default_dataset = DATASETS_DIR / ("email_phishing.csv" if args.task == "email" else "sms_spam.csv")
    train_text_transformer(
        task=args.task,
        dataset_path=Path(args.dataset) if args.dataset else default_dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
