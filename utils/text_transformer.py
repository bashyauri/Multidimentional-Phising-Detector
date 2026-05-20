from pathlib import Path

import numpy as np


class TextTransformerUnavailable(RuntimeError):
    pass


class TransformerTextClassifier:
    def __init__(self, model_dir: Path, max_length: int = 256, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise TextTransformerUnavailable(
                "PyTorch and transformers are required for DistilBERT/BERT text models."
            ) from exc

        if not model_dir.exists():
            raise FileNotFoundError(model_dir)

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.expects_clean_text = False
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probs = self.torch.softmax(logits, dim=1).detach().cpu().numpy()

        if probs.shape[1] == 1:
            phishing = probs[:, 0]
            return np.column_stack([1 - phishing, phishing])
        return probs
