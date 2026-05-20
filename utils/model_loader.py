import json
from pathlib import Path

import joblib

from utils.deepfake_cnn import DeepfakeCNNDetector, DeepfakeCNNUnavailable
from utils.text_transformer import TextTransformerUnavailable, TransformerTextClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelRegistry:
    def __init__(self) -> None:
        self.url_model = None
        self.qr_model = None
        self.email_model = None
        self.sms_model = None
        self.email_transformer_model = None
        self.sms_transformer_model = None
        self.deepfake_model = None
        self.deepfake_cnn_model = None
        self.metrics = {}

    def load(self) -> None:
        self.url_model = self._safe_load(MODELS_DIR / "url_model.pkl")
        self.qr_model = self._safe_load(MODELS_DIR / "qr_model.pkl")
        self.email_model = self._safe_load(MODELS_DIR / "email_model.pkl")
        self.sms_model = self._safe_load(MODELS_DIR / "sms_model.pkl")
        self.email_transformer_model = self._safe_load_transformer(MODELS_DIR / "email_distilbert")
        self.sms_transformer_model = self._safe_load_transformer(MODELS_DIR / "sms_distilbert")
        self.deepfake_cnn_model = self._safe_load_deepfake_cnn(MODELS_DIR / "deepfake_efficientnet_b0.pt")
        self.deepfake_model = None

        metrics_path = MODELS_DIR / "metrics_summary.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @staticmethod
    def _safe_load(path: Path):
        if path.exists():
            return joblib.load(path)
        return None

    @staticmethod
    def _safe_load_transformer(path: Path):
        try:
            if path.exists():
                return TransformerTextClassifier(path)
        except (FileNotFoundError, TextTransformerUnavailable, OSError):
            return None
        return None

    @staticmethod
    def _safe_load_deepfake_cnn(path: Path):
        try:
            if path.exists():
                return DeepfakeCNNDetector(path)
        except (FileNotFoundError, DeepfakeCNNUnavailable, OSError, RuntimeError):
            return None
        return None


registry = ModelRegistry()
registry.load()
