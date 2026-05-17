import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelRegistry:
    def __init__(self) -> None:
        self.url_model = None
        self.email_model = None
        self.sms_model = None
        self.deepfake_model = None
        self.metrics = {}

    def load(self) -> None:
        self.url_model = self._safe_load(MODELS_DIR / "url_model.pkl")
        self.email_model = self._safe_load(MODELS_DIR / "email_model.pkl")
        self.sms_model = self._safe_load(MODELS_DIR / "sms_model.pkl")
        self.deepfake_model = self._safe_load(MODELS_DIR / "deepfake_model.pkl")

        metrics_path = MODELS_DIR / "metrics_summary.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @staticmethod
    def _safe_load(path: Path):
        if path.exists():
            return joblib.load(path)
        return None


registry = ModelRegistry()
registry.load()
