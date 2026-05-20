import io
import re

import cv2
import numpy as np
import pandas as pd

from utils.preprocessing import extract_url_features
from utils.qr_utils import decode_qr_image


QR_MODEL_FEATURE_COLUMNS = [
    "decoded_success",
    "decoded_length",
    "decoded_has_url",
    "url_model_available",
    "url_phishing_prob",
    "img_mean_intensity",
    "img_std_intensity",
    "img_black_ratio",
    "img_edge_density",
    "img_laplacian_var",
    "img_aspect_ratio",
    "img_area_log",
]


def decode_qr_text_from_bytes(file_bytes: bytes) -> str | None:
    return decode_qr_image(io.BytesIO(file_bytes))


def _extract_visual_features(file_bytes: bytes) -> dict[str, float]:
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode uploaded QR image")

    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    black_ratio = float((binary == 0).mean())
    edge_density = float((edges > 0).mean())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area_log = float(np.log1p(h * w))

    return {
        "img_mean_intensity": float(gray.mean()),
        "img_std_intensity": float(gray.std()),
        "img_black_ratio": black_ratio,
        "img_edge_density": edge_density,
        "img_laplacian_var": laplacian_var,
        "img_aspect_ratio": float(w / max(h, 1)),
        "img_area_log": area_log,
    }


def _normalized_url_candidate(decoded_text: str) -> tuple[str | None, bool]:
    text = (decoded_text or "").strip()
    if not text:
        return None, False

    has_url = bool(re.search(r"https?://|www\.|[a-z0-9.-]+\.[a-z]{2,}", text.lower()))
    if not has_url:
        return None, False

    if text.lower().startswith(("http://", "https://")):
        return text, True
    return f"https://{text}", True


def build_qr_feature_frame(
    file_bytes: bytes,
    url_model=None,
    decoded_text: str | None = None,
) -> pd.DataFrame:
    decoded = decoded_text if decoded_text is not None else decode_qr_text_from_bytes(file_bytes)
    normalized_url, has_url = _normalized_url_candidate(decoded or "")

    url_prob = 0.5
    url_model_available = 1 if url_model is not None else 0
    if url_model is not None and normalized_url:
        url_features = extract_url_features(normalized_url)
        expected_columns = list(getattr(url_model, "feature_names_in_", url_features.columns))
        url_features = url_features.reindex(columns=expected_columns, fill_value=0)
        url_prob = float(url_model.predict_proba(url_features)[0][1])

    row = {
        "decoded_success": 1 if decoded else 0,
        "decoded_length": len(decoded or ""),
        "decoded_has_url": 1 if has_url else 0,
        "url_model_available": url_model_available,
        "url_phishing_prob": float(url_prob),
    }
    row.update(_extract_visual_features(file_bytes))

    return pd.DataFrame([row], columns=QR_MODEL_FEATURE_COLUMNS)
