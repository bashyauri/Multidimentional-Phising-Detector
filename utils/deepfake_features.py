import tempfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def iter_media_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            yield path


def _frame_features(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    gray_hist = gray_hist / max(gray_hist.sum(), 1)

    color_hists = []
    for channel in range(3):
        hist = cv2.calcHist([frame], [channel], None, [8], [0, 256]).flatten()
        color_hists.extend(hist / max(hist.sum(), 1))

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    stats = [
        float(gray.mean()) / 255.0,
        float(gray.std()) / 255.0,
        float(hsv[:, :, 1].mean()) / 255.0,
        float(hsv[:, :, 1].std()) / 255.0,
        float(hsv[:, :, 2].mean()) / 255.0,
        float(hsv[:, :, 2].std()) / 255.0,
        min(float(laplacian_var) / 1000.0, 1.0),
        edge_density,
    ]

    return np.array([*gray_hist, *color_hists, *stats], dtype=np.float32)


def _extract_image_features(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    return _frame_features(frame)


def _extract_video_features(path: Path, max_frames: int = 12) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not read video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        positions = np.linspace(0, max(frame_count - 1, 0), max_frames, dtype=int)
    else:
        positions = np.arange(max_frames)

    features = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok and frame is not None:
            features.append(_frame_features(frame))

    capture.release()
    if not features:
        raise ValueError(f"No usable frames found in video: {path}")

    return np.mean(features, axis=0).astype(np.float32)


def extract_media_features(path: Path, max_frames: int = 12) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image_features(path)
    if suffix in VIDEO_EXTENSIONS:
        return _extract_video_features(path, max_frames=max_frames)
    raise ValueError(f"Unsupported media type: {path.suffix}")


def extract_media_features_from_bytes(file_bytes: bytes, filename: str, max_frames: int = 12) -> np.ndarray:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        return _frame_features(frame)

    if suffix in VIDEO_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return _extract_video_features(Path(tmp.name), max_frames=max_frames)

    raise ValueError(f"Unsupported media type: {suffix or 'unknown'}")
