import tempfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepfakeCNNUnavailable(RuntimeError):
    pass


def iter_media_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            yield path


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(frame, (2, 0, 1))


def frames_from_media_path(path: Path, frames_per_video: int, image_size: int) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        return [_preprocess_frame(frame, image_size)]

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported media type: {path.suffix}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not read video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        positions = np.linspace(0, max(frame_count - 1, 0), frames_per_video, dtype=int)
    else:
        positions = np.arange(frames_per_video)

    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(_preprocess_frame(frame, image_size))

    capture.release()
    if not frames:
        raise ValueError(f"No usable frames found in video: {path}")
    return frames


def frames_from_media_bytes(
    file_bytes: bytes,
    filename: str,
    frames_per_video: int,
    image_size: int,
) -> list[np.ndarray]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        return [_preprocess_frame(frame, image_size)]

    if suffix in VIDEO_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return frames_from_media_path(Path(tmp.name), frames_per_video, image_size)

    raise ValueError(f"Unsupported media type: {suffix or 'unknown'}")


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = False):
    try:
        import torch
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except Exception as exc:
        raise DeepfakeCNNUnavailable(
            "PyTorch and torchvision are required for the EfficientNet deepfake model."
        ) from exc

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = efficientnet_b0(weights=weights)
    except Exception:
        if not pretrained:
            raise
        print("[WARN] Could not load/download ImageNet EfficientNet weights; training from random weights.")
        model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


class DeepfakeCNNDetector:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        try:
            import torch
        except Exception as exc:
            raise DeepfakeCNNUnavailable(
                "PyTorch is not installed, so the EfficientNet deepfake model cannot load."
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.torch = torch
        self.device = torch.device(device)
        artifact = torch.load(model_path, map_location=self.device)
        self.image_size = int(artifact.get("image_size", 224))
        self.frames_per_video = int(artifact.get("frames_per_video", 8))
        self.decision_threshold = float(artifact.get("decision_threshold", 0.5))

        self.model = build_efficientnet_b0(num_classes=2, pretrained=False)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_probability(self, file_bytes: bytes, filename: str) -> float:
        frames = frames_from_media_bytes(
            file_bytes,
            filename,
            frames_per_video=self.frames_per_video,
            image_size=self.image_size,
        )
        batch = self.torch.tensor(np.stack(frames), dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            logits = self.model(batch)
            probs = self.torch.softmax(logits, dim=1)[:, 1]
        return float(probs.mean().item())
