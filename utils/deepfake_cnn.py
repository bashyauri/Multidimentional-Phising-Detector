import tempfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_FACE_CASCADE = None


class DeepfakeCNNUnavailable(RuntimeError):
    pass


def iter_media_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            yield path


def _load_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(cascade_path))
    return _FACE_CASCADE


def _center_square_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    y0 = max((height - side) // 2, 0)
    x0 = max((width - side) // 2, 0)
    return frame[y0:y0 + side, x0:x0 + side]


def _crop_largest_face(frame: np.ndarray, margin: float = 0.35) -> np.ndarray:
    cascade = _load_face_cascade()
    if cascade.empty():
        return _center_square_crop(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return _center_square_crop(frame)

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    side = int(max(width, height) * (1.0 + margin))
    center_x = x + width // 2
    center_y = y + height // 2
    x0 = max(center_x - side // 2, 0)
    y0 = max(center_y - side // 2, 0)
    x1 = min(x0 + side, frame.shape[1])
    y1 = min(y0 + side, frame.shape[0])
    x0 = max(x1 - side, 0)
    y0 = max(y1 - side, 0)
    return frame[y0:y1, x0:x1]


def _preprocess_frame(frame: np.ndarray, image_size: int, face_crop: bool = False) -> np.ndarray:
    if face_crop:
        frame = _crop_largest_face(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(frame, (2, 0, 1))


def frames_from_media_path(
    path: Path,
    frames_per_video: int,
    image_size: int,
    face_crop: bool = False,
) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        return [_preprocess_frame(frame, image_size, face_crop=face_crop)]

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
            frames.append(_preprocess_frame(frame, image_size, face_crop=face_crop))

    capture.release()
    if not frames:
        raise ValueError(f"No usable frames found in video: {path}")
    return frames


def frames_from_media_bytes(
    file_bytes: bytes,
    filename: str,
    frames_per_video: int,
    image_size: int,
    face_crop: bool = False,
) -> list[np.ndarray]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image")
        return [_preprocess_frame(frame, image_size, face_crop=face_crop)]

    if suffix in VIDEO_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return frames_from_media_path(
                Path(tmp.name),
                frames_per_video,
                image_size,
                face_crop=face_crop,
            )

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


def build_convnext_tiny(num_classes: int = 2, pretrained: bool = False):
    try:
        import torch
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    except Exception as exc:
        raise DeepfakeCNNUnavailable(
            "PyTorch and torchvision are required for the ConvNeXt-Tiny deepfake model."
        ) from exc

    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    try:
        model = convnext_tiny(weights=weights)
    except Exception:
        if not pretrained:
            raise
        print("[WARN] Could not load/download ImageNet ConvNeXt-Tiny weights; training from random weights.")
        model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int = 2, pretrained: bool = False):
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception as exc:
        raise DeepfakeCNNUnavailable(
            "PyTorch and torchvision are required for the ResNet-50 deepfake model."
        ) from exc

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    try:
        model = resnet50(weights=weights)
    except Exception:
        if not pretrained:
            raise
        print("[WARN] Could not load/download ImageNet ResNet-50 weights; training from random weights.")
        model = resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, num_classes),
    )
    return model


def resolve_torch_device(torch, preferred: str = "auto"):
    preferred = (preferred or "auto").lower()
    if preferred not in {"auto", "cpu", "cuda", "dml"}:
        raise ValueError(f"Unsupported device preference: {preferred}")

    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA-capable device is available.")
        return torch.device("cuda")

    if preferred == "dml":
        try:
            import torch_directml
        except Exception as exc:
            raise RuntimeError(
                "DirectML requested but torch-directml is not installed. "
                "Install it with: pip install torch-directml"
            ) from exc
        return torch_directml.device()

    # auto mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


class DeepfakeCNNDetector:
    def __init__(self, model_path: Path, device: str = "auto", architecture: str | None = None) -> None:
        try:
            import torch
        except Exception as exc:
            raise DeepfakeCNNUnavailable(
                "PyTorch is not installed, so the EfficientNet deepfake model cannot load."
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.torch = torch
        self.device = resolve_torch_device(torch, device)
        try:
            artifact = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(model_path, map_location="cpu")
        self.image_size = int(artifact.get("image_size", 224))
        self.frames_per_video = int(artifact.get("frames_per_video", 8))
        self.face_crop = bool(artifact.get("face_crop", False))
        self.decision_threshold = float(artifact.get("decision_threshold", 0.5))
        self.invert_probability = bool(artifact.get("invert_probability", False))
        self.architecture = architecture or artifact.get("architecture")
        if not self.architecture:
            self.architecture = "resnet50" if "resnet50" in model_path.stem.lower() else "efficientnet_b0"

        if self.architecture == "resnet50":
            self.model = build_resnet50(num_classes=2, pretrained=False)
        elif self.architecture == "efficientnet_b0":
            self.model = build_efficientnet_b0(num_classes=2, pretrained=False)
        elif self.architecture == "convnext_tiny":
            self.model = build_convnext_tiny(num_classes=2, pretrained=False)
        else:
            raise DeepfakeCNNUnavailable(f"Unsupported deepfake CNN architecture: {self.architecture}")
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_probability(self, file_bytes: bytes, filename: str) -> float:
        frames = frames_from_media_bytes(
            file_bytes,
            filename,
            frames_per_video=self.frames_per_video,
            image_size=self.image_size,
            face_crop=self.face_crop,
        )
        batch = self.torch.tensor(np.stack(frames), dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            logits = self.model(batch)
            probs = self.torch.softmax(logits, dim=1)[:, 1]
        prob = float(probs.mean().item())
        if self.invert_probability:
            prob = 1.0 - prob
        return prob
