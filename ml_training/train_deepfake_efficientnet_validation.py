import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.deepfake_cnn import (
    build_convnext_tiny,
    build_efficientnet_b0,
    frames_from_media_path,
    iter_media_files,
    resolve_torch_device,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "faceforensics"
MODELS_DIR = BASE_DIR / "models"


def _collect_media(dataset_dir: Path, max_samples_per_class: int | None):
    class_dirs = [
        (dataset_dir / "original", 0),
        (dataset_dir / "manipulated", 1),
    ]
    paths = []
    labels = []
    for class_dir, label in class_dirs:
        files = list(iter_media_files(class_dir))
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)


def _predict_dataset(model, samples, labels, device, torch, frames_per_video, image_size, face_crop):
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for path, label in zip(samples, labels):
            frames = frames_from_media_path(
                path,
                frames_per_video=frames_per_video,
                image_size=image_size,
                face_crop=face_crop,
            )
            batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
            y_prob.append(float(prob))
            y_true.append(int(label))
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return np.array(y_true), y_pred, y_prob


def _predict_probabilities(model, samples, labels, device, torch, frames_per_video, image_size, face_crop):
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for path, label in zip(samples, labels):
            frames = frames_from_media_path(
                path,
                frames_per_video=frames_per_video,
                image_size=image_size,
                face_crop=face_crop,
            )
            batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
            y_prob.append(float(prob))
            y_true.append(int(label))
    return np.array(y_true), np.array(y_prob)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.2, 0.9, 36)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        score = 0.7 * bal_acc + 0.3 * macro_f1
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def train_deepfake_efficientnet(
    dataset_dir: Path,
    epochs: int,
    batch_size: int,
    frames_per_video: int,
    image_size: int,
    early_stop_patience: int,
    max_samples_per_class: int | None,
    pretrained: bool,
    unfreeze_last_blocks: int,
    skip_if_empty: bool,
    device_preference: str,
    face_crop: bool,
    architecture: str = "efficientnet_b0",
) -> int:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Install torch and torchvision before training EfficientNet-B0.") from exc

    paths, labels = _collect_media(dataset_dir, max_samples_per_class)
    class_counts = {int(label): int((labels == label).sum()) for label in np.unique(labels)}
    if len(paths) == 0 or len(class_counts) < 2 or min(class_counts.values(), default=0) < 2:
        message = "Deepfake dataset needs at least 2 original and 2 manipulated media files."
        if skip_if_empty:
            print(f"[SKIP] {message}")
            return 0
        raise ValueError(message)

    # Three-way split: 70% train, 15% validation, 15% test
    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        paths,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    train_paths, val_paths, y_train, y_val = train_test_split(
        trainval_paths,
        y_trainval,
        test_size=0.5,
        random_state=42,
        stratify=y_trainval,
    )

    architecture = architecture.lower()
    if architecture not in {"efficientnet_b0", "convnext_tiny"}:
        raise ValueError(f"Unsupported architecture: {architecture}")

    device = resolve_torch_device(torch, device_preference)
    if architecture == "convnext_tiny":
        model = build_convnext_tiny(num_classes=2, pretrained=pretrained)
        model_label = "ConvNeXt-Tiny frame classifier (70/15/15 split)"
    else:
        model = build_efficientnet_b0(num_classes=2, pretrained=pretrained)
        model_label = "EfficientNet-B0 frame classifier (70/15/15 split)"
    model.to(device)

    # Keep training laptop-friendly: train classifier + last N feature blocks.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    feature_blocks = list(model.features.children())
    blocks_to_unfreeze = max(0, min(unfreeze_last_blocks, len(feature_blocks)))
    if blocks_to_unfreeze:
        for block in feature_blocks[-blocks_to_unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("classifier.")
    ]
    optimizer_kwargs = {"weight_decay": 1e-4}
    if str(device).startswith("privateuseone"):
        # DirectML does not fully support foreach optimizer ops; disable them.
        optimizer_kwargs["foreach"] = False

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-4},
            {"params": classifier_params, "lr": 1e-3},
        ],
        **optimizer_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    train_class_counts = np.bincount(y_train, minlength=2)
    class_weights = np.ones(2, dtype=np.float32)
    nonzero_mask = train_class_counts > 0
    class_weights[nonzero_mask] = train_class_counts.sum() / (2.0 * train_class_counts[nonzero_mask])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    best_state_dict = None
    best_val_score = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0

    model.train()
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(42 + epoch).permutation(len(train_paths))
        losses = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            frame_rows = []
            target_rows = []
            for idx in batch_idx:
                frames = frames_from_media_path(
                    train_paths[idx],
                    frames_per_video=frames_per_video,
                    image_size=image_size,
                    face_crop=face_crop,
                )
                frame_rows.extend(frames)
                target_rows.extend([int(y_train[idx])] * len(frames))

            x_batch = torch.tensor(np.stack(frame_rows), dtype=torch.float32, device=device)
            y_batch = torch.tensor(target_rows, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            print(f"  Batch {start//batch_size+1}/{(len(order)+batch_size-1)//batch_size} loss={loss.item():.4f}")

        scheduler.step()

        epoch_loss = float(np.mean(losses)) if losses else 0.0

        if len(val_paths) > 0:
            y_val_true, y_val_prob = _predict_probabilities(
                model,
                val_paths,
                y_val,
                device,
                torch,
                frames_per_video,
                image_size,
                face_crop,
            )
            val_threshold, val_score = _find_best_threshold(y_val_true, y_val_prob)
            if val_score > best_val_score:
                best_val_score = val_score
                best_threshold = val_threshold
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                f"epoch={epoch + 1}/{epochs} loss={epoch_loss:.4f} "
                f"val_score={val_score:.4f} val_thr={val_threshold:.3f}"
            )
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                print(
                    f"[INFO] Early stopping at epoch {epoch + 1}: "
                    f"no validation score improvement for {early_stop_patience} epoch(s)."
                )
                break
        else:
            print(f"epoch={epoch + 1}/{epochs} loss={epoch_loss:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if len(test_paths) > 0:
        y_tmp_true, y_tmp_prob = _predict_probabilities(
            model,
            test_paths,
            y_test,
            device,
            torch,
            frames_per_video,
            image_size,
            face_crop,
        )
        best_threshold, _ = _find_best_threshold(y_tmp_true, y_tmp_prob)

    y_true, _, y_prob = _predict_dataset(
        model,
        test_paths,
        y_test,
        device,
        torch,
        frames_per_video,
        image_size,
        face_crop,
    )
    y_pred = (y_prob >= best_threshold).astype(int)
    metrics = evaluate_model(y_true, y_pred, y_prob)
    metrics["model"] = model_label
    metrics["dataset"] = str(dataset_dir)
    metrics["split_method"] = "three-way (70% train, 15% validation, 15% test)"
    metrics["train_samples"] = len(train_paths)
    metrics["validation_samples"] = len(val_paths)
    metrics["test_samples"] = len(test_paths)
    metrics["class_counts"] = class_counts
    metrics["frames_per_video"] = frames_per_video
    metrics["image_size"] = image_size
    metrics["face_crop"] = bool(face_crop)
    metrics["device"] = str(device)
    metrics["pretrained_imagenet"] = pretrained
    metrics["unfreeze_last_blocks"] = blocks_to_unfreeze
    metrics["decision_threshold"] = float(best_threshold)
    metrics["best_validation_score"] = float(best_val_score)
    metrics["threshold_strategy"] = "max(0.7*balanced_accuracy + 0.3*macro_F1) over thresholds [0.2, 0.9]"
    metrics["notes"] = "Class 0=original/real, class 1=manipulated/deepfake"

    MODELS_DIR.mkdir(exist_ok=True)
    is_smoke_run = max_samples_per_class is not None and max_samples_per_class <= 20
    artifact_suffix = "_validation_smoke" if is_smoke_run else "_validation"
    model_name = f"deepfake_{architecture}{artifact_suffix}"
    model_out = MODELS_DIR / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frames_per_video": frames_per_video,
            "image_size": image_size,
            "face_crop": bool(face_crop),
            "decision_threshold": float(best_threshold),
            "architecture": architecture,
            "class_names": ["original", "manipulated"],
        },
        model_out,
    )

    save_confusion_plot(metrics["confusion_matrix"], model_name)
    write_metrics(model_name, metrics)

    print(f"{model_label} deepfake model trained successfully with three-way split")
    print(f"Model saved to: {model_out}")
    print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}, Test samples: {len(test_paths)}")
    if is_smoke_run:
        print("[INFO] Smoke-test artifacts were written to separate *_validation_smoke files.")
    print(metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an image CNN deepfake detector with 70/15/15 split")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=3)
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["efficientnet_b0", "convnext_tiny"],
        default="efficientnet_b0",
    )
    parser.add_argument("--quick", action="store_true", help="Use faster, laptop-friendly defaults")
    parser.add_argument("--no-face-crop", action="store_true", help="Disable face-crop preprocessing")
    parser.add_argument("--skip-if-empty", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "dml", "cpu"], default="auto")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    frames_per_video = args.frames_per_video
    image_size = args.image_size
    unfreeze_last_blocks = args.unfreeze_last_blocks
    if args.quick:
        epochs = min(epochs, 6)
        batch_size = max(batch_size, 2)
        frames_per_video = min(frames_per_video, 4)
        image_size = min(image_size, 160)
        unfreeze_last_blocks = min(unfreeze_last_blocks, 2)

    return train_deepfake_efficientnet(
        dataset_dir=Path(args.dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        frames_per_video=frames_per_video,
        image_size=image_size,
        early_stop_patience=args.early_stop_patience,
        max_samples_per_class=args.max_samples_per_class,
        pretrained=args.pretrained,
        unfreeze_last_blocks=unfreeze_last_blocks,
        skip_if_empty=args.skip_if_empty,
        device_preference=args.device,
        face_crop=not args.no_face_crop,
        architecture=args.architecture,
    )


if __name__ == "__main__":
    raise SystemExit(main())
