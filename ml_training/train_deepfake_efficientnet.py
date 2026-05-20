import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.deepfake_cnn import build_efficientnet_b0, frames_from_media_path, iter_media_files


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


def _predict_dataset(model, samples, labels, device, torch, frames_per_video, image_size):
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for path, label in zip(samples, labels):
            frames = frames_from_media_path(path, frames_per_video=frames_per_video, image_size=image_size)
            batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
            y_prob.append(float(prob))
            y_true.append(int(label))
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return np.array(y_true), y_pred, y_prob


def _predict_probabilities(model, samples, labels, device, torch, frames_per_video, image_size):
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for path, label in zip(samples, labels):
            frames = frames_from_media_path(path, frames_per_video=frames_per_video, image_size=image_size)
            batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
            y_prob.append(float(prob))
            y_true.append(int(label))
    return np.array(y_true), np.array(y_prob)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    # Keep the threshold in a stable range to avoid degenerate all-positive predictions.
    candidates = np.linspace(0.3, 0.8, 21)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        score = 0.7 * f1 + 0.3 * bal_acc
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
    val_split: float,
    early_stop_patience: int,
    max_samples_per_class: int | None,
    pretrained: bool,
    unfreeze_last_blocks: int,
    skip_if_empty: bool,
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

    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    use_validation = len(trainval_paths) >= 10 and 0.0 < val_split < 0.5
    if use_validation:
        train_paths, val_paths, y_train, y_val = train_test_split(
            trainval_paths,
            y_trainval,
            test_size=val_split,
            random_state=42,
            stratify=y_trainval,
        )
    else:
        train_paths, y_train = trainval_paths, y_trainval
        val_paths, y_val = np.array([], dtype=object), np.array([], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_b0(num_classes=2, pretrained=pretrained)
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
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-4},
            {"params": classifier_params, "lr": 1e-3},
        ],
        weight_decay=1e-4,
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

        scheduler.step()

        epoch_loss = float(np.mean(losses)) if losses else 0.0

        if use_validation and len(val_paths) > 0:
            y_val_true, y_val_prob = _predict_probabilities(
                model,
                val_paths,
                y_val,
                device,
                torch,
                frames_per_video,
                image_size,
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

    if not use_validation and len(test_paths) > 0:
        y_tmp_true, y_tmp_prob = _predict_probabilities(
            model,
            test_paths,
            y_test,
            device,
            torch,
            frames_per_video,
            image_size,
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
    )
    y_pred = (y_prob >= best_threshold).astype(int)
    metrics = evaluate_model(y_true, y_pred, y_prob)
    metrics["model"] = "EfficientNet-B0 frame classifier"
    metrics["dataset"] = str(dataset_dir)
    metrics["class_counts"] = class_counts
    metrics["frames_per_video"] = frames_per_video
    metrics["image_size"] = image_size
    metrics["pretrained_imagenet"] = pretrained
    metrics["unfreeze_last_blocks"] = blocks_to_unfreeze
    metrics["decision_threshold"] = float(best_threshold)
    metrics["validation_split"] = float(val_split if use_validation else 0.0)
    if use_validation:
        metrics["best_validation_score"] = float(best_val_score)
    metrics["threshold_strategy"] = "max(0.7*F1 + 0.3*balanced_accuracy) over thresholds [0.3, 0.8]"
    metrics["notes"] = "Class 0=original/real, class 1=manipulated/deepfake"

    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "deepfake_efficientnet_b0.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frames_per_video": frames_per_video,
            "image_size": image_size,
            "decision_threshold": float(best_threshold),
            "class_names": ["original", "manipulated"],
        },
        model_out,
    )

    save_confusion_plot(metrics["confusion_matrix"], "deepfake")
    write_metrics("deepfake", metrics)

    legacy_model = MODELS_DIR / "deepfake_model.pkl"
    if legacy_model.exists():
        legacy_model.rename(MODELS_DIR / "deepfake_model_legacy_random_forest.pkl")

    print("EfficientNet-B0 deepfake model trained successfully")
    print(f"Model saved to: {model_out}")
    print(metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 deepfake detector")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Use faster, laptop-friendly defaults")
    parser.add_argument("--skip-if-empty", action="store_true")
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
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
        max_samples_per_class=args.max_samples_per_class,
        pretrained=args.pretrained,
        unfreeze_last_blocks=unfreeze_last_blocks,
        skip_if_empty=args.skip_if_empty,
    )


if __name__ == "__main__":
    raise SystemExit(main())
