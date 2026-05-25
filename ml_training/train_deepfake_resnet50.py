import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.deepfake_cnn import (
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


def _predict_video_probabilities(
    model,
    samples,
    labels,
    device,
    torch,
    frames_per_video,
    image_size,
):
    y_true = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for path, label in zip(samples, labels):
            frames = frames_from_media_path(
                path,
                frames_per_video=frames_per_video,
                image_size=image_size,
            )

            batch = torch.tensor(
                np.stack(frames),
                dtype=torch.float32,
                device=device,
            )

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]

            video_prob = probs.mean().item()

            y_true.append(int(label))
            y_prob.append(float(video_prob))

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


def _set_batchnorm_eval(model, torch):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _augment_frame_array(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = frame.copy()
    if rng.random() < 0.5:
        augmented = augmented[:, :, ::-1]

    if rng.random() < 0.35:
        brightness = rng.normal(0.0, 0.035, size=(3, 1, 1)).astype(np.float32)
        augmented = augmented + brightness

    if rng.random() < 0.25:
        noise = rng.normal(0.0, 0.015, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise

    return augmented.astype(np.float32)


def build_resnet50(num_classes=2, pretrained=True):
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required."
        ) from exc

    weights = ResNet50_Weights.DEFAULT if pretrained else None

    try:
        model = resnet50(weights=weights)
    except Exception:
        print("[WARN] Could not load pretrained weights.")
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


def train_deepfake_resnet50(
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
    device_preference: str,
):

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Install torch and torchvision."
        ) from exc

    paths, labels = _collect_media(
        dataset_dir,
        max_samples_per_class,
    )

    class_counts = {
        int(label): int((labels == label).sum())
        for label in np.unique(labels)
    }

    if len(paths) == 0:
        raise ValueError("Dataset is empty.")

    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_paths, val_paths, y_train, y_val = train_test_split(
        trainval_paths,
        y_trainval,
        test_size=val_split,
        random_state=42,
        stratify=y_trainval,
    )

    device = resolve_torch_device(torch, device_preference)

    model = build_resnet50(
        num_classes=2,
        pretrained=pretrained,
    )

    model.to(device)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    # Unfreeze last ResNet blocks
    blocks = [model.layer4, model.layer3]

    remaining = unfreeze_last_blocks

    for block_group in blocks:
        if remaining <= 0:
            break

        children = list(block_group.children())

        take = min(len(children), remaining)

        for block in children[-take:]:
            for param in block.parameters():
                param.requires_grad = True

        remaining -= take

    classifier_params = [
        p for p in model.fc.parameters()
        if p.requires_grad
    ]

    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("fc.")
    ]

    optimizer_kwargs = {"weight_decay": 1e-4}
    if str(device).startswith("privateuseone"):
        # DirectML does not fully support foreach optimizer ops; disable them.
        optimizer_kwargs["foreach"] = False

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": 1e-5,
            },
            {
                "params": classifier_params,
                "lr": 1e-4,
            },
        ],
        **optimizer_kwargs,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    train_class_counts = np.bincount(y_train, minlength=2)

    class_weights = np.ones(2, dtype=np.float32)

    nonzero_mask = train_class_counts > 0

    class_weights[nonzero_mask] = (
        train_class_counts.sum()
        / (2.0 * train_class_counts[nonzero_mask])
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    best_state_dict = None
    best_auc = -1.0
    best_auc_raw = -1.0
    best_invert_probability = False
    best_threshold = 0.5
    best_validation_classification_score = -1.0
    best_monitor_score = -1.0

    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()

        _set_batchnorm_eval(model, torch)

        order = np.random.default_rng(
            42 + epoch
        ).permutation(len(train_paths))

        losses = []

        for start in range(0, len(order), batch_size):

            batch_idx = order[start:start + batch_size]

            frame_rows = []
            target_rows = []

            rng = np.random.default_rng(1000 + epoch * 100000 + start)

            for idx in batch_idx:

                frames = frames_from_media_path(
                    train_paths[idx],
                    frames_per_video=frames_per_video,
                    image_size=image_size,
                )

                frame_rows.extend(_augment_frame_array(frame, rng) for frame in frames)

                target_rows.extend(
                    [int(y_train[idx])] * len(frames)
                )

            x_batch = torch.tensor(
                np.stack(frame_rows),
                dtype=torch.float32,
                device=device,
            )

            y_batch = torch.tensor(
                target_rows,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad()

            logits = model(x_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(float(loss.item()))

            # Batch-level progress output
            print(f"epoch={epoch+1}/{epochs} batch={(start//batch_size)+1}/{(len(order)+batch_size-1)//batch_size} loss={loss.item():.4f}")

            del x_batch, y_batch, logits, loss

        scheduler.step()

        epoch_loss = float(np.mean(losses))

        # Validation
        y_val_true, y_val_prob = _predict_video_probabilities(
            model,
            val_paths,
            y_val,
            device,
            torch,
            frames_per_video,
            image_size,
        )

        try:
            val_auc_raw = float(
                roc_auc_score(
                    y_val_true,
                    y_val_prob,
                )
            )
        except Exception:
            val_auc_raw = 0.0

        invert_probability = val_auc_raw < 0.5
        y_val_prob_adj = 1.0 - y_val_prob if invert_probability else y_val_prob
        val_auc = 1.0 - val_auc_raw if invert_probability else val_auc_raw
        val_threshold, val_cls_score = _find_best_threshold(y_val_true, y_val_prob_adj)
        monitor_score = 0.6 * val_auc + 0.4 * val_cls_score

        if monitor_score > best_monitor_score:

            best_monitor_score = monitor_score
            best_auc = val_auc
            best_auc_raw = val_auc_raw
            best_invert_probability = invert_probability
            best_threshold = val_threshold
            best_validation_classification_score = val_cls_score

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch+1}/{epochs} "
            f"loss={epoch_loss:.4f} "
            f"val_auc={val_auc:.4f} "
            f"raw_auc={val_auc_raw:.4f} "
            f"val_thr={val_threshold:.3f} "
            f"val_cls={val_cls_score:.4f} "
            f"invert={invert_probability}"
        )

        # Debug probabilities
        print(
            f"val prob stats => "
            f"min={y_val_prob.min():.4f} "
            f"max={y_val_prob.max():.4f} "
            f"mean={y_val_prob.mean():.4f}"
        )

        if (
            early_stop_patience > 0
            and epochs_without_improvement >= early_stop_patience
        ):
            print("[INFO] Early stopping triggered.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final testing
    y_true, y_prob = _predict_video_probabilities(
        model,
        test_paths,
        y_test,
        device,
        torch,
        frames_per_video,
        image_size,
    )

    if best_invert_probability:
        y_prob = 1.0 - y_prob

    threshold = float(best_threshold)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_model(
        y_true,
        y_pred,
        y_prob,
    )

    metrics["model"] = "ResNet-50 deepfake detector"
    metrics["dataset"] = str(dataset_dir)
    metrics["class_counts"] = class_counts
    metrics["frames_per_video"] = frames_per_video
    metrics["image_size"] = image_size
    metrics["decision_threshold"] = threshold
    metrics["best_validation_auc"] = best_auc
    metrics["best_validation_auc_raw"] = best_auc_raw
    metrics["best_validation_classification_score"] = best_validation_classification_score
    metrics["best_monitor_score"] = best_monitor_score
    metrics["threshold_strategy"] = "max(0.7*balanced_accuracy + 0.3*macro_F1) over thresholds [0.2, 0.9]"
    metrics["invert_probability"] = bool(best_invert_probability)
    metrics["augmentation"] = "horizontal_flip + light_brightness_jitter + light_gaussian_noise"
    metrics["device"] = str(device)

    MODELS_DIR.mkdir(exist_ok=True)

    is_smoke_run = max_samples_per_class is not None and max_samples_per_class <= 20
    artifact_suffix = "_smoke" if is_smoke_run else ""

    model_name = f"deepfake_resnet50{artifact_suffix}"
    model_out = MODELS_DIR / f"{model_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frames_per_video": frames_per_video,
            "image_size": image_size,
            "decision_threshold": threshold,
            "invert_probability": bool(best_invert_probability),
            "architecture": "resnet50",
            "class_names": [
                "original",
                "manipulated",
            ],
        },
        model_out,
    )

    save_confusion_plot(metrics["confusion_matrix"], model_name)

    write_metrics(model_name, metrics)

    print("\nTraining completed successfully.")
    print(f"Model saved to: {model_out}")
    if is_smoke_run:
        print("[INFO] Smoke-test artifacts were written to separate *_smoke files.")
    print(metrics)

    return 0


def main():

    parser = argparse.ArgumentParser(
        description="Train ResNet50 deepfake detector"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DATASET_DIR),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--skip-if-empty",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "dml", "cpu"],
        default="auto",
    )

    args = parser.parse_args()

    return train_deepfake_resnet50(
        dataset_dir=Path(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
        max_samples_per_class=args.max_samples_per_class,
        pretrained=args.pretrained,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        skip_if_empty=args.skip_if_empty,
        device_preference=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
