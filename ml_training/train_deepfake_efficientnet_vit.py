"""
Train a hybrid EfficientNet + Vision Transformer (ViT) model for deepfake detection.
Feature fusion: EfficientNet and ViT features are concatenated and classified.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml_training.common import evaluate_model, save_confusion_plot, write_metrics
from utils.deepfake_cnn import frames_from_media_path, resolve_torch_device

# Hybrid model definition
class EfficientNetViTFusion(nn.Module):
    def __init__(self, eff_name="efficientnet_b0", vit_name="vit_base_patch16_224", num_classes=2, pretrained=True):
        super().__init__()
        import timm
        self.eff = timm.create_model(eff_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        eff_dim = self.eff.num_features
        vit_dim = self.vit.num_features
        self.classifier = nn.Sequential(
            nn.Linear(eff_dim + vit_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        eff_feat = self.eff(x)
        vit_feat = self.vit(x)
        feat = torch.cat([eff_feat, vit_feat], dim=1)
        return self.classifier(feat)

def _collect_media(dataset_dir: Path, max_samples_per_class: int | None):
    class_dirs = [
        (dataset_dir / "original", 0),
        (dataset_dir / "manipulated", 1),
    ]
    paths = []
    labels = []
    for class_dir, label in class_dirs:
        files = list(class_dir.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv", ".webm"}]
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)

def _find_best_threshold(y_true, y_prob):
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

def train_deepfake_efficientnet_vit(
    dataset_dir: Path,
    epochs: int,
    batch_size: int,
    frames_per_video: int,
    image_size: int,
    val_split: float,
    early_stop_patience: int,
    max_samples_per_class: int | None,
    pretrained: bool,
    device_preference: str,
):
    paths, labels = _collect_media(dataset_dir, max_samples_per_class)
    class_counts = {int(label): int((labels == label).sum()) for label in np.unique(labels)}
    if len(paths) == 0 or len(class_counts) < 2 or min(class_counts.values(), default=0) < 2:
        raise ValueError("Deepfake dataset needs at least 2 original and 2 manipulated media files.")

    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    use_validation = len(trainval_paths) >= 10 and 0.0 < val_split < 0.5
    if use_validation:
        train_paths, val_paths, y_train, y_val = train_test_split(
            trainval_paths, y_trainval, test_size=val_split, random_state=42, stratify=y_trainval
        )
    else:
        train_paths, y_train = trainval_paths, y_trainval
        val_paths, y_val = np.array([], dtype=object), np.array([], dtype=np.int64)

    import timm
    device = resolve_torch_device(torch, device_preference)
    model = EfficientNetViTFusion(pretrained=pretrained)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = torch.nn.CrossEntropyLoss()

    best_state_dict = None
    best_val_score = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0

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
            print(f"  Batch {start//batch_size+1}/{(len(order)+batch_size-1)//batch_size} loss={loss.item():.4f}")
        scheduler.step()
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        if use_validation and len(val_paths) > 0:
            model.eval()
            y_val_true, y_val_prob = [], []
            with torch.no_grad():
                for path, label in zip(val_paths, y_val):
                    frames = frames_from_media_path(
                        path,
                        frames_per_video=frames_per_video,
                        image_size=image_size,
                    )
                    batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
                    logits = model(batch)
                    prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
                    y_val_prob.append(float(prob))
                    y_val_true.append(int(label))
            y_val_true = np.array(y_val_true)
            y_val_prob = np.array(y_val_prob)
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
    # Final testing
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for path, label in zip(test_paths, y_test):
            frames = frames_from_media_path(
                path,
                frames_per_video=frames_per_video,
                image_size=image_size,
            )
            batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
            y_prob.append(float(prob))
            y_true.append(int(label))
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    threshold = float(best_threshold)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = evaluate_model(y_true, y_pred, y_prob)
    metrics["model"] = "EfficientNet+ViT hybrid"
    metrics["dataset"] = str(dataset_dir)
    metrics["class_counts"] = class_counts
    metrics["frames_per_video"] = frames_per_video
    metrics["image_size"] = image_size
    metrics["decision_threshold"] = threshold
    metrics["best_validation_classification_score"] = best_val_score
    metrics["threshold_strategy"] = "max(0.7*balanced_accuracy + 0.3*macro_F1) over thresholds [0.2, 0.9]"
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    model_out = MODELS_DIR / "deepfake_efficientnet_vit.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "frames_per_video": frames_per_video,
        "image_size": image_size,
        "decision_threshold": threshold,
        "architecture": "efficientnet_vit",
        "class_names": ["original", "manipulated"],
    }, model_out)
    save_confusion_plot(metrics["confusion_matrix"], "deepfake_efficientnet_vit")
    write_metrics("deepfake_efficientnet_vit", metrics)
    print("\nTraining completed successfully.")
    print(f"Model saved to: {model_out}")
    print(metrics)
    return 0

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet+ViT hybrid deepfake detector")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device-preference", type=str, default="auto")
    args = parser.parse_args()
    train_deepfake_efficientnet_vit(
        dataset_dir=Path(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
        max_samples_per_class=args.max_samples_per_class,
        pretrained=args.pretrained,
        device_preference=args.device_preference,
    )

if __name__ == "__main__":
    main()
