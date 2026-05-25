import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from utils.deepfake_cnn import frames_from_media_path, iter_media_files, resolve_torch_device
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetLSTM(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256, num_layers=1, num_classes=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.feature_extractor = nn.Sequential(*(list(base.children())[:-2]))  # Remove classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        b, seq, C, H, W = x.shape
        x = x.view(b * seq, C, H, W)
        feats = self.feature_extractor(x)  # (b*seq, 1280, H', W')
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  # (b*seq, 1280)
        feats = feats.view(b, seq, -1)  # (b, seq, 1280)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]  # Last time step
        logits = self.classifier(out)
        return logits

def collect_media(dataset_dir: Path, max_samples_per_class: int | None):
    class_dirs = [
        (dataset_dir / "original", 0),
        (dataset_dir / "manipulated", 1),
    ]
    paths, labels = [], []
    for class_dir, label in class_dirs:
        files = list(iter_media_files(class_dir))
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        paths.extend(files)
        labels.extend([label] * len(files))
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = resolve_torch_device(torch, args.device)
    paths, labels = collect_media(Path(args.dataset_dir), args.max_samples_per_class)
    train_paths, val_paths, y_train, y_val = train_test_split(
        paths, labels, test_size=args.val_split, random_state=42, stratify=labels
    )

    model = EfficientNetLSTM(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def make_batch(paths, labels):
        batch_x, batch_y = [], []
        for path, label in zip(paths, labels):
            frames = frames_from_media_path(path, args.frames_per_video, args.image_size)
            batch_x.append(np.stack(frames))
            batch_y.append(label)
        x = torch.tensor(np.stack(batch_x), dtype=torch.float32, device=device)
        y = torch.tensor(batch_y, dtype=torch.long, device=device)
        return x, y

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(train_paths))
        train_paths, y_train = train_paths[perm], y_train[perm]
        num_batches = (len(train_paths) + args.batch_size - 1) // args.batch_size
        for batch_idx, i in enumerate(range(0, len(train_paths), args.batch_size), 1):
            x_batch, y_batch = make_batch(train_paths[i:i+args.batch_size], y_train[i:i+args.batch_size])
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch+1}/{args.epochs} batch={batch_idx}/{num_batches} loss={loss.item():.4f}")
        print(f"Epoch {epoch+1}/{args.epochs} done.")

    # Validation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for i in range(0, len(val_paths), args.batch_size):
            x_batch, y_batch = make_batch(val_paths[i:i+args.batch_size], y_val[i:i+args.batch_size])
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probs[:,1].cpu().numpy())
    acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"Validation Balanced Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
