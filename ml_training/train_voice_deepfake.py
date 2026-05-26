"""
Voice Deepfake Detection Training Script
- Uses CNN on spectrograms (can be extended to LSTM/transformer)
- Ready for ASVspoof/FakeAVCeleb or similar datasets
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torchaudio
import os

class VoiceDeepfakeDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate=16000, duration=3.0):
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        mel = self.mel(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel, label

class SimpleVoiceCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def collect_voice_files(dataset_dir):
    real_dir = Path(dataset_dir) / "real"
    fake_dir = Path(dataset_dir) / "fake"
    real_files = list(real_dir.rglob("*.wav"))
    fake_files = list(fake_dir.rglob("*.wav"))
    files = real_files + fake_files
    labels = [0]*len(real_files) + [1]*len(fake_files)
    return files, labels

def train_voice_deepfake(dataset_dir, epochs=10, batch_size=16, lr=1e-3, device="cuda"):
    files, labels = collect_voice_files(dataset_dir)
    files = np.array(files)
    labels = np.array(labels)
    idx = np.random.permutation(len(files))
    files, labels = files[idx], labels[idx]
    split = int(0.8 * len(files))
    train_files, val_files = files[:split], files[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    train_ds = VoiceDeepfakeDataset(train_files, train_labels)
    val_ds = VoiceDeepfakeDataset(val_files, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = SimpleVoiceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    import json
    for epoch in range(epochs):
        model.train()
        train_losses = []
        batch_count = len(train_loader)
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.long())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            print(f"  Batch {batch_idx+1}/{batch_count} loss={loss.item():.4f}")
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(yb.cpu().numpy())
        acc = accuracy_score(val_targets, val_preds)
        f1 = f1_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{epochs} loss={np.mean(train_losses):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            # Save full model (not just state_dict) for consistency with other .pt files
            torch.save({
                "model_state_dict": model.state_dict(),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "architecture": "SimpleVoiceCNN",
                "class_names": ["real", "fake"],
            }, "models/voice_deepfake_cnn.pt")

    # Save metrics to JSON and confusion matrix plot
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
    import matplotlib.pyplot as plt
    cm = confusion_matrix(val_targets, val_preds, labels=[0,1])
    precision = precision_score(val_targets, val_preds, zero_division=0)
    recall = recall_score(val_targets, val_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(val_targets, val_preds)
    except Exception:
        roc_auc = None
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": roc_auc,
        "model": "SimpleVoiceCNN",
        "dataset": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    os.makedirs("models", exist_ok=True)
    with open("models/voice_deepfake_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix plot
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center", color="red")
    plt.xticks([0,1], ["Real", "Fake"])
    plt.yticks([0,1], ["Real", "Fake"])
    plt.tight_layout()
    plt.savefig("models/voice_deepfake_confusion.png")
    plt.close()

    print("Training complete. Best val_acc=", best_acc)
    print("Metrics saved to models/voice_deepfake_metrics.json")
    print("Confusion matrix plot saved to models/voice_deepfake_confusion.png")

def main():
    parser = argparse.ArgumentParser(description="Train Voice Deepfake Detector")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train_voice_deepfake(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

if __name__ == "__main__":
    main()
