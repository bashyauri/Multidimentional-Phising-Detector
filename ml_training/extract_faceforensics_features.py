import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

# Paths
DATASET_DIR = "datasets/faceforensics"
OUTPUT_CSV = "datasets/faceforensics/features_classical_cnn.csv"

# Feature extraction functions

def extract_classical_features(img):
    arr = np.array(img)
    # Color histogram (flattened)
    hist = []
    for i in range(3):
        h, _ = np.histogram(arr[..., i], bins=32, range=(0, 255), density=True)
        hist.append(h)
    hist = np.concatenate(hist)
    # Simple texture: mean, std per channel
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([hist, means, stds])

# CNN feature extractor
cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn.eval()
cnn = torch.nn.Sequential(*(list(cnn.children())[:-1]))  # Remove classifier
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_cnn_features(img):
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = cnn(x).squeeze().numpy()
    return feat

# Collect all images and labels
rows = []
for label_name, label in [("original", 0), ("manipulated", 1)]:
    folder = os.path.join(DATASET_DIR, label_name)
    for root, _, files in os.walk(folder):
        for fname in tqdm(files, desc=f"{label_name}"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    classical = extract_classical_features(img)
                    cnn_feat = extract_cnn_features(img)
                    features = np.concatenate([classical, cnn_feat])
                    rows.append(np.concatenate([[fpath], features, [label]]))
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

# Write to CSV
header = ["path"] + [f"hist_{i}" for i in range(96)] + ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"] + [f"cnn_{i}" for i in range(2048)] + ["label"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print(f"Saved features to {OUTPUT_CSV}")
