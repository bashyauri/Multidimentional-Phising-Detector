import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import cv2

# Parameters
DATASET_DIR = "datasets/faceforensics"
OUTPUT_CSV = "datasets/faceforensics/features_video_classical_cnn.csv"
FRAMES_PER_VIDEO = 8

# Feature extraction functions
def extract_classical_features(img):
    arr = np.array(img)
    hist = []
    for i in range(3):
        h, _ = np.histogram(arr[..., i], bins=32, range=(0, 255), density=True)
        hist.append(h)
    hist = np.concatenate(hist)
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([hist, means, stds])

cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn.eval()
cnn = torch.nn.Sequential(*(list(cnn.children())[:-1]))
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

def sample_frames_from_video(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        indices = np.linspace(0, frame_count - 1, frame_count, dtype=int)
    else:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        frames.append(img)
    cap.release()
    return frames

rows = []
for label_name, label in [("original", 0), ("manipulated", 1)]:
    folder = os.path.join(DATASET_DIR, label_name)
    for root, _, files in os.walk(folder):
        for fname in tqdm(files, desc=f"{label_name}"):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                fpath = os.path.join(root, fname)
                try:
                    frames = sample_frames_from_video(fpath, FRAMES_PER_VIDEO)
                    classical_feats = []
                    cnn_feats = []
                    for img in frames:
                        classical_feats.append(extract_classical_features(img))
                        cnn_feats.append(extract_cnn_features(img))
                    if classical_feats and cnn_feats:
                        classical_agg = np.mean(classical_feats, axis=0)
                        cnn_agg = np.mean(cnn_feats, axis=0)
                        features = np.concatenate([classical_agg, cnn_agg])
                        rows.append(np.concatenate([[fpath], features, [label]]))
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

header = ["path"] + [f"hist_{i}" for i in range(96)] + ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"] + [f"cnn_{i}" for i in range(2048)] + ["label"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print(f"Saved features to {OUTPUT_CSV}")
