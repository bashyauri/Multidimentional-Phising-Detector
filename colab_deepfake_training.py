"""
Google Colab Script for Deepfake Model Training
This script trains an improved deepfake detection model using ConvNeXt-Tiny
with enhanced parameters to achieve 90%+ accuracy.

Instructions:
1. Upload this script to Google Colab
2. Upload your faceforensics dataset to Google Drive
3. Run the script sections in order
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

# =============================================================================
# SECTION 1: SETUP AND DEPENDENCIES
# =============================================================================

def setup_environment():
    """Install required dependencies and setup environment."""
    print("Installing dependencies...")
    !pip install torch torchvision opencv-python-headless scikit-learn tqdm
    !pip install timm  # For ConvNeXt architecture
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    return torch

# =============================================================================
# SECTION 2: MOUNT GOOGLE DRIVE AND SETUP PATHS
# =============================================================================

def setup_paths():
    """Mount Google Drive and setup file paths."""
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Update these paths based on your Google Drive structure
    DRIVE_PATH = Path('/content/drive/MyDrive')
    DATASET_PATH = DRIVE_PATH / 'faceforensics'  # Update this path
    OUTPUT_PATH = DRIVE_PATH / 'deepfake_models'
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    return DATASET_PATH, OUTPUT_PATH

# =============================================================================
# SECTION 3: DATA PREPARATION
# =============================================================================

def collect_media_files(dataset_path, max_samples_per_class=None):
    """Collect video/image files from dataset directory."""
    import cv2
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    
    class_dirs = [
        (dataset_path / "original", 0),
        (dataset_path / "manipulated", 1),
    ]
    
    paths = []
    labels = []
    
    for class_dir, label in class_dirs:
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
            
        files = sorted(class_dir.rglob("*"))
        media_files = [f for f in files if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS]
        
        if max_samples_per_class:
            media_files = media_files[:max_samples_per_class]
        
        paths.extend(media_files)
        labels.extend([label] * len(media_files))
        print(f"Found {len(media_files)} files in {class_dir.name}")
    
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)

# =============================================================================
# SECTION 4: MODEL ARCHITECTURE
# =============================================================================

def build_convnext_tiny(num_classes=2, pretrained=True):
    """Build ConvNeXt-Tiny model for deepfake detection."""
    import timm
    
    model = timm.create_model(
        'convnext_tiny',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3
    )
    return model

def build_efficientnet_b0(num_classes=2, pretrained=True):
    """Build EfficientNet-B0 model for deepfake detection."""
    import timm
    
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3
    )
    return model

# =============================================================================
# SECTION 5: DATA PREPROCESSING
# =============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(frame, image_size=224):
    """Preprocess a single frame for model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(frame, (2, 0, 1))

def extract_frames_from_video(video_path, frames_per_video=8, image_size=224):
    """Extract frames from video file."""
    import cv2
    
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not read video: {video_path}")
    
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count == 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frames evenly
    indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    frames = []
    
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if ret:
            frames.append(preprocess_frame(frame, image_size))
    
    capture.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not extract frames from: {video_path}")
    
    return frames

def extract_frames_from_image(image_path, image_size=224):
    """Extract frame from image file."""
    import cv2
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    return [preprocess_frame(frame, image_size)]

def frames_from_media_path(path, frames_per_video=8, image_size=224):
    """Extract frames from either video or image file."""
    suffix = path.suffix.lower()
    
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return extract_frames_from_image(path, image_size)
    elif suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return extract_frames_from_video(path, frames_per_video, image_size)
    else:
        raise ValueError(f"Unsupported media type: {path.suffix}")

# =============================================================================
# SECTION 6: TRAINING FUNCTIONS
# =============================================================================

def find_best_threshold(y_true, y_prob):
    """Find optimal decision threshold."""
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

def predict_dataset(model, samples, labels, device, frames_per_video, image_size):
    """Predict on dataset and return metrics."""
    model.eval()
    y_prob = []
    y_true = []
    
    with torch.no_grad():
        for path, label in zip(samples, labels):
            try:
                frames = frames_from_media_path(path, frames_per_video, image_size)
                batch = torch.tensor(np.stack(frames), dtype=torch.float32, device=device)
                logits = model(batch)
                prob = torch.softmax(logits, dim=1)[:, 1].mean().item()
                y_prob.append(float(prob))
                y_true.append(int(label))
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
    
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_true, y_pred, y_prob

def train_deepfake_model(
    dataset_path,
    output_path,
    architecture="convnext_tiny",
    epochs=20,
    batch_size=4,
    frames_per_video=8,
    image_size=224,
    val_split=0.2,
    early_stop_patience=3,
    max_samples_per_class=None,
    pretrained=True,
    unfreeze_last_blocks=4,
    device="cuda"
):
    """Train deepfake detection model with improved parameters."""
    
    print(f"Training {architecture} model with improved parameters")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Frames per video: {frames_per_video}, Image size: {image_size}")
    print(f"Unfreeze blocks: {unfreeze_last_blocks}")
    
    # Collect data
    paths, labels = collect_media_files(dataset_path, max_samples_per_class)
    class_counts = {int(label): int((labels == label).sum()) for label in np.unique(labels)}
    
    if len(paths) == 0 or len(class_counts) < 2:
        raise ValueError("Dataset needs at least 2 original and 2 manipulated media files.")
    
    print(f"Total samples: {len(paths)}")
    print(f"Class counts: {class_counts}")
    
    # Split data
    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, y_train, y_val = train_test_split(
        trainval_paths, y_trainval, test_size=val_split, random_state=42, stratify=y_trainval
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Build model
    if architecture == "convnext_tiny":
        model = build_convnext_tiny(num_classes=2, pretrained=pretrained)
        model_name = "ConvNeXt-Tiny deepfake detector"
    elif architecture == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes=2, pretrained=pretrained)
        model_name = "EfficientNet-B0 deepfake detector"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    model = model.to(device)
    
    # Freeze backbone, unfreeze classifier and last N blocks
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    
    # Unfreeze last N feature blocks
    if hasattr(model, 'stages'):  # ConvNeXt
        feature_blocks = model.stages
        blocks_to_unfreeze = min(unfreeze_last_blocks, len(feature_blocks))
        for block in feature_blocks[-blocks_to_unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    elif hasattr(model, 'features'):  # EfficientNet
        feature_blocks = list(model.features.children())
        blocks_to_unfreeze = min(unfreeze_last_blocks, len(feature_blocks))
        for block in feature_blocks[-blocks_to_unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    
    # Setup optimizer
    classifier_params = [p for p in model.head.parameters() if p.requires_grad] if hasattr(model, 'head') else []
    backbone_params = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("head.")]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-4},
        {"params": classifier_params, "lr": 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Class weights for imbalanced data
    train_class_counts = np.bincount(y_train, minlength=2)
    class_weights = np.ones(2, dtype=np.float32)
    nonzero_mask = train_class_counts > 0
    class_weights[nonzero_mask] = train_class_counts.sum() / (2.0 * train_class_counts[nonzero_mask])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    
    # Training loop
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
                try:
                    frames = frames_from_media_path(train_paths[idx], frames_per_video, image_size)
                    frame_rows.extend(frames)
                    target_rows.extend([int(y_train[idx])] * len(frames))
                except Exception as e:
                    print(f"Error loading {train_paths[idx]}: {e}")
                    continue
            
            if len(frame_rows) == 0:
                continue
            
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
        
        # Validation
        y_val_true, y_val_pred, y_val_prob = predict_dataset(
            model, val_paths, y_val, device, frames_per_video, image_size
        )
        val_threshold, val_score = find_best_threshold(y_val_true, y_val_prob)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = val_threshold
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Val Score={val_score:.4f}, Best={best_val_score:.4f}")
        
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    # Test evaluation
    y_test_true, y_test_pred, y_test_prob = predict_dataset(
        model, test_paths, y_test, device, frames_per_video, image_size
    )
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    y_test_pred_final = (y_test_prob >= best_threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test_true, y_test_pred_final)),
        "precision": float(precision_score(y_test_true, y_test_pred_final, zero_division=0)),
        "recall": float(recall_score(y_test_true, y_test_pred_final, zero_division=0)),
        "f1": float(f1_score(y_test_true, y_test_pred_final, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_true, y_test_pred_final).tolist(),
        "roc_auc": float(roc_auc_score(y_test_true, y_test_prob)),
        "model": model_name,
        "dataset": str(dataset_path),
        "class_counts": class_counts,
        "frames_per_video": frames_per_video,
        "image_size": image_size,
        "pretrained_imagenet": pretrained,
        "unfreeze_last_blocks": unfreeze_last_blocks,
        "decision_threshold": float(best_threshold),
        "validation_split": float(val_split),
        "best_validation_score": float(best_val_score),
        "epochs_trained": epoch + 1,
        "threshold_strategy": "max(0.7*balanced_accuracy + 0.3*macro_F1) over thresholds [0.2, 0.9]"
    }
    
    # Save model
    model_path = output_path / f"deepfake_{architecture}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "frames_per_video": frames_per_video,
        "image_size": image_size,
        "decision_threshold": float(best_threshold),
        "architecture": architecture,
        "class_names": ["original", "manipulated"],
        "metrics": metrics
    }, model_path)
    
    # Save metrics
    metrics_path = output_path / f"deepfake_{architecture}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics

# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Setup
    torch = setup_environment()
    dataset_path, output_path = setup_paths()
    
    # Train model with improved parameters
    model, metrics = train_deepfake_model(
        dataset_path=dataset_path,
        output_path=output_path,
        architecture="convnext_tiny",  # Use ConvNeXt-Tiny for better performance
        epochs=20,  # More epochs for better training
        batch_size=4,  # Adjust based on GPU memory
        frames_per_video=8,  # More frames for better temporal analysis
        image_size=224,  # Higher resolution
        val_split=0.2,
        early_stop_patience=3,
        max_samples_per_class=None,  # Use all available data
        pretrained=True,
        unfreeze_last_blocks=4,  # More fine-tuning capacity
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\nTraining completed successfully!")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Target achieved: {'YES' if metrics['accuracy'] >= 0.90 else 'NO'}")

if __name__ == "__main__":
    main()
