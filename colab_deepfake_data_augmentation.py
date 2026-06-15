"""
Google Colab Script for Deepfake Dataset Augmentation
This script helps expand your deepfake dataset from 2000 to 5000 videos
using data augmentation techniques.

Instructions:
1. Upload this script to Google Colab
2. Mount Google Drive with your existing faceforensics dataset
3. Run the script to augment your dataset
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# =============================================================================
# SECTION 1: SETUP LOCAL PATHS
# =============================================================================

def setup_paths():
    """Setup local file paths for Windows."""
    # Update these paths based on your local directory structure
    BASE_PATH = Path(__file__).parent  # Project root (Phising folder)
    DATASET_PATH = BASE_PATH / 'datasets' / 'faceforensics'
    OUTPUT_PATH = BASE_PATH / 'datasets' / 'faceforensics_augmented'
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    return DATASET_PATH, OUTPUT_PATH

# =============================================================================
# SECTION 2: VIDEO AUGMENTATION FUNCTIONS
# =============================================================================

def augment_video_frame(frame, augmentation_type):
    """Apply augmentation to a single frame."""
    if augmentation_type == "horizontal_flip":
        return cv2.flip(frame, 1)
    
    elif augmentation_type == "brightness":
        # Random brightness adjustment
        factor = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif augmentation_type == "contrast":
        # Random contrast adjustment
        factor = random.uniform(0.8, 1.2)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * factor, 0, 255)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif augmentation_type == "gaussian_blur":
        # Light Gaussian blur
        return cv2.GaussianBlur(frame, (3, 3), 0.5)
    
    elif augmentation_type == "rotation":
        # Small rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    
    elif augmentation_type == "noise":
        # Add light Gaussian noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    else:
        return frame

def augment_video(input_path, output_path, num_augmented=2):
    """Create augmented versions of a video."""
    cap = cv2.VideoCapture(str(input_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    augmentation_types = [
        "horizontal_flip", "brightness", "contrast", 
        "gaussian_blur", "rotation", "noise"
    ]
    
    for i in range(num_augmented):
        aug_type = random.choice(augmentation_types)
        output_file = output_path / f"{input_path.stem}_aug{i+1}_{aug_type}{input_path.suffix}"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        # Process each frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply augmentation
            aug_frame = augment_video_frame(frame, aug_type)
            out.write(aug_frame)
        
        out.release()
    
    cap.release()
    return num_augmented

def augment_image(input_path, output_path, num_augmented=2):
    """Create augmented versions of an image."""
    augmentation_types = [
        "horizontal_flip", "brightness", "contrast", 
        "gaussian_blur", "rotation", "noise"
    ]
    
    for i in range(num_augmented):
        aug_type = random.choice(augmentation_types)
        output_file = output_path / f"{input_path.stem}_aug{i+1}_{aug_type}{input_path.suffix}"
        
        frame = cv2.imread(str(input_path))
        aug_frame = augment_video_frame(frame, aug_type)
        cv2.imwrite(str(output_file), aug_frame)
    
    return num_augmented

# =============================================================================
# SECTION 3: DATASET EXPANSION
# =============================================================================

def expand_dataset(dataset_path, output_path, target_per_class=2500, augment_per_video=2):
    """Expand dataset to target number of samples per class."""
    
    # Create output directories
    original_output = output_path / "original"
    manipulated_output = output_path / "manipulated"
    original_output.mkdir(exist_ok=True)
    manipulated_output.mkdir(exist_ok=True)
    
    # Collect existing files
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    
    original_files = list((dataset_path / "original").rglob("*"))
    original_files = [f for f in original_files if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS]
    
    manipulated_files = list((dataset_path / "manipulated").rglob("*"))
    manipulated_files = [f for f in manipulated_files if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS]
    
    print(f"Original files: {len(original_files)}")
    print(f"Manipulated files: {len(manipulated_files)}")
    print(f"Target per class: {target_per_class}")
    
    # Calculate how many augmented samples needed
    original_needed = max(0, target_per_class - len(original_files))
    manipulated_needed = max(0, target_per_class - len(manipulated_files))
    
    print(f"Original augmented needed: {original_needed}")
    print(f"Manipulated augmented needed: {manipulated_needed}")
    
    # Copy original files first
    print("Copying original files...")
    for file in tqdm(original_files):
        output_file = original_output / file.name
        if not output_file.exists():
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(file))
                cv2.imwrite(str(output_file), frame)
            else:
                # Copy video file
                import shutil
                shutil.copy(file, output_file)
    
    print("Copying manipulated files...")
    for file in tqdm(manipulated_files):
        output_file = manipulated_output / file.name
        if not output_file.exists():
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(file))
                cv2.imwrite(str(output_file), frame)
            else:
                # Copy video file
                import shutil
                shutil.copy(file, output_file)
    
    # Augment original files
    if original_needed > 0:
        print(f"\nAugmenting original files (need {original_needed} more)...")
        aug_per_video = max(1, original_needed // len(original_files) + 1)
        
        for file in tqdm(original_files):
            if len(list(original_output.glob("*"))) >= target_per_class:
                break
            
            try:
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    augment_image(file, original_output, num_augmented=aug_per_video)
                else:
                    augment_video(file, original_output, num_augmented=aug_per_video)
            except Exception as e:
                print(f"Error augmenting {file}: {e}")
    
    # Augment manipulated files
    if manipulated_needed > 0:
        print(f"\nAugmenting manipulated files (need {manipulated_needed} more)...")
        aug_per_video = max(1, manipulated_needed // len(manipulated_files) + 1)
        
        for file in tqdm(manipulated_files):
            if len(list(manipulated_output.glob("*"))) >= target_per_class:
                break
            
            try:
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    augment_image(file, manipulated_output, num_augmented=aug_per_video)
                else:
                    augment_video(file, manipulated_output, num_augmented=aug_per_video)
            except Exception as e:
                print(f"Error augmenting {file}: {e}")
    
    # Count final results
    final_original = len(list(original_output.glob("*")))
    final_manipulated = len(list(manipulated_output.glob("*")))
    
    print(f"\nFinal dataset size:")
    print(f"Original: {final_original}")
    print(f"Manipulated: {final_manipulated}")
    print(f"Total: {final_original + final_manipulated}")
    
    return final_original, final_manipulated

# =============================================================================
# SECTION 4: DOWNLOAD ADDITIONAL DATA FROM FACEFORENSICS++
# =============================================================================

def download_faceforensics_samples(output_path, num_samples_per_class=500):
    """
    Download additional samples from FaceForensics++ dataset.
    Note: This requires the dataset to be available or accessible.
    """
    print("Note: FaceForensics++ dataset download requires:")
    print("1. Dataset access from https://github.com/ondyari/FaceForensics")
    print("2. Manual download or using their download scripts")
    print("3. This function is a placeholder for the download process")
    
    # This is a placeholder - actual implementation would depend on
    # how the user has access to the FaceForensics++ dataset
    print(f"\nTo download {num_samples_per_class} samples per class:")
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Use their download scripts or manually download")
    print("3. Place files in the appropriate directories")

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Setup
    dataset_path, output_path = setup_paths()
    
    # Option 1: Augment existing data
    print("=== Option 1: Augmenting Existing Dataset ===")
    original_count, manipulated_count = expand_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        target_per_class=2500,  # 2500 per class = 5000 total
        augment_per_video=2
    )
    
    print(f"\nAugmentation completed!")
    print(f"Use the augmented dataset at: {output_path}")
    print(f"Update your colab_deepfake_training.py to use this path")
    
    # Option 2: Download more data (placeholder)
    print("\n=== Option 2: Download Additional Data ===")
    print("If you want to download more real samples instead of augmenting:")
    download_faceforensics_samples(output_path, num_samples_per_class=500)

if __name__ == "__main__":
    main()
