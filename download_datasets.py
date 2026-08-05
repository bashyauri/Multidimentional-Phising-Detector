#!/usr/bin/env python3
"""
Dataset Download Script for Phishing Detection Research App
Automatically downloads all required datasets for model training and evaluation.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import subprocess

# Configuration
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

def print_step(step_num, total, message):
    """Print formatted step information"""
    print(f"\n[{step_num}/{total}] {message}")
    print("=" * 60)

def download_file(url, destination, description):
    """Download a file from URL with progress indication"""
    print(f"Downloading {description}...")
    print(f"From: {url}")
    print(f"To: {destination}")
    
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Successfully extracted")
        return True
    except Exception as e:
        print(f"✗ Failed to extract: {e}")
        return False

def download_uci_sms():
    """Download SMS Spam Collection from UCI"""
    print_step(1, 6, "Downloading SMS Spam Collection Dataset")
    
    sms_dir = DATASETS_DIR / "sms"
    sms_dir.mkdir(exist_ok=True)
    
    # UCI SMS Spam Collection direct download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = sms_dir / "smsspamcollection.zip"
    
    if download_file(url, zip_path, "SMS Spam Collection"):
        if extract_zip(zip_path, sms_dir):
            # Rename the extracted file to expected name
            extracted_file = sms_dir / "SMSSpamCollection"
            target_file = sms_dir / "sms_spam.csv"
            if extracted_file.exists():
                extracted_file.rename(target_file)
                print(f"✓ Renamed to sms_spam.csv")
                return True
    
    return False

def download_phiusiil_url():
    """Download PhiUSIIL Phishing URL Dataset from Kaggle"""
    print_step(2, 6, "Downloading PhiUSIIL Phishing URL Dataset")
    
    print("This dataset requires Kaggle API credentials.")
    print("To download manually:")
    print("1. Go to: https://www.kaggle.com/datasets/anshulmehta1603/phiusiil-phishing-url-dataset")
    print("2. Download the dataset")
    print("3. Extract to: datasets/url_phishing.csv")
    print("\nOr set up Kaggle API credentials:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Get API key from: https://www.kaggle.com/settings")
    print("3. Place key in: ~/.kaggle/kaggle.json")
    
    # Try to use Kaggle API if available
    try:
        import kaggle
        print("\nAttempting Kaggle API download...")
        kaggle.api.dataset_download_files(
            'anshulmehta1603/phiusiil-phishing-url-dataset',
            path=str(DATASETS_DIR),
            unzip=True
        )
        print("✓ Successfully downloaded via Kaggle API")
        
        # Find and rename the CSV file
        for file in DATASETS_DIR.glob("*.csv"):
            if "phishing" in file.name.lower() or "phiusiil" in file.name.lower():
                file.rename(DATASETS_DIR / "url_phishing.csv")
                print(f"✓ Renamed to url_phishing.csv")
                return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
    
    return False

def download_email_dataset():
    """Download Email Phishing Dataset from Kaggle"""
    print_step(3, 6, "Downloading Email Phishing Dataset")
    
    print("This dataset requires Kaggle API credentials.")
    print("To download manually:")
    print("1. Go to: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset")
    print("2. Download the dataset")
    print("3. Extract to: datasets/email_phishing.csv")
    
    # Try to use Kaggle API if available
    try:
        import kaggle
        print("\nAttempting Kaggle API download...")
        kaggle.api.dataset_download_files(
            'naserabdullahalam/phishing-email-dataset',
            path=str(DATASETS_DIR),
            unzip=True
        )
        print("✓ Successfully downloaded via Kaggle API")
        
        # Find and rename the CSV file
        for file in DATASETS_DIR.glob("*.csv"):
            if "email" in file.name.lower() and "phishing" in file.name.lower():
                file.rename(DATASETS_DIR / "email_phishing.csv")
                print(f"✓ Renamed to email_phishing.csv")
                return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
    
    return False

def download_voice_dataset():
    """Download Voice Deepfake Dataset"""
    print_step(4, 6, "Downloading Voice Deepfake Dataset")
    
    voice_dir = DATASETS_DIR / "voice"
    voice_dir.mkdir(exist_ok=True)
    
    print("Voice deepfake datasets are available on Kaggle.")
    print("To download manually:")
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Search for 'voice deepfake' or 'audio deepfake'")
    print("3. Download a balanced dataset")
    print("4. Place CSV file at: datasets/voice/DATASET-balanced.csv")
    
    # Try to use Kaggle API if available
    try:
        import kaggle
        print("\nAttempting Kaggle API download...")
        # Note: This is a placeholder - actual dataset name may vary
        kaggle.api.dataset_download_files(
            'kingabzpro/deep-voice-deepfake-detection',
            path=str(voice_dir),
            unzip=True
        )
        print("✓ Successfully downloaded via Kaggle API")
        return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
    
    return False

def setup_qr_dataset():
    """Setup QR dataset (generate from URL dataset)"""
    print_step(5, 6, "Setting Up QR Code Dataset")
    
    qr_dir = DATASETS_DIR / "qr"
    qr_dir.mkdir(exist_ok=True)
    
    print("QR codes are generated from the URL dataset.")
    print("To generate QR codes:")
    print("1. Ensure url_phishing.csv is available")
    print("2. Run: python -m ml_training.generate_qr_from_phiusiil")
    print("3. This will create benign_qr_images_500 and malicious_qr_images_500 folders")
    
    # Try to generate if URL dataset exists
    url_dataset = DATASETS_DIR / "url_phishing.csv"
    if url_dataset.exists():
        try:
            print("\nAttempting to generate QR codes...")
            result = subprocess.run(
                [sys.executable, "-m", "ml_training.generate_qr_from_phiusiil"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ Successfully generated QR codes")
                return True
            else:
                print(f"QR generation failed: {result.stderr}")
        except Exception as e:
            print(f"QR generation error: {e}")
    
    return False

def setup_deepfake_dataset():
    """Setup Deepfake Dataset"""
    print_step(6, 6, "Setting Up Deepfake Dataset")
    
    faceforensics_dir = DATASETS_DIR / "faceforensics"
    faceforensics_dir.mkdir(exist_ok=True)
    
    print("Deepfake dataset from FaceForensics++")
    print("To download:")
    print("1. Go to: https://github.com/ondyari/FaceForensics")
    print("2. Follow download instructions")
    print("3. Or run: download_faceforensics_subset.bat (if available)")
    print("4. Place videos in datasets/faceforensics/original/ and manipulated/")
    
    # Check if download script exists
    download_script = Path("download_faceforensics_subset.bat")
    if download_script.exists():
        print(f"\nFound download script: {download_script}")
        print("Run this script manually to download FaceForensics++ subset")
    
    return False

def create_placeholder_datasets():
    """Create placeholder files for datasets that couldn't be downloaded"""
    print("\n" + "=" * 60)
    print("Creating placeholder files for missing datasets...")
    
    placeholder_info = {
        "url_phishing.csv": "Download from Kaggle: https://www.kaggle.com/datasets/anshulmehta1603/phiusiil-phishing-url-dataset",
        "email_phishing.csv": "Download from Kaggle: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset",
        "sms_spam.csv": "Downloaded from UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    }
    
    for filename, info in placeholder_info.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(f"# PLACEHOLDER - {info}\n")
            print(f"✓ Created placeholder: {filename}")

def main():
    """Main download function"""
    print("=" * 60)
    print("DATASET DOWNLOAD SCRIPT")
    print("Phishing Detection Research App")
    print("=" * 60)
    
    results = {
        "SMS Spam Collection": download_uci_sms(),
        "PhiUSIIL URL Dataset": download_phiusiil_url(),
        "Email Phishing Dataset": download_email_dataset(),
        "Voice Deepfake Dataset": download_voice_dataset(),
        "QR Code Dataset": setup_qr_dataset(),
        "Deepfake Dataset": setup_deepfake_dataset(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED/MANUAL"
        print(f"{dataset}: {status}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Check datasets/ directory for downloaded files")
    print("2. Manually download any datasets that failed (see instructions above)")
    print("3. Place datasets in correct locations:")
    print("   - datasets/url_phishing.csv")
    print("   - datasets/email_phishing.csv")
    print("   - datasets/sms_spam.csv")
    print("   - datasets/voice/DATASET-balanced.csv")
    print("   - datasets/qr/benign_qr_images_500/")
    print("   - datasets/qr/malicious_qr_images_500/")
    print("   - datasets/faceforensics/original/")
    print("   - datasets/faceforensics/manipulated/")
    print("4. Run: python validate_datasets.py")
    print("5. Train models: Double-click train_models.bat")

if __name__ == "__main__":
    main()
