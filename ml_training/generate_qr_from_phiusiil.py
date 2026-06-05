import argparse
import random
from pathlib import Path

import pandas as pd
import qrcode
from PIL import Image, ImageEnhance, ImageFilter


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "qr" / "phiusiil_qr_generated"


def apply_augmentation(img: Image.Image, augment: bool = True) -> Image.Image:
    """Apply data augmentation to QR code image."""
    if not augment:
        return img
    
    # Ensure image is in RGB mode for consistent processing
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Random rotation (-5 to 5 degrees) - reduced range
    if random.random() > 0.6:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=False, fillcolor='white')
    
    # Random brightness adjustment
    if random.random() > 0.6:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.9, 1.1)
        img = enhancer.enhance(factor)
    
    # Random contrast adjustment
    if random.random() > 0.6:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.9, 1.1)
        img = enhancer.enhance(factor)
    
    # Removed blur and noise - they cause decoding issues
    
    return img


def generate_qr_code(url: str, output_path: Path, size: int = 500, augment: bool = True, error_correction: str = 'L') -> bool:
    """Generate a QR code from a URL and save it as an image."""
    try:
        # Map error correction levels
        ec_levels = {
            'L': qrcode.constants.ERROR_CORRECT_L,
            'M': qrcode.constants.ERROR_CORRECT_M,
            'Q': qrcode.constants.ERROR_CORRECT_Q,
            'H': qrcode.constants.ERROR_CORRECT_H
        }
        
        # Generate QR code with auto version detection
        qr = qrcode.QRCode(
            version=None,  # Auto-detect version
            error_correction=ec_levels.get(error_correction, qrcode.constants.ERROR_CORRECT_L),
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Check if version is too large
        if qr.version > 40:
            return False  # Skip URLs that require version > 40
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Apply augmentation if enabled
        if augment:
            img = apply_augmentation(img, augment=True)
        
        # Resize to desired size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save the image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return True
        
    except Exception as e:
        # Silently skip problematic URLs
        return False


def generate_qr_dataset(
    csv_path: Path,
    output_dir: Path,
    max_samples: int = 5000,
    augment: bool = True,
    error_correction: str = 'L'
):
    """Generate QR codes from PhiUSIIL dataset."""
    df = pd.read_csv(csv_path)
    
    # Filter by label
    benign_df = df[df['label'] == 0]
    malicious_df = df[df['label'] == 1]
    
    # Create output directories
    benign_output = output_dir / "benign_qr_images_500"
    malicious_output = output_dir / "malicious_qr_images_500"
    benign_output.mkdir(parents=True, exist_ok=True)
    malicious_output.mkdir(parents=True, exist_ok=True)
    
    # Generate benign QR codes
    print(f"Generating benign QR codes...")
    benign_count = 0
    for idx, row in benign_df.head(max_samples).iterrows():
        url = row.get('URL', row.get('url', ''))
        if not url or pd.isna(url):
            continue
        
        output_path = benign_output / f"benign_{idx}.png"
        # Randomly select error correction level for diversity
        ec = random.choice(['L', 'M', 'Q', 'H']) if error_correction == 'random' else error_correction
        if generate_qr_code(url, output_path, augment=augment, error_correction=ec):
            benign_count += 1
    
    print(f"Benign: {benign_count} images")
    
    # Generate malicious QR codes
    print(f"Generating malicious QR codes...")
    malicious_count = 0
    for idx, row in malicious_df.head(max_samples).iterrows():
        url = row.get('URL', row.get('url', ''))
        if not url or pd.isna(url):
            continue
        
        output_path = malicious_output / f"malicious_{idx}.png"
        # Randomly select error correction level for diversity
        ec = random.choice(['L', 'M', 'Q', 'H']) if error_correction == 'random' else error_correction
        if generate_qr_code(url, output_path, augment=augment, error_correction=ec):
            malicious_count += 1
    
    print(f"Malicious: {malicious_count} images")
    print(f"Total: {benign_count + malicious_count} images")


def main():
    parser = argparse.ArgumentParser(description="Generate QR codes from PhiUSIIL dataset")
    parser.add_argument("--dataset", type=str, default=str(DATASETS_DIR / "qr" / "PhiUSIIL_Phishing_URL_Dataset.csv"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--error-correction", type=str, default='L', choices=['L', 'M', 'Q', 'H', 'random'])
    args = parser.parse_args()
    
    generate_qr_dataset(
        csv_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        augment=not args.no_augment,
        error_correction=args.error_correction
    )


if __name__ == "__main__":
    main()
