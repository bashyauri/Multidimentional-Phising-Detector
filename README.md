# A Machine Learning-Based Multimodal Approach for Detection and Prevention of Emerging Phishing Attacks

This repository contains an academic Flask prototype that trains and deploys multimodal phishing detectors for URL, Email, SMS, QR, Deepfake, and Voice deepfake media streams. It includes model training scripts, a weighted fusion layer, SQLite logging, and a dashboard with performance and operational analytics.

## Technology Stack
- Flask
- scikit-learn
- XGBoost
- pandas / numpy
- NLTK (text preprocessing)
- OpenCV / pyzbar (QR decoding)
- PyTorch (deepfake detection)
- SQLite (via Flask-SQLAlchemy)
- Chart.js + Bootstrap

## Project Structure

```text
project/
|
|-- app.py
|-- models/
|   |-- url_model.pkl
|   |-- email_model.pkl
|   |-- sms_model.pkl
|   |-- qr_model.pkl
|   |-- deepfake_efficientnet_b0.pt
|   |-- voice_model.pkl
|   |-- metrics_summary.json
|
|-- ml_training/
|   |-- train_url.py
|   |-- train_email.py
|   |-- train_sms.py
|   |-- train_qr.py
|   |-- train_deepfake_efficientnet.py
|   |-- train_voice_deepfake_balanced.py
|   |-- common.py
|
|-- routes/
|   |-- detection.py
|
|-- templates/
|   |-- index.html
|
|-- static/
|   |-- css/
|   |   |-- style.css
|   |-- js/
|   |   |-- app.js
|
|-- utils/
|   |-- fusion.py
|   |-- model_loader.py
|   |-- qr_features.py
|
|-- datasets/
|   |-- url_phishing.csv
|   |-- email_phishing.csv
|   |-- sms_spam.csv
|   |-- qr/
|   |-- faceforensics/
|   |-- voice/
|
|-- database/
|-- MODEL_DECISIONS_AND_RATIONALE.md
```

## System Requirements

- **Python**: 3.9 or higher (3.11+ recommended for best compatibility)
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 8GB (16GB recommended for deepfake training)
- **Storage**: Minimum 10GB free space (50GB+ for deepfake datasets)
- **GPU**: Optional (NVIDIA GPU with CUDA for faster deepfake training)

## Installation Guide

### Windows Installation (Recommended - One-Click Setup)

**For Non-Technical Users - Simplest Method:**

1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/)
   - During installation, **CHECK "Add Python to PATH"** (this is critical!)
   - Select "Install for all users" (optional but recommended)

2. **Download/clone** this repository to a **short path** on your computer
   - **IMPORTANT**: Windows has a 260 character path limit
   - **Recommended paths**: `C:\Phising` or `C:\Projects\Phising`
   - **Avoid**: OneDrive, Desktop with long names, deeply nested folders
   - Example of problematic path: `C:\Users\Name\OneDrive - Company\Desktop\LongProjectName\`
   - If you get "filename too long" error, move to a shorter path

3. **(Optional) Download Datasets** - For model training:
   - Double-click `download_datasets.bat` to automatically download required datasets
   - Some datasets may require manual download from Kaggle (script provides instructions)
   - Datasets are placed in the `datasets/` folder

4. **Double-click `launch_client.bat`**
   - **First run only**: Automatically creates virtual environment and installs all dependencies
   - **Subsequent runs**: Starts the application immediately
   - This file handles everything automatically - no manual commands needed

5. **Your browser opens automatically** at: http://127.0.0.1:5000

**Optional Enhancements:**
- **Desktop Shortcut**: Double-click `create_desktop_shortcut.bat` to add a launcher to your desktop
- **Model Training**: After adding datasets, double-click `train_models.bat` to train all models

**That's it!** The batch file handles all technical setup automatically.

#### Option 2: Manual Installation (Technical Users)

1. **Install Python 3.9+**
   ```bash
   # Download from https://www.python.org/downloads/
   # Ensure "Add Python to PATH" is checked during installation
   python --version  # Verify installation
   ```

2. **Clone or Download Repository**
   ```bash
   git clone https://github.com/bashyauri/Multidimentional-Phising-Detector.git
   cd Multidimentional-Phising-Detector
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify Installation**
   ```bash
   python -c "import flask; import sklearn; import torch; print('Dependencies OK')"
   ```

### Linux/macOS Installation

1. **Install Python 3.9+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.9 python3.9-venv python3-pip

   # macOS (using Homebrew)
   brew install python@3.9
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/bashyauri/Multidimentional-Phising-Detector.git
   cd Multidimentional-Phising-Detector
   ```

3. **Create Virtual Environment**
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify Installation**
   ```bash
   python -c "import flask; import sklearn; import torch; print('Dependencies OK')"
   ```

## Dataset Setup

### Required Datasets

The application requires datasets for training models. Place datasets in the `datasets/` directory:

**CSV Datasets**:
- `datasets/url_phishing.csv` - URL phishing dataset
- `datasets/email_phishing.csv` - Email phishing dataset  
- `datasets/sms_spam.csv` - SMS spam dataset

**Image/Video Datasets**:
- `datasets/qr/benign_qr_images_500/` - Benign QR code images
- `datasets/qr/malicious_qr_images_500/` - Malicious QR code images
- `datasets/faceforensics/original/` - Real face videos/images
- `datasets/faceforensics/manipulated/` - Deepfake videos/images
- `datasets/voice/DATASET-balanced.csv` - Voice deepfake dataset

### Dataset Sources

**URL Phishing**:
- PhiUSIIL Phishing URL Dataset: https://www.kaggle.com/datasets/anshulmehta1603/phiusiil-phishing-url-dataset
- Kaggle Phishing Website Dataset: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset

**Email Phishing**:
- Phishing Email Dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- Enron Email Dataset: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

**SMS Spam**:
- UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**QR Codes**:
- Generate using `ml_training/generate_qr_from_phiusiil.py` script

**Deepfake**:
- FaceForensics++: https://github.com/ondyari/FaceForensics
- Run `download_faceforensics_subset.bat` to download a subset

**Voice Deepfake**:
- Various voice deepfake datasets available on Kaggle

### Dataset Preparation

1. **Download datasets** from the sources above
2. **Rename files** to match expected names:
   - URL dataset → `url_phishing.csv`
   - Email dataset → `email_phishing.csv`
   - SMS dataset → `sms_spam.csv`
3. **Place files** in `datasets/` directory
4. **Validate datasets**:
   ```bash
   python validate_datasets.py
   # Or double-click validate_datasets.bat on Windows
   ```

### Dataset Column Requirements

**URL Dataset** (`url_phishing.csv`):
- Required columns: URL features (63 columns), label column
- Label values: 0 (legitimate), 1 (phishing)

**Email Dataset** (`email_phishing.csv`):
- Required columns: email text, label
- Label values: 0 (legitimate), 1 (phishing)

**SMS Dataset** (`sms_spam.csv`):
- Required columns: SMS text, label
- Label values: ham (legitimate), spam (phishing)

## Model Training

### Quick Training (Windows)

Double-click `train_models.bat` to train all models sequentially.

### Evaluation Methodology

For detailed information on dataset splitting, validation datasets, and evaluation methodology, see `EVALUATION_DATA_GUIDE.md`. This guide covers:

- Three-way dataset split (Training/Validation/Testing)
- Recommended split ratios based on dataset size
- Validation dataset requirements and storage
- Why validation datasets are required for robust ML methodology
- Sample requirements per module for demonstration and defence

### Manual Training

**Train URL Model**:
```bash
python -m ml_training.train_url --dataset datasets/url_phishing.csv
```

**Train Email Model**:
```bash
python -m ml_training.train_email --dataset datasets/email_phishing.csv
```

**Train SMS Model**:
```bash
python -m ml_training.train_sms --dataset datasets/sms_spam.csv
```

**Train QR Model**:
```bash
python -m ml_training.train_qr --dataset-dir datasets/qr
```

**Train Deepfake Model**:
```bash
python -m ml_training.train_deepfake --dataset-dir datasets/faceforensics
```

**Train Voice Model**:
```bash
python -m ml_training.train_voice_deepfake_balanced
```

### Training Artifacts

After training, the following files are created in `models/`:
- `url_model.pkl` - XGBoost URL classifier
- `email_model.pkl` - Logistic Regression email classifier
- `sms_model.pkl` - Logistic Regression SMS classifier
- `qr_model.pkl` - XGBoost QR classifier
- `deepfake_efficientnet_b0.pt` - EfficientNet deepfake detector
- `voice_model.pkl` - Random Forest voice classifier
- `metrics_summary.json` - Consolidated model metrics
- Confusion matrix plots in `static/plots/`

### Google Colab Training

For deepfake training with GPU acceleration, use the provided Colab notebooks:
- `colab_deepfake_training.ipynb` - Deepfake training with ConvNeXt-Tiny
- `colab_efficientnet_training.ipynb` - Deepfake training with EfficientNet-B0
- `colab_deepfake_data_augmentation.py` - Dataset augmentation script

Download the trained `.pt` and `.json` files from Colab and place them in `models/`.

## Running the Application

### Windows (Recommended)

**Option 1: One-Click Launch**
```bash
# Double-click run_client.bat
```

**Option 2: Manual Launch**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run application
python app.py
```

### Linux/macOS

```bash
# Activate virtual environment
source .venv/bin/activate

# Run application
python app.py
```

### Access the Application

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

The application will be available at this local address.

### Configuration Options

**Environment Variables**:
```bash
# Model priority flags
set SMS_PREFER_TRANSFORMER=false
set EMAIL_PREFER_TRANSFORMER=true
python app.py
```

**Port Configuration**:
```bash
# Run on different port
python app.py --port 8080
```

## Application Features

### Detection Modalities

1. **URL Phishing Detection**
   - XGBoost classifier with 63 URL features
   - Heuristic boosts for suspicious patterns
   - URL redirection handling

2. **Email Phishing Detection**
   - TF-IDF + Logistic Regression
   - Text preprocessing and feature extraction
   - Class imbalance handling

3. **SMS Phishing Detection**
   - TF-IDF + Logistic Regression (balanced)
   - Short text optimization
   - Class weighting for imbalance

4. **QR Code Phishing Detection**
   - Multimodal fusion (visual + URL content)
   - QR decoding and URL extraction
   - Weighted fusion (URL: 0.7, QR: 0.3)

5. **Deepfake Detection**
   - EfficientNet-B0 CNN with transfer learning
   - Frame-level video processing
   - Optimized threshold (0.44)

6. **Voice Deepfake Detection**
   - Random Forest with hand-crafted audio features
   - MFCC and spectral feature extraction
   - Balanced subsampling

### Multimodal Fusion

- **Weighted Probability Fusion**: Combines multiple modalities
- **Fusion Weights**: URL (0.35), Email (0.25), SMS (0.20), QR (0.15), Deepfake (0.05)
- **Decision Threshold**: 0.5
- **Error Correction**: Fusion can correct individual model errors

### Dashboard Analytics

- **Model Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Confusion Matrix**: TP, TN, FP, FN components
- **Operational Analytics**: Label distribution, detection trends, response times
- **Live Predictions**: Real-time prediction logging and display

### Research-Focused API Responses

All API endpoints include comprehensive debug information:
- Response time metrics
- Model parameters and configuration
- Decision thresholds
- Feature extraction details
- Fusion weights and individual probabilities

## API Endpoints

### Detection Endpoints

- `POST /api/detect/url` - URL phishing detection
- `POST /api/detect/email` - Email phishing detection
- `POST /api/detect/sms` - SMS phishing detection
- `POST /api/detect/qr` - QR code phishing detection
- `POST /api/detect/deepfake` - Deepfake detection
- `POST /api/detect/voice` - Voice deepfake detection
- `POST /api/detect/fusion` - Multimodal fusion detection

### Management Endpoints

- `POST /api/reload-models` - Reload all models and metrics
- `GET /api/dashboard-data` - Get dashboard analytics data

## Troubleshooting

### Common Issues

**Python not found**:
- Ensure Python 3.9+ is installed
- Verify "Add Python to PATH" was checked during installation

**"No module named 'xgboost'" or other missing dependencies**:
- Delete the `.venv` folder if it exists
- Double-click `launch_client.bat` again (it will reinstall all dependencies)
- This happens if requirements.txt was updated after initial installation

**Dependencies installation fails**:
- Ensure you have internet connection
- Try running as Administrator
- Check that Python is properly installed: `python --version`

**Module not found errors**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

**Model not loading**:
- Check model files exist in `models/` directory
- Verify file naming conventions
- Check file permissions

**Dataset errors**:
- Validate datasets with `validate_datasets.py`
- Check CSV column names match requirements
- Ensure label values are correct

**Port already in use**:
- Change port: `python app.py --port 8080`
- Kill process using port 5000

**GPU not detected (deepfake)**:
- Install CUDA-compatible PyTorch
- Update GPU drivers
- Training will fall back to CPU (slower)

### Getting Help

1. Check `MODEL_DECISIONS_AND_RATIONALE.md` for detailed model explanations
2. Review error messages in terminal/console
3. Verify all dependencies are installed correctly
4. Ensure datasets are properly formatted

## Advanced Configuration

### Model Priority Flags

Control which models are used for text classification:

```bash
# SMS: Use Transformer first (default: Logistic Regression)
set SMS_PREFER_TRANSFORMER=true

# Email: Use Logistic Regression first (default: Transformer)
set EMAIL_PREFER_TRANSFORMER=false
```

### Custom Model Integration

To add custom models:

1. Place model file in `models/` with correct format
2. Add loader in `utils/model_loader.py`
3. Implement `predict_probability()` method
4. Update `metrics_summary.json` with model metrics
5. Call `/api/reload-models` or restart app

### Database Management

The application uses SQLite for logging predictions:

- **Database Location**: `database/research_app.db`
- **Backup**: Copy `.db` file for backup
- **Reset**: Delete `.db` file to clear logs
- **Export**: Use SQLite tools to export data

## Performance Optimization

### For Better Performance

- Use GPU for deepfake training (if available)
- Reduce dataset size for faster training
- Use smaller batch sizes for memory efficiency
- Enable model caching in production

### For Research Purposes

- Use full datasets for accurate metrics
- Train with multiple random seeds
- Cross-validate model performance
- Log all hyperparameters and results

## Security Considerations

- **API Security**: Add authentication for production deployment
- **Input Validation**: All inputs are validated server-side
- **File Uploads**: File size and type restrictions enforced
- **Database**: SQLite is for development only; use PostgreSQL for production
- **HTTPS**: Use HTTPS in production environments

## Citation and Acknowledgments

If you use this system for research, please cite:

```
A Machine Learning-Based Multimodal Approach for Detection and Prevention 
of Emerging Phishing Attacks
```

**Acknowledgments**:
- Datasets from Kaggle, UCI, and FaceForensics++
- ML frameworks: scikit-learn, XGBoost, PyTorch
- Web framework: Flask
- Visualization: Chart.js

## License

This is an academic research prototype. Please ensure compliance with dataset licenses when using this system.

## Support and Documentation

- **Detailed Model Rationale**: `MODEL_DECISIONS_AND_RATIONALE.md`
- **Dataset Requirements**: `datasets/README.md`
- **Training Scripts**: `ml_training/` directory
- **API Documentation**: See inline code documentation

## Version History

- **v2.0**: Added light mode UI, comprehensive debug information, voice detection
- **v1.0**: Initial release with URL, email, SMS, QR, and deepfake detection
