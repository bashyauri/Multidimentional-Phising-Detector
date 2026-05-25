# A Machine Learning-Based Multimodal Approach for Detection and Prevention of Emerging Phishing Attacks

This repository contains an academic Flask prototype that trains and deploys multimodal phishing detectors for URL, Email, SMS, QR, and Deepfake media streams. It includes model training scripts, a weighted fusion layer, SQLite logging, and a dashboard with performance and operational analytics.

## Technology Stack
- Flask
- scikit-learn
- pandas / numpy
- NLTK (text preprocessing)
- OpenCV / pyzbar (QR decoding)
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
|   |-- deepfake_model.pkl
|   |-- metrics_summary.json
|
|-- ml_training/
|   |-- train_url.py
|   |-- train_email.py
|   |-- train_sms.py
|   |-- train_deepfake.py
|   |-- common.py
|
|-- routes/
|   |-- detection.py
|
|-- templates/
|-- static/
|-- utils/
|-- datasets/
|-- database/
```

## Client Laptop Quick Start (Windows)

1. Install Python 3.13+ from python.org and ensure "Add Python to PATH" is enabled.
2. Put this project folder on the laptop.
3. Double-click `launch_client.bat`.
4. On first run, setup is automatic; later runs start the app directly.
5. (Optional) double-click `create_desktop_shortcut.bat` to add a desktop launcher.
6. (Optional for model training) add datasets in `datasets/` as described in `datasets/README.md`.

### One-Click Files for Non-Technical Users

- `launch_client.bat`: one command flow (setup if needed + run app)
- `create_desktop_shortcut.bat`: creates "Phishing Research App" shortcut on Desktop
- `setup_client.bat`: manual first-time setup only
- `run_client.bat`: manual run only
- `validate_datasets.bat`: validates dataset files and required columns
- `train_models.bat`: model training workflow

## Setup (Manual)

1. Create and activate virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add datasets to `datasets/` (see `datasets/README.md`).

## Train Models

Fast option on Windows: double-click `train_models.bat`.

Recommended before training: double-click `validate_datasets.bat`.

Run all scripts from the project root.

```bash
python -m ml_training.train_url --dataset datasets/url_phishing.csv
python -m ml_training.train_qr --dataset-dir datasets/qr
python -m ml_training.train_email --dataset datasets/email_phishing.csv
python -m ml_training.train_sms --dataset datasets/sms_spam.csv
python -m ml_training.train_deepfake --dataset-dir datasets/faceforensics
```

Artifacts produced:
- `models/url_model.pkl`
- `models/qr_model.pkl`
- `models/email_model.pkl`
- `models/sms_model.pkl`
- `models/deepfake_model.pkl`
- `models/metrics_summary.json`
- confusion matrix images in `static/plots/`

## QR Code and Deepfake Modules

### QR Code Detection
- The app can train a QR-specific classifier from image datasets in `datasets/qr`.
- At inference time, uploaded QR images are decoded and scored with URL risk signals.
- If `models/qr_model.pkl` is available, the API fuses URL risk with QR-model risk.
- Supported in UI via the QR upload form and API endpoint `/api/detect/qr`.

### Deepfake Detection
- Put FaceForensics++ real/original samples in `datasets/faceforensics/original/`.
- Put fake/manipulated samples in `datasets/faceforensics/manipulated/`.
- Train with `train_deepfake.bat` or:

```bash
python -m ml_training.train_deepfake --dataset-dir datasets/faceforensics
```

- The web app uses `models/deepfake_model.pkl` automatically when it exists.
- If no trained deepfake model exists, the endpoint falls back to the old simulated score so the demo remains usable.
- Supported in UI via media upload and API endpoint `/api/detect/deepfake`.
- Fusion endpoint can also accept manual deepfake probability input.

## How to Get Datasets

Place CSV files in `datasets/` using the expected names:
- `datasets/url_phishing.csv`
- `datasets/email_phishing.csv`
- `datasets/sms_spam.csv`
- `datasets/faceforensics/original/`
- `datasets/faceforensics/manipulated/`

Recommended sources:
- URL phishing:
  - Kaggle Phishing Website Dataset: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
  - UCI Phishing Websites: https://archive.ics.uci.edu/dataset/327/phishing+websites
- Email phishing:
  - Enron Email Dataset (Kaggle): https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
  - Phishing Email Dataset (Kaggle): https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- SMS phishing/spam:
  - UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Deepfake:
  - FaceForensics++: https://github.com/ondyari/FaceForensics

Practical workflow:
1. Download dataset CSV files from Kaggle/UCI.
2. Create/prepare 3 CSV files and rename them exactly as:
  - `url_phishing.csv`
  - `email_phishing.csv`
  - `sms_spam.csv`
3. Ensure each CSV has a text/URL column and a label column.
4. Copy all 3 files into `datasets/`.
5. For deepfake training, place FaceForensics++ real samples in `original/` and fake samples in `manipulated/`.
   You can also run `download_faceforensics_subset.bat` to download a small `c23` subset directly into those folders.
6. Run `validate_datasets.bat`.
7. Run `train_models.bat`, `train_deepfake.bat`, or manual training commands.

Notes:
- Kaggle downloads require a Kaggle account login.
- If your source dataset is not CSV, convert it to CSV before training.

Detailed column requirements are documented in `datasets/README.md`.

## Run Web App

Fast option on Windows: double-click `run_client.bat`.

```bash
python app.py
```

Open http://127.0.0.1:5000

### Model Priority Flags

You can switch text-model priority at runtime with environment variables:

```bash
set SMS_PREFER_TRANSFORMER=true
set EMAIL_PREFER_TRANSFORMER=false
python app.py
```

- `SMS_PREFER_TRANSFORMER=false` by default, so SMS uses Logistic Regression first and falls back to DistilBERT.
- `EMAIL_PREFER_TRANSFORMER=true` by default, so email uses DistilBERT first and falls back to Logistic Regression.
- Accepted true values are `1`, `true`, `yes`, and `on`.

## Features Implemented

- URL phishing detection with handcrafted URL features + RandomForest
- Email phishing detection with TF-IDF + Logistic Regression
- SMS phishing detection with TF-IDF + Naive Bayes
- QR decoding and URL model reuse
- Trainable FaceForensics-style deepfake media detector with simulation fallback
- Multimodal weighted fusion decision layer
- SQLite logging for predictions and metadata
- Dashboard charts:
  - confusion matrix components (TP/TN/FP/FN)
  - model accuracy comparison
  - precision/recall/F1 comparison
  - ROC-AUC comparison
  - phishing vs legitimate distribution
  - detection trend over time
  - response time analysis

## Notes

- This is an academic prototype for research demonstration and grading.
- Use real datasets and retrain models before evaluation.
- If models are retrained while app is running, call `POST /api/reload-models`.
- For reproducible installs, use `requirements-lock.txt` instead of `requirements.txt`.

## Model Integration and Updating (All Modalities)

### Model File Formats
- **Deepfake (PyTorch):** `.pt` (e.g., `deepfake_efficientnet_b0.pt`)
- **URL/QR/Email/SMS (scikit-learn/XGBoost):** `.pkl`

### Adding or Updating Models
1. Place new or updated model files in the `models/` folder using the correct format and naming convention.
   - Example: `models/deepfake_efficientnet_b0.pt`, `models/url_model.pkl`, etc.
2. Place updated metrics files (e.g., `deepfake_efficientnet_b0_metrics.json`) in `models/`.
3. (Optional) Merge or update `metrics_summary.json` for dashboard analytics.
4. Regenerate confusion matrix and other plots using the provided scripts (see `update_efficientnet_confusion_matrix.py` for an example).
5. Use the `/api/reload-models` endpoint or restart the app to reload all models and metrics.

### Training Deepfake Models in Colab
- You can train deepfake models in Google Colab and download the `.pt` and metrics `.json` files.
- Copy these files into the `models/` folder as above.
- The app will use them automatically after reload.

### Consistent Client Experience
- All detection types (URL, Email, SMS, QR, Deepfake) are available in the web UI and API.
- The backend automatically selects the correct model for each detection type.
- All metrics and plots are updated and displayed together for a unified experience.

### Adding New Model Types (e.g., Flax/JAX)
- Add a loader in `utils/model_loader.py` for the new format.
- Expose a `predict_probability()` method for the new model.
- Add a form and endpoint if you want a new UI section.

### Troubleshooting
- If a model is not detected, check file placement and naming in `models/`.
- If metrics or plots are outdated, rerun the update scripts and reload models.
