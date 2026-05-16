# A Machine Learning-Based Multimodal Approach for Detection and Prevention of Emerging Phishing Attacks

This repository contains an academic Flask prototype that trains and deploys multimodal phishing detectors for URL, Email, SMS, QR, and simulated Deepfake streams. It includes model training scripts, a weighted fusion layer, SQLite logging, and a dashboard with performance and operational analytics.

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
|   |-- metrics_summary.json
|
|-- ml_training/
|   |-- train_url.py
|   |-- train_email.py
|   |-- train_sms.py
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
python -m ml_training.train_email --dataset datasets/email_phishing.csv
python -m ml_training.train_sms --dataset datasets/sms_spam.csv
```

Artifacts produced:
- `models/url_model.pkl`
- `models/email_model.pkl`
- `models/sms_model.pkl`
- `models/metrics_summary.json`
- confusion matrix images in `static/plots/`

## QR Code and Deepfake Modules

### QR Code Detection
- No separate QR training dataset is required.
- The app decodes uploaded QR images and extracts embedded text/URL.
- If a URL is detected, it is classified using the trained URL model.
- Supported in UI via the QR upload form and API endpoint `/api/detect/qr`.

### Deepfake Detection (Academic Simplified)
- This prototype uses a simulated deepfake probability output for academic demonstration.
- No deepfake model training is required to run the current system.
- Supported in UI via media upload and API endpoint `/api/detect/deepfake`.
- Fusion endpoint can also accept manual deepfake probability input.

## How to Get Datasets

Place CSV files in `datasets/` using the expected names:
- `datasets/url_phishing.csv`
- `datasets/email_phishing.csv`
- `datasets/sms_spam.csv`

Recommended sources:
- URL phishing:
  - Kaggle Phishing Website Dataset: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
  - UCI Phishing Websites: https://archive.ics.uci.edu/dataset/327/phishing+websites
- Email phishing:
  - Enron Email Dataset (Kaggle): https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
  - Phishing Email Dataset (Kaggle): https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- SMS phishing/spam:
  - UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Practical workflow:
1. Download dataset CSV files from Kaggle/UCI.
2. Create/prepare 3 CSV files and rename them exactly as:
  - `url_phishing.csv`
  - `email_phishing.csv`
  - `sms_spam.csv`
3. Ensure each CSV has a text/URL column and a label column.
4. Copy all 3 files into `datasets/`.
5. Run `validate_datasets.bat`.
6. Run `train_models.bat` or manual training commands.

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

## Features Implemented

- URL phishing detection with handcrafted URL features + RandomForest
- Email phishing detection with TF-IDF + Logistic Regression
- SMS phishing detection with TF-IDF + Naive Bayes
- QR decoding and URL model reuse
- Simulated deepfake phishing probability module
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
