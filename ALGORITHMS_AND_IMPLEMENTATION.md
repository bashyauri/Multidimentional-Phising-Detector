# Algorithms and Implementation Details

This document provides detailed information about the machine learning algorithms used in the phishing detection system, including code snippets suitable for academic reference and implementation details.

## Table of Contents

1. [URL Phishing Detection](#url-phishing-detection)
2. [Email Phishing Detection](#email-phishing-detection)
3. [SMS Phishing Detection](#sms-phishing-detection)
4. [QR Code Phishing Detection](#qr-code-phishing-detection)
5. [Deepfake Detection](#deepfake-detection)
6. [Voice Deepfake Detection](#voice-deepfake-detection)
7. [Fusion Layer](#fusion-layer)

---

## URL Phishing Detection

### Algorithm: XGBoost Classifier with URL Feature Engineering

**Rationale:** XGBoost (Extreme Gradient Boosting) was selected for URL phishing detection due to its ability to handle structured features, handle missing values, and provide excellent performance on tabular data. The algorithm uses gradient boosting with decision trees, which is particularly effective for the 63 URL features extracted.

### Implementation Code

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train_url_model(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Extract URL features (63 features including:
    # - URL length, character counts
    # - Special character presence
    # - Domain information
    # - Subdomain count
    # - TLD analysis
    # - Path characteristics
    # - Query parameters
    # - Security indicators (HTTPS, etc.)
    feature_rows = df['url'].astype(str).apply(extract_url_features)
    x = pd.concat(feature_rows.to_list(), ignore_index=True)
    y = df['label'].apply(normalize_label)
    
    # Three-way dataset split (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # XGBoost Classifier configuration
    model = XGBClassifier(
        n_estimators=450,              # Number of boosting rounds
        max_depth=5,                   # Maximum tree depth
        learning_rate=0.05,            # Step size shrinkage
        subsample=0.9,                # Subsample ratio of training instances
        colsample_bytree=0.9,         # Subsample ratio of columns
        eval_metric="logloss",        # Evaluation metric
        random_state=42,               # Random seed for reproducibility
        n_jobs=2,                     # Parallel processing
        tree_method="hist",           # Histogram-based algorithm
    )
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predictions
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    # Save model
    joblib.dump(model, 'url_model.pkl')
    
    return model, y_pred, y_prob
```

### Key Parameters Explained

- **n_estimators=450**: Number of boosting rounds. Higher values improve performance but increase training time.
- **max_depth=5**: Controls tree complexity. Prevents overfitting by limiting tree depth.
- **learning_rate=0.05**: Step size shrinkage. Lower values require more trees but often yield better generalization.
- **subsample=0.9**: Fraction of samples used per tree. Adds regularization by sampling.
- **colsample_bytree=0.9**: Fraction of features used per tree. Prevents overfitting.
- **tree_method="hist"**: Histogram-based algorithm for faster training on large datasets.

### Feature Engineering

URL features are extracted using the following approach:

```python
def extract_url_features(url):
    """Extract 63 features from URL for phishing detection"""
    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_at_signs': url.count('@'),
        'num_equals': url.count('='),
        'num_question_marks': url.count('?'),
        'num_ampersands': url.count('&'),
        'has_https': 1 if url.startswith('https://') else 0,
        'has_ip_address': 1 if has_ip_pattern(url) else 0,
        # ... additional 50+ features
    }
    return pd.Series(features)
```

---

## Email Phishing Detection

### Algorithm: Logistic Regression with TF-IDF Vectorization

**Rationale:** Logistic Regression with TF-IDF (Term Frequency-Inverse Document Frequency) was chosen for email phishing detection due to its interpretability, efficiency with text data, and strong performance on classification tasks. TF-IDF captures the importance of words in the email content.

### Implementation Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_email_model(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Extract text and labels
    x = df['text'].astype(str).apply(clean_text)
    y = df['label'].apply(normalize_label)
    
    # Three-way dataset split (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,      # Maximum number of features
            ngram_range=(1, 2),      # Unigrams and bigrams
            stop_words='english',    # Remove common stop words
            lowercase=True           # Convert to lowercase
        )),
        ('clf', LogisticRegression(
            max_iter=2000,           # Maximum iterations for convergence
            random_state=42,         # Random seed
            C=1.0                    # Regularization strength
        ))
    ])
    
    # Train the model
    pipeline.fit(x_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    
    # Save model
    joblib.dump(pipeline, 'email_model.pkl')
    
    return pipeline, y_pred, y_prob
```

### Key Parameters Explained

- **max_features=10000**: Limits vocabulary size to most frequent 10,000 words, preventing overfitting.
- **ngram_range=(1, 2)**: Captures both individual words and word pairs for better context.
- **max_iter=2000**: Ensures convergence for the optimization algorithm.
- **C=1.0**: Inverse regularization strength. Lower values increase regularization.

### Text Preprocessing

```python
def clean_text(text):
    """Clean and preprocess email text"""
    import re
    import nltk
    from nltk.corpus import stopwords
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)
```

---

## SMS Phishing Detection

### Algorithm: Logistic Regression with TF-IDF (Balanced)

**Rationale:** Similar to email detection, Logistic Regression with TF-IDF is used for SMS phishing detection. The key difference is the use of class weighting to handle potential class imbalance in SMS datasets.

### Implementation Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def train_sms_model(dataset_path):
    # Load dataset (handle tab-separated UCI format)
    try:
        df = pd.read_csv(dataset_path)
    except:
        df = pd.read_csv(dataset_path, sep='\t', names=['label', 'text'], header=None)
    
    # Extract text and labels
    x = df['text'].astype(str).apply(clean_text)
    y = df['label'].apply(normalize_label)
    
    # Three-way dataset split (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Pipeline with TF-IDF and Logistic Regression (balanced)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,       # Slightly fewer features for shorter SMS text
            ngram_range=(1, 2),      # Unigrams and bigrams
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            max_iter=3000,           # Higher iterations for convergence
            class_weight='balanced', # Handle class imbalance
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(x_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    
    # Save model
    joblib.dump(pipeline, 'sms_model.pkl')
    
    return pipeline, y_pred, y_prob
```

### Key Differences from Email Model

- **max_features=8000**: Reduced vocabulary size due to shorter SMS messages.
- **class_weight='balanced'**: Automatically adjusts weights inversely proportional to class frequencies.
- **max_iter=3000**: Increased iterations to ensure convergence with balanced weights.

---

## QR Code Phishing Detection

### Algorithm: XGBoost with Multi-Modal Features

**Rationale:** QR code detection uses XGBoost with a combination of image features, decoded URL features, and structural QR code properties. This multi-modal approach captures both visual characteristics and the content of the QR code.

### Implementation Code

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def train_qr_model(dataset_dir):
    # Load QR code images and extract features
    benign_dir = dataset_dir / "benign_qr_images_500"
    malicious_dir = dataset_dir / "malicious_qr_images_500"
    
    rows = []
    labels = []
    
    # Extract features from images
    for class_dir, label in [(benign_dir, 0), (malicious_dir, 1)]:
        for image_path in class_dir.glob("*.png"):
            file_bytes = image_path.read_bytes()
            
            # Decode QR code text
            decoded_text = decode_qr_text_from_bytes(file_bytes)
            
            # Extract multi-modal features
            features = build_qr_feature_frame(
                file_bytes=file_bytes,
                url_model=url_model,      # Pre-trained URL model
                decoded_text=decoded_text
            )
            
            rows.append(features.iloc[0].to_dict())
            labels.append(label)
    
    x = pd.DataFrame(rows).fillna(0)
    y = np.array(labels, dtype=np.int64)
    
    # Three-way dataset split (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Calculate class weights for imbalance
    class_counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = float(class_counts[0] / max(class_counts[1], 1))
    
    # XGBoost Classifier
    model = XGBClassifier(
        n_estimators=1000,           # More trees for complex features
        learning_rate=0.03,           # Lower learning rate
        max_depth=4,                  # Shallower trees
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,               # L2 regularization
        gamma=0.0,                    # Minimum loss reduction
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )
    
    # Train with early stopping
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
        early_stopping_rounds=50      # Stop if validation loss doesn't improve
    )
    
    # Predictions
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    # Save model
    joblib.dump(model, 'qr_model.pkl')
    
    return model, y_pred, y_prob
```

### Multi-Modal Feature Extraction

```python
def build_qr_feature_frame(file_bytes, url_model, decoded_text):
    """Extract multi-modal features from QR code image"""
    import cv2
    from PIL import Image
    
    # Image features
    img = Image.open(io.BytesIO(file_bytes))
    img_array = np.array(img)
    
    features = {
        # Image statistics
        'image_mean': img_array.mean(),
        'image_std': img_array.std(),
        'image_size': img_array.size,
        
        # QR code structure
        'qr_version': extract_qr_version(img_array),
        'qr_error_correction': extract_error_level(img_array),
        
        # Decoded URL features (if decodable)
        'is_decodable': 1 if decoded_text else 0,
        'url_length': len(decoded_text) if decoded_text else 0,
    }
    
    # Add URL model predictions if URL is decodable
    if decoded_text and url_model:
        url_features = extract_url_features(decoded_text)
        url_prob = url_model.predict_proba([url_features])[0, 1]
        features['url_phishing_prob'] = url_prob
    
    return pd.Series(features)
```

### Key Parameters Explained

- **n_estimators=1000**: More trees for complex multi-modal features.
- **learning_rate=0.03**: Lower rate for better generalization with many trees.
- **max_depth=4**: Shallower trees to prevent overfitting on image features.
- **scale_pos_weight**: Handles class imbalance in QR datasets.
- **early_stopping_rounds=50**: Prevents overfitting by stopping when validation loss plateaus.

---

## Deepfake Detection

### Algorithm: EfficientNet-B0 with Transfer Learning

**Rationale:** EfficientNet-B0 is a convolutional neural network architecture that achieves state-of-the-art performance with fewer parameters. Transfer learning from ImageNet weights provides a strong foundation for deepfake detection.

### Implementation Code

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Modify final classifier layer
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_deepfake_model(train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector().to(device)
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Acc: {100*correct/total:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'deepfake_efficientnet_b0.pt')
    
    return model
```

### Data Augmentation

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Key Parameters Explained

- **pretrained=True**: Uses ImageNet weights for transfer learning.
- **Dropout(p=0.3)**: Regularization to prevent overfitting.
- **lr=0.0001**: Low learning rate for fine-tuning pre-trained weights.
- **weight_decay=0.01**: L2 regularization for optimizer.
- **CosineAnnealingLR**: Learning rate schedule for better convergence.
- **class weighting**: Handles imbalance between real and deepfake samples.

---

## Voice Deepfake Detection

### Algorithm: Random Forest with Audio Features

**Rationale:** Random Forest classifier with audio feature extraction using librosa. The algorithm extracts spectral and temporal features from audio signals to detect synthetic voice patterns.

### Implementation Code

```python
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def extract_audio_features(audio_path, max_length=3.0):
    """Extract audio features for deepfake detection"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=max_length)
    
    # Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Combine all features
    features = np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean, chroma_std,
        [np.mean(spectral_centroid), np.std(spectral_centroid)],
        [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
        [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
        [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],
        [tempo]
    ])
    
    return features

def train_voice_model(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Extract features from all audio files
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        feature_vector = extract_audio_features(audio_path)
        features.append(feature_vector)
        labels.append(row['label'])
    
    x = np.array(features)
    y = np.array(labels)
    
    # Three-way dataset split (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=400,                # Number of trees
        random_state=42,
        class_weight='balanced_subsample', # Handle class imbalance
        min_samples_leaf=2,               # Prevent overfitting
        n_jobs=-1,                        # Use all cores
        max_features='sqrt'                # Features per split
    )
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predictions
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    # Save model
    joblib.dump(model, 'voice_model.pkl')
    
    return model, y_pred, y_prob
```

### Audio Features Explained

- **MFCCs (13 features)**: Mel-frequency cepstral coefficients capture spectral envelope of audio.
- **Chroma features (12 features)**: Represent pitch-related information.
- **Spectral centroid**: Center of mass of the spectrum, indicates brightness.
- **Spectral rolloff**: Frequency below which a specified percentage of energy lies.
- **Spectral bandwidth**: Width of the spectrum.
- **Zero crossing rate**: Rate at which signal changes sign, indicates noisiness.
- **Tempo**: Beats per minute of the audio.

### Key Parameters Explained

- **n_estimators=400**: Number of decision trees in the forest.
- **class_weight='balanced_subsample'**: Weights classes based on bootstrap samples.
- **min_samples_leaf=2**: Minimum samples required at leaf node for regularization.
- **max_features='sqrt'**: Uses sqrt(n_features) for each split, reducing correlation between trees.

---

## Fusion Layer

### Algorithm: Weighted Fusion of Multi-Modal Predictions

**Rationale:** The fusion layer combines predictions from multiple detection modalities using weighted averaging based on model confidence scores. This ensemble approach improves overall detection accuracy and robustness.

### Implementation Code

```python
import numpy as np

class WeightedFusion:
    def __init__(self, weights=None):
        """
        Initialize fusion layer with optional weights.
        If weights is None, uses confidence-based dynamic weighting.
        """
        self.weights = weights
        self.modalities = ['url', 'email', 'sms', 'qr', 'deepfake', 'voice']
    
    def fuse_predictions(self, predictions):
        """
        Fuse predictions from multiple modalities.
        
        Args:
            predictions: dict of {modality: probability_score}
            
        Returns:
            fused_score: weighted average of predictions
            confidence: confidence in the fused prediction
        """
        if self.weights is None:
            # Dynamic weighting based on prediction confidence
            weights = self._compute_dynamic_weights(predictions)
        else:
            weights = self.weights
        
        # Weighted average
        fused_score = sum(
            predictions[mod] * weights[mod] 
            for mod in self.modalities 
            if mod in predictions
        )
        
        # Normalize weights
        total_weight = sum(weights.get(mod, 0) for mod in self.modalities)
        if total_weight > 0:
            fused_score /= total_weight
        
        # Compute confidence
        confidence = self._compute_confidence(predictions, weights)
        
        return fused_score, confidence
    
    def _compute_dynamic_weights(self, predictions):
        """Compute weights based on prediction confidence"""
        weights = {}
        for mod in self.modalities:
            if mod in predictions:
                prob = predictions[mod]
                # Higher confidence predictions get higher weights
                # Weight = |prob - 0.5| (distance from uncertainty)
                weights[mod] = abs(prob - 0.5) + 0.1  # Minimum weight of 0.1
        return weights
    
    def _compute_confidence(self, predictions, weights):
        """Compute overall confidence in fused prediction"""
        # Confidence is weighted average of individual confidences
        confidences = []
        for mod in self.modalities:
            if mod in predictions:
                prob = predictions[mod]
                # Confidence = max(prob, 1-prob)
                conf = max(prob, 1 - prob)
                confidences.append(conf * weights.get(mod, 0))
        
        if confidences:
            return sum(confidences) / sum(weights.values())
        return 0.5

# Example usage
fusion = WeightedFusion()

predictions = {
    'url': 0.85,        # High confidence phishing
    'email': 0.72,      # Moderate confidence phishing
    'sms': 0.45,        # Low confidence (near threshold)
    'qr': 0.91,         # High confidence phishing
    'deepfake': 0.15,   # High confidence legitimate
    'voice': 0.30       # Moderate confidence legitimate
}

fused_score, confidence = fusion.fuse_predictions(predictions)

print(f"Fused phishing probability: {fused_score:.3f}")
print(f"Confidence: {confidence:.3f}")
```

### Static Weight Configuration

```python
# Predefined weights based on model performance
STATIC_WEIGHTS = {
    'url': 0.25,        # URL model has high accuracy
    'email': 0.20,      # Email model performs well
    'sms': 0.15,       # SMS model moderate performance
    'qr': 0.20,        # QR model strong performance
    'deepfake': 0.10,  # Deepfake model lower confidence
    'voice': 0.10       # Voice model lower confidence
}

fusion = WeightedFusion(weights=STATIC_WEIGHTS)
```

### Decision Thresholds

```python
def make_decision(fused_score, confidence, threshold=0.5):
    """
    Make final decision based on fused score and confidence.
    
    Args:
        fused_score: Fused phishing probability
        confidence: Confidence in the prediction
        threshold: Decision threshold (default 0.5)
        
    Returns:
        decision: 'phishing', 'legitimate', or 'uncertain'
    """
    if confidence < 0.6:
        return 'uncertain'  # Low confidence, require manual review
    
    if fused_score >= threshold:
        return 'phishing'
    else:
        return 'legitimate'
```

---

## Evaluation Metrics

All models are evaluated using the following metrics:

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob):
    """Comprehensive model evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics
```

### Dataset Split Methodology

All models use a three-way split for robust evaluation:
- **Training set (70%)**: Model training and parameter tuning
- **Validation set (15%)**: Hyperparameter optimization and early stopping
- **Test set (15%)**: Final model evaluation without any tuning

This approach prevents data leakage and provides unbiased performance estimates.

---

## References

1. XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining.

2. EfficientNet: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning.

3. TF-IDF: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information processing & management.

4. Random Forest: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

5. Librosa: McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python.

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{multimodal_phishing_detection,
  title={A Machine Learning-Based Multimodal Approach for Detection and Prevention of Emerging Phishing Attacks},
  author={Your Name},
  year={2024},
  url={https://github.com/bashyauri/Multidimentional-Phising-Detector}
}
```
