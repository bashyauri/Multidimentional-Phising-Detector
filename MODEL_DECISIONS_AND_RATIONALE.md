# Model Selection and Parameter Rationale
## A Machine Learning-Based Multimodal Approach for Detection and Prevention of Emerging Phishing Attacks

---

### Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [URL Phishing Detection Model](#url-phishing-detection-model)
3. [Email Phishing Detection Model](#email-phishing-detection-model)
4. [SMS Phishing Detection Model](#sms-phishing-detection-model)
5. [QR Code Phishing Detection Model](#qr-code-phishing-detection-model)
6. [Deepfake Detection Model](#deepfake-detection-model)
7. [Voice Deepfake Detection Model](#voice-deepfake-detection-model)
8. [Multimodal Fusion Strategy](#multimodal-fusion-strategy)
9. [Dataset Selection and Preprocessing](#dataset-selection-and-preprocessing)
10. [Performance Evaluation Metrics](#performance-evaluation-metrics)
11. [API Response Structure and Research-Focused Debug Information](#api-response-structure-and-research-focused-debug-information)
12. [Conclusion](#conclusion)

---

## System Architecture Overview

The phishing detection system employs a **multimodal ensemble approach** that combines multiple specialized models to detect phishing across different attack vectors: URLs, emails, SMS messages, QR codes, deepfake media, and voice deepfakes. This architecture was chosen because:

1. **Attack Vector Diversity**: Modern phishing attacks use multiple channels (URL, email, SMS, QR, media), requiring specialized detection for each modality
2. **Complementary Strengths**: Different models capture different signal types (textual patterns, structural features, visual artifacts, audio characteristics)
3. **Redundancy and Robustness**: Multiple detection pathways provide fallback mechanisms if one modality fails
4. **Fusion Benefits**: Combining multiple signals improves overall detection accuracy beyond individual model performance

The system uses a **weighted fusion approach** where each modality contributes to a final phishing probability based on its reliability and the specific attack scenario.

---

## ML Model Selection Rationale by Pipeline

### 1. URL Phishing Detection Pipeline

**Selected Model**: XGBoost Classifier

**Itemized Reasons for Selection**:

1. **Superior Performance on Tabular Data**: XGBoost consistently outperforms other tree-based ensembles (Random Forest, Gradient Boosting) on structured URL feature data, achieving 99.74% accuracy
2. **Non-linear Relationship Handling**: URL features exhibit complex, non-linear relationships (e.g., domain length vs. phishing probability) that gradient boosting captures effectively better than linear models
3. **Feature Importance Interpretability**: XGBoost provides clear feature importance scores, enabling explainability of which URL characteristics (e.g., IP address, special characters, subdomain count) are most predictive
4. **Built-in Regularization**: L1/L2 regularization parameters prevent overfitting on high-dimensional URL feature spaces (63 features)
5. **Computational Efficiency**: Histogram-based tree building (tree_method="hist") enables fast training on large datasets (47,000+ URLs) with minimal memory usage
6. **Missing Value Robustness**: Handles missing features gracefully, important when some URL features cannot be extracted (e.g., WHOIS data unavailable)
7. **Proven Track Record**: XGBoost is the state-of-the-art for tabular classification tasks and has won numerous Kaggle competitions for similar feature-based problems

**Why Not Other Models**:
- **Logistic Regression**: Cannot capture non-linear relationships in URL features
- **Random Forest**: Slightly lower performance and less efficient training on large datasets
- **Neural Networks**: Overkill for tabular data, harder to interpret, requires more computational resources
- **SVM**: Poor scalability to large datasets, difficult to handle high-dimensional feature spaces

---

### 2. Email Phishing Detection Pipeline

**Selected Model**: Logistic Regression with TF-IDF

**Itemized Reasons for Selection**:

1. **Text Classification Baseline**: Logistic Regression with TF-IDF is a strong baseline for text classification, often competitive with more complex models on well-structured datasets
2. **Interpretability**: Model coefficients directly indicate which terms are most predictive of phishing (e.g., "verify," "account," "urgent"), aiding in explainability and debugging
3. **Training Efficiency**: Fast training time (seconds vs. hours for deep learning), suitable for iterative development and rapid prototyping
4. **Low Resource Requirements**: No GPU required, making deployment easier and more cost-effective
5. **Effective with TF-IDF**: TF-IDF captures term importance which is sufficient for phishing email detection where specific suspicious terms are highly indicative
6. **Proven Effectiveness**: Logistic Regression with TF-IDF achieves 99.30% accuracy on email phishing, demonstrating that simple models can be highly effective on this task
7. **Regularization Built-in**: L2 regularization prevents overfitting on high-dimensional text features (10,000 features)

**Why Not Other Models**:
- **Naive Bayes**: Assumes feature independence which doesn't hold for email text, lower accuracy
- **SVM**: Slower training on large text datasets, harder to scale
- **BERT/Transformers**: Overkill for this task, requires GPU, longer training time, marginal accuracy improvement (99.30% vs. potential 99.5%)
- **LSTM/RNN**: More complex, harder to train, not necessary for email classification where bag-of-words features work well

---

### 3. SMS Phishing Detection Pipeline

**Selected Model**: Logistic Regression with TF-IDF (Balanced)

**Itemized Reasons for Selection**:

1. **Short Text Suitability**: SMS messages are short (160 chars), making complex models prone to overfitting. Simple models generalize better on short texts
2. **Class Imbalance Handling**: Built-in class weighting addresses imbalance in SMS datasets (more legitimate than phishing messages)
3. **Fast Inference**: Critical for real-time SMS filtering applications where low latency is required
4. **Proven Effectiveness**: Logistic Regression with TF-IDF is state-of-the-art for SMS spam detection, achieving 97.85% accuracy
5. **Computational Efficiency**: Can run on mobile devices or edge computing platforms for on-device SMS filtering
6. **Interpretability**: Clear coefficients show which SMS terms (e.g., "free," "win," "call now") are most suspicious
7. **Lower Feature Dimensionality**: SMS texts have less vocabulary diversity than emails, allowing smaller feature sets (8,000 vs. 10,000)

**Why Not Other Models**:
- **Naive Bayes**: Commonly used for SMS spam but lower precision (91.39% vs. higher for LR)
- **BERT**: Overkill for short texts, high computational cost for marginal accuracy gain
- **XGBoost**: More complex than needed for short texts, harder to interpret
- **SVM**: Slower training and inference, not suitable for real-time SMS filtering

---

### 4. QR Code Phishing Detection Pipeline

**Selected Model**: XGBoost with Multimodal Features

**Itemized Reasons for Selection**:

1. **Multimodal Feature Fusion**: Combines visual QR features (image statistics, error correction patterns) with decoded URL analysis. XGBoost effectively handles heterogeneous feature types (numeric visual features + URL probability)
2. **Interpretability**: Feature importance reveals which aspects (visual vs. URL) contribute most to detection, aiding in model improvement
3. **Computational Efficiency**: Faster than CNN-based QR classification approaches, suitable for real-time QR scanning
4. **Handling Missing Data**: Gracefully handles cases where QR cannot be decoded (visual-only analysis), providing robustness
5. **High Performance**: Achieves 90.65% accuracy on QR phishing detection, competitive with more complex approaches
6. **Feature Engineering Flexibility**: Can easily incorporate additional features (e.g., QR version, error correction level, module count)
7. **Regularization**: Strong regularization (early stopping, L2 regularization) prevents overfitting on visual features which can be noisy

**Why Not Other Models**:
- **CNN-based QR Classification**: More complex, requires more training data, harder to interpret, similar accuracy
- **Vision Transformers**: Overkill for QR classification, requires large datasets, high computational cost
- **Ensemble Methods**: More complex than needed, XGBoost already provides ensemble benefits
- **URL-only Analysis**: Misses visual patterns in malicious QR codes, lower accuracy

---

### 5. Deepfake Detection Pipeline

**Selected Model**: EfficientNet-B0 CNN with Frame-Level Classification

**Itemized Reasons for Selection**:

1. **Efficiency-Accuracy Tradeoff**: EfficientNet-B0 provides excellent accuracy (83.50%) with minimal parameters (5.3M) and computational cost, making it laptop-friendly
2. **Transfer Learning Benefits**: Pretrained on ImageNet provides strong low-level feature extractors (edges, textures) that transfer well to deepfake detection
3. **Frame-Level Processing**: Processes individual frames rather than full videos, reducing computational complexity while maintaining detection capability
4. **Laptop-Friendly**: Can be trained on CPU or low-end GPUs, making it accessible for research environments without expensive hardware
5. **Proven Performance**: EfficientNet architectures have shown strong results on deepfake detection tasks in research literature
6. **Fine-Tuning Strategy**: Unfreezing only last 2 feature blocks preserves most pretrained knowledge while adapting to deepfake-specific patterns, preventing overfitting on small datasets
7. **Memory Efficiency**: Frame-level processing with batch size 2 keeps memory usage low, suitable for systems with limited RAM

**Why Not Other Models**:
- **ResNet-50**: More parameters (25M), higher computational cost, similar accuracy
- **VGG-16**: Very large parameters (138M), slow training, prone to overfitting on small datasets
- **ConvNeXt-Tiny**: More complex, higher computational cost, marginal accuracy improvement on small datasets
- **3D CNNs**: Much higher computational cost, requires more training data, overkill for frame-level detection
- **Vision Transformers**: Requires large datasets, high computational cost, not suitable for small datasets (2,000 videos)

---

### 6. Voice Deepfake Detection Pipeline

**Selected Model**: Random Forest Classifier

**Itemized Reasons for Selection**:

1. **Robustness to Feature Scaling**: Random Forest doesn't require feature normalization, simplifying preprocessing pipeline
2. **Handling Non-linear Relationships**: Audio features (MFCC, spectral centroid, pitch) have complex, non-linear relationships that ensemble methods capture well
3. **Interpretability**: Feature importance reveals which audio characteristics (pitch, spectral features) are most discriminative for voice deepfake detection
4. **Training Efficiency**: Faster than neural networks for tabular audio features, suitable for iterative development
5. **Resistance to Overfitting**: Ensemble averaging and feature sampling provide built-in regularization, important for audio features which can be noisy
6. **High Performance**: Achieves 98.51% accuracy on voice deepfake detection, excellent performance for this modality
7. **Parallel Processing**: n_jobs=-1 enables training across all CPU cores, significantly reducing training time

**Why Not Other Models**:
- **XGBoost**: Similar performance but Random Forest is simpler and requires less hyperparameter tuning
- **SVM**: Poor scalability to high-dimensional audio features, slower training
- **Neural Networks**: Overkill for hand-crafted audio features, requires more data, harder to interpret
- **CNN for Spectrograms**: More complex, requires more training data, similar accuracy to hand-crafted features + Random Forest

---

## URL Phishing Detection Model

### Model Selection: XGBoost Classifier

**Chosen Model**: XGBoost (Extreme Gradient Boosting) Classifier  
**Alternative Models Considered**: Random Forest, Logistic Regression, Neural Networks, SVM

**Rationale for XGBoost Selection**:

1. **Superior Performance on Tabular Data**: XGBoost consistently outperforms other tree-based ensembles on structured feature data [Chen & Guestrin, 2016]
2. **Handling Non-linear Relationships**: URL features exhibit complex, non-linear relationships that gradient boosting captures effectively
3. **Feature Importance Interpretability**: XGBoost provides clear feature importance scores, aiding in model explainability
4. **Regularization Capabilities**: Built-in L1/L2 regularization prevents overfitting on high-dimensional URL feature spaces
5. **Computational Efficiency**: Histogram-based tree building (tree_method="hist") enables fast training on large datasets
6. **Missing Value Handling**: Robust to missing features, which is important when some URL features cannot be extracted

### Parameter Selection

```python
XGBClassifier(
    n_estimators=450,           # Number of boosting rounds
    max_depth=5,               # Maximum tree depth
    learning_rate=0.05,        # Step size shrinkage
    subsample=0.9,             # Row sampling for regularization
    colsample_bytree=0.9,      # Column sampling for regularization
    eval_metric="logloss",      # Evaluation metric
    random_state=42,           # Reproducibility
    n_jobs=2,                  # Parallel processing
    tree_method="hist",         # Histogram-based algorithm
)
```

**Parameter Rationale**:

- **n_estimators=450**: Higher number of trees provides better model capacity while maintaining regularization through learning rate. Chosen through cross-validation to balance performance and training time.
- **max_depth=5**: Limits tree complexity to prevent overfitting. URL features have moderate complexity; depth 5 captures sufficient interactions without memorizing noise.
- **learning_rate=0.05**: Conservative learning rate combined with higher n_estimators provides better generalization. Smaller learning rates typically yield better final performance [Chen & Guestrin, 2016].
- **subsample=0.9, colsample_bytree=0.9**: Stochastic gradient boosting with 90% sampling provides regularization, reducing overfitting while maintaining model diversity.
- **tree_method="hist"**: Histogram-based algorithm is faster and more memory-efficient than exact greedy method, suitable for large URL datasets.

### Feature Engineering

**Dataset**: PhiUSIIL Phishing URL Dataset  
**Features Extracted**: 63 structural, lexical, and host-based features including:
- URL length and character distributions
- Presence of IP addresses, special characters
- Domain age and WHOIS information
- Subdomain count and TLD analysis
- Suspicious token presence

**Feature Selection Rationale**: URL phishing detection relies heavily on structural patterns rather than semantic content. Features were selected based on prior research showing their effectiveness [Moghaddam et al., 2022].

### Performance Metrics

- **Accuracy**: 99.74%
- **Precision**: 99.57%
- **Recall**: 99.97%
- **F1-Score**: 99.77%
- **ROC-AUC**: 99.90%

**Interpretation**: The high performance indicates that URL structural features are highly discriminative for phishing detection, making XGBoost an excellent choice for this modality.

---

## Email Phishing Detection Model

### Model Selection: Logistic Regression with TF-IDF

**Chosen Model**: Logistic Regression with TF-IDF Vectorization  
**Alternative Models Considered**: Naive Bayes, SVM, BERT, LSTM, XGBoost

**Rationale for Logistic Regression Selection**:

1. **Text Classification Baseline**: Logistic Regression with TF-IDF is a strong baseline for text classification, often competitive with more complex models on well-structured datasets [Wang & Manning, 2012]
2. **Interpretability**: Model coefficients directly indicate which terms are most predictive of phishing, aiding in explainability
3. **Training Efficiency**: Fast training time compared to deep learning approaches, suitable for iterative development
4. **Low Resource Requirements**: No GPU required, making deployment easier
5. **Effective with TF-IDF**: TF-IDF captures term importance which is sufficient for phishing email detection where specific suspicious terms are highly indicative

### Parameter Selection

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,      # Maximum vocabulary size
        ngram_range=(1, 2)        # Unigrams and bigrams
    )),
    ("clf", LogisticRegression(
        max_iter=2000            # Maximum optimization iterations
    ))
])
```

**Parameter Rationale**:

- **max_features=10000**: Limits vocabulary to most frequent 10,000 terms, reducing dimensionality while capturing important terms. Balances model capacity and computational efficiency.
- **ngram_range=(1, 2)**: Captures both individual words and word pairs. Bigrams capture phrases like "verify account" or "urgent action" that are indicative of phishing.
- **max_iter=2000**: Sufficient iterations for convergence on email datasets. Higher than default ensures convergence on larger feature spaces.

### Feature Engineering

**Dataset**: Custom email phishing dataset  
**Preprocessing**: Text cleaning (lowercasing, removing special characters, stopword removal)  
**Vectorization**: TF-IDF weighting scheme

**Rationale for TF-IDF**: Term Frequency-Inverse Document Frequency emphasizes rare, discriminative terms while downweighting common terms. This is ideal for phishing detection where terms like "verify," "account," "urgent" are strong indicators.

### Performance Metrics

- **Accuracy**: 99.30%
- **Precision**: 99.20%
- **Recall**: 99.54%
- **F1-Score**: 99.37%
- **ROC-AUC**: 99.96%

**Interpretation**: Excellent performance demonstrates that phishing emails contain distinctive linguistic patterns that TF-IDF + Logistic Regression effectively captures.

---

## SMS Phishing Detection Model

### Model Selection: Logistic Regression with TF-IDF (Balanced)

**Chosen Model**: Logistic Regression with TF-IDF and class weighting  
**Alternative Models Considered**: Naive Bayes, SVM, BERT, LSTM

**Rationale for Logistic Regression Selection**:

Similar to email detection, with additional considerations:
1. **Class Imbalance Handling**: SMS datasets often have imbalance (more legitimate than phishing messages). Class weighting addresses this.
2. **Short Text Nature**: SMS messages are short (160 chars), making complex models prone to overfitting. Simple models generalize better.
3. **Fast Inference**: Critical for real-time SMS filtering applications.
4. **Proven Effectiveness**: Logistic Regression with TF-IDF is state-of-the-art for SMS spam detection [Almeida et al., 2011]

### Parameter Selection

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,       # Slightly smaller than email due to shorter texts
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=3000,           # Higher iterations for convergence
        class_weight="balanced"  # Handle class imbalance
    ))
])
```

**Parameter Rationale**:

- **max_features=8000**: Reduced from 10,000 (email) because SMS messages are shorter and have less vocabulary diversity.
- **class_weight="balanced"**: Automatically adjusts weights inversely proportional to class frequencies, addressing imbalance in SMS datasets.
- **max_iter=3000**: Increased from 2000 to ensure convergence with class weighting.

### Feature Engineering

**Dataset**: SMS Spam Collection Dataset  
**Preprocessing**: Text cleaning, handling tab-separated format variations  
**Vectorization**: TF-IDF with bigrams

**Rationale**: Similar to email, but adapted for shorter texts. Bigrams capture phrases like "free msg," "call now," "win prize" common in SMS spam.

### Performance Metrics

- **Accuracy**: 97.85%
- **Precision**: 91.39%
- **Recall**: 92.62%
- **F1-Score**: 92.00%
- **ROC-AUC**: 98.77%

**Interpretation**: Slightly lower than email due to shorter text length and higher variability in SMS language patterns. Still strong performance for practical deployment.

---

## QR Code Phishing Detection Model

### Model Selection: XGBoost with Multimodal Features

**Chosen Model**: XGBoost Classifier with QR visual features + URL analysis  
**Alternative Models Considered**: CNN-based QR classification, Vision Transformers, Ensemble methods

**Rationale for XGBoost Selection**:

1. **Multimodal Feature Fusion**: Combines visual QR features (image statistics, error correction patterns) with decoded URL analysis. XGBoost effectively handles heterogeneous feature types.
2. **Interpretability**: Feature importance reveals which aspects (visual vs. URL) contribute most to detection.
3. **Computational Efficiency**: Faster than CNN approaches for QR classification, suitable for real-time scanning.
4. **Handling Missing Data**: Gracefully handles cases where QR cannot be decoded (visual-only analysis).

### Parameter Selection

```python
XGBClassifier(
    n_estimators=1000,          # Higher than URL model for complex visual features
    learning_rate=0.03,         # Lower learning rate for better convergence
    max_depth=4,                # Shallower trees to prevent overfitting on visual features
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=1.0,             # L2 regularization
    gamma=0.0,                  # Minimum loss reduction
    scale_pos_weight=scale_pos_weight,  # Class imbalance handling
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2,
    early_stopping_rounds=50    # Early stopping on validation set
)
```

**Parameter Rationale**:

- **n_estimators=1000**: Higher than URL model because visual features are more complex and require more boosting rounds.
- **learning_rate=0.03**: Lower learning rate (vs 0.05 for URL) because visual features are noisier; slower learning prevents overfitting.
- **max_depth=4**: Shallower than URL model (depth 5) because visual features are less structured; prevents overfitting.
- **reg_lambda=1.0**: L2 regularization adds penalty for large coefficients, important for high-dimensional visual features.
- **early_stopping_rounds=50**: Prevents overfitting by stopping when validation performance plateaus.
- **scale_pos_weight**: Addresses class imbalance in QR dataset (may have unequal benign/malicious samples).

### Feature Engineering

**Dataset**: PhiUSIIL QR Code Dataset (generated from URL dataset)  
**Features**:
- **Visual Features**: Image statistics (mean, std, entropy), error correction level, QR version, module count
- **Decoded URL Features**: If QR decodes successfully, extract URL features and analyze with URL model
- **Hybrid Features**: Combination of visual patterns and decoded URL characteristics

**Rationale**: QR phishing detection requires analyzing both visual patterns (some malicious QRs have visual artifacts) and the embedded URL content. Multimodal fusion improves detection over single-modality approaches.

### Performance Metrics

- **Accuracy**: 90.65%
- **Precision**: 88.83%
- **Recall**: 93.00%
- **F1-Score**: 90.86%
- **ROC-AUC**: 96.72%

**Interpretation**: Lower than URL/email models because QR visual features are less discriminative than text patterns. However, combining visual and URL analysis achieves strong performance suitable for practical use.

---

## Deepfake Detection Model

### Model Selection: EfficientNet-B0 Frame Classifier

**Chosen Model**: EfficientNet-B0 CNN with frame-level classification  
**Alternative Models Considered**: ResNet-50, VGG-16, ConvNeXt-Tiny, Vision Transformers, 3D CNNs

**Rationale for EfficientNet-B0 Selection**:

1. **Efficiency-Accuracy Tradeoff**: EfficientNet-B0 provides excellent accuracy with minimal parameters and computational cost [Tan & Le, 2019]
2. **Transfer Learning Benefits**: Pretrained on ImageNet, providing strong feature extraction for face/manipulation detection
3. **Frame-Level Processing**: Processes individual frames rather than full videos, reducing computational complexity while maintaining detection capability
4. **Laptop-Friendly**: Can be trained on CPU or low-end GPUs, making it accessible for research environments
5. **Proven Performance**: EfficientNet architectures have shown strong results on deepfake detection tasks [Dolhansky et al., 2020]

### Parameter Selection

```python
# Model Architecture
architecture = "efficientnet_b0"
image_size = 160                # Reduced from standard 224 for efficiency
frames_per_video = 4            # Sample 4 frames per video
face_crop = True                # Crop to face region
pretrained = True               # Use ImageNet weights
unfreeze_last_blocks = 2        # Fine-tune last 2 feature blocks

# Training Parameters
epochs = 8
batch_size = 2
learning_rate_backbone = 2e-4   # Lower LR for pretrained features
learning_rate_classifier = 1e-3 # Higher LR for new classifier
weight_decay = 1e-4
early_stop_patience = 2
```

**Parameter Rationale**:

- **image_size=160**: Reduced from standard 224 to improve training speed and memory usage. Face regions don't require full resolution for deepfake detection.
- **frames_per_video=4**: Sampling 4 frames provides temporal coverage while keeping computational cost manageable. More frames provide diminishing returns for detection accuracy.
- **face_crop=True**: Cropping to face region focuses on manipulation artifacts (blending inconsistencies, artifacts) rather than background noise.
- **pretrained=True**: ImageNet pretraining provides strong low-level feature extractors (edges, textures) that transfer well to deepfake detection.
- **unfreeze_last_blocks=2**: Fine-tuning only the last 2 blocks preserves most pretrained features while adapting to deepfake-specific patterns. Full fine-tuning risks overfitting on small datasets.
- **learning_rate_backbone=2e-4, learning_rate_classifier=1e-3**: Differential learning rates allow classifier to learn faster while backbone adapts slowly, preserving pretrained knowledge.
- **weight_decay=1e-4**: L2 regularization prevents overfitting on small deepfake datasets (2000 videos).
- **early_stop_patience=2**: Aggressive early stopping prevents overfitting on limited data.

### Feature Engineering

**Dataset**: FaceForensics++ Dataset  
**Preprocessing**:
- Face detection and cropping using MTCNN
- Frame sampling (uniform temporal distribution)
- Image normalization (ImageNet statistics)
- Data augmentation (horizontal flip, brightness, contrast)

**Rationale**: Deepfake detection focuses on facial manipulation artifacts. Face cropping removes background noise and focuses on regions where manipulation is most visible.

### Performance Metrics

- **Accuracy**: 83.50%
- **Precision**: 87.22%
- **Recall**: 78.50%
- **F1-Score**: 82.63%
- **ROC-AUC**: 92.10%

**Interpretation**: Lower performance than text-based models due to:
1. Limited dataset size (2000 videos vs 47,000+ URLs)
2. Complexity of visual manipulation detection
3. Variability in deepfake generation techniques

Performance is competitive with state-of-the-art on similar dataset sizes. Accuracy can be improved with larger datasets (planned expansion to 10,000 videos).

---

## Voice Deepfake Detection Model

### Model Selection: Random Forest Classifier

**Chosen Model**: Random Forest Classifier  
**Alternative Models Considered**: XGBoost, SVM, Neural Networks, CNN for spectrograms

**Rationale for Random Forest Selection**:

1. **Robustness to Feature Scaling**: Random Forest doesn't require feature normalization, simplifying preprocessing
2. **Handling Non-linear Relationships**: Audio features have complex, non-linear relationships that ensemble methods capture well
3. **Interpretability**: Feature importance reveals which audio characteristics (pitch, spectral features) are most discriminative
4. **Training Efficiency**: Faster than neural networks for tabular audio features
5. **Resistance to Overfitting**: Ensemble averaging and feature sampling provide built-in regularization

### Parameter Selection

```python
RandomForestClassifier(
    n_estimators=100,            # Number of trees
    random_state=42,
    class_weight="balanced_subsample",  # Handle class imbalance
    min_samples_leaf=2,         # Prevent overfitting
    n_jobs=-1                   # Use all CPU cores
)
```

**Parameter Rationale**:

- **n_estimators=100**: Sufficient trees for stable predictions without excessive computational cost. Performance plateaus beyond 100 trees for this dataset.
- **class_weight="balanced_subsample"**: Addresses class imbalance while maintaining bootstrap sampling benefits.
- **min_samples_leaf=2**: Prevents overfitting by requiring minimum samples in leaf nodes.
- **n_jobs=-1**: Parallel training across all CPU cores for faster training.

### Feature Engineering

**Dataset**: Balanced voice deepfake dataset  
**Features**: Audio spectral features (MFCC, spectral centroid, spectral contrast, pitch, zero-crossing rate)  
**Preprocessing**: Audio normalization, feature extraction using librosa

**Rationale**: Voice deepfakes introduce spectral artifacts (unnatural pitch, spectral inconsistencies). Hand-crafted audio features capture these artifacts effectively without requiring deep learning.

### Performance Metrics

- **Accuracy**: 98.51%
- **Precision**: 97.98%
- **Recall**: 99.07%
- **F1-Score**: 98.52%
- **ROC-AUC**: 99.92%

**Interpretation**: Excellent performance indicates that voice deepfakes introduce distinctive spectral artifacts that Random Forest with hand-crafted features effectively captures.

---

## Multimodal Fusion Strategy

### Fusion Approach: Weighted Probability Fusion

**Chosen Strategy**: Weighted fusion of individual model probabilities with modality-specific weights  
**Alternative Strategies Considered**: Majority voting, stacking, neural network fusion, Bayesian fusion

**Rationale for Weighted Fusion Selection**:

1. **Probabilistic Interpretation**: Each model outputs a phishing probability, making fusion straightforward
2. **Modality-Specific Weights**: Different modalities have different reliability in different scenarios (e.g., URL analysis is more reliable than QR visual analysis)
3. **Explainability**: Weights can be interpreted as confidence in each modality
4. **Flexibility**: Weights can be adjusted based on attack patterns or domain knowledge
5. **Computational Efficiency**: Simple weighted sum is fast for real-time inference

### Fusion Weights and Decision Logic

**QR-URL Fusion**:
```python
if url_probability >= 0.5:
    # URL dominates for security if clearly suspicious
    final_probability = url_probability
    fusion_weights = {"url": 1.0, "qr": 0.0}
else:
    # Weighted fusion when URL is not clearly suspicious
    final_probability = 0.7 * url_probability + 0.3 * qr_probability
    fusion_weights = {"url": 0.7, "qr": 0.3}
```

**Rationale**:
- **URL weight 0.7**: URL analysis examines actual domain content, which is more reliable for phishing detection than QR visual patterns
- **QR weight 0.3**: QR visual model analyzes image patterns, which can be less reliable
- **URL dominance rule**: If URL probability >= 0.5, URL dominates completely for security - prevents QR visual model from overriding a clearly malicious URL

**General Multimodal Fusion**:
- Each modality contributes its phishing probability
- Weights are assigned based on modality reliability and attack scenario
- Final decision based on weighted probability compared to threshold (0.5)

### Decision Threshold

**Standard Threshold**: 0.5 for all modalities  
**Rationale**: Binary classification threshold of 0.5 provides balanced false positive/negative rates. Some modalities (deepfake) use optimized thresholds found through validation (e.g., 0.44 for deepfake) to maximize balanced accuracy and F1-score.

---

## Dataset Selection and Preprocessing

### URL Dataset: PhiUSIIL Phishing URL Dataset

**Selection Rationale**:
1. **Large Scale**: 47,159+ URLs provide sufficient training data
2. **Recent**: Contains modern phishing patterns (2022-2023)
3. **Diverse**: Covers various phishing types (credential harvesting, banking, etc.)
4. **Well-Labeled**: High-quality labels with minimal noise
5. **Feature-Rich**: Includes precomputed features for comprehensive analysis

### Email Dataset: Custom Email Phishing Dataset

**Selection Rationale**:
1. **Domain-Specific**: Tailored to phishing email patterns
2. **Balanced**: Equal phishing/legitimate samples prevent bias
3. **Real-World**: Contains actual phishing email samples
4. **Text Quality**: Clean, well-formatted email content

### SMS Dataset: SMS Spam Collection

**Selection Rationale**:
1. **Benchmark Dataset**: Widely used in SMS spam research [Almeida et al., 2011]
2. **Real Messages**: Contains actual SMS spam and legitimate messages
3. **Standard Format**: Tab-separated format with clear labels
4. **Community Validation**: Extensively validated by research community

### QR Dataset: PhiUSIIL QR Code Dataset (Generated)

**Selection Rationale**:
1. **Consistent with URL Dataset**: Generated from same URLs as URL dataset, ensuring consistency
2. **Large Scale**: 9,999 QR codes (5,000 benign, 5,000 malicious)
3. **Realistic**: QR codes generated with realistic error correction levels and versions
4. **Decodable**: 100% decode rate ensures URL analysis is always possible

### Deepfake Dataset: FaceForensics++

**Selection Rationale**:
1. **Benchmark Dataset**: Standard dataset for deepfake detection research [Rössler et al., 2019]
2. **Multiple Manipulation Types**: Deepfakes, FaceSwap, Face2Face, NeuralTextures
3. **High Quality**: Professional-grade manipulations
4. **Large Scale**: 1,000 original + 1,000 manipulated videos (subset used)
5. **Face-Centric**: Focuses on facial manipulation, relevant for phishing deepfakes

**Note**: Current dataset size (2,000 videos) limits performance. Planned expansion to 10,000 videos to improve accuracy to 90%+.

### Voice Dataset: Balanced Voice Deepfake Dataset

**Selection Rationale**:
1. **Balanced**: Equal real/fake samples
2. **Spectral Features**: Pre-extracted audio features reduce preprocessing complexity
3. **Modern**: Contains recent deepfake voice generation techniques
4. **High Quality**: Clean audio with minimal noise

---

## Performance Evaluation Metrics

### Metric Selection

**Primary Metrics**:
- **Accuracy**: Overall correctness, suitable for balanced datasets
- **Precision**: False positive rate (important for user experience - minimize false alarms)
- **Recall**: False negative rate (important for security - minimize missed attacks)
- **F1-Score**: Harmonic mean of precision and recall, balances both concerns
- **ROC-AUC**: Area under ROC curve, threshold-independent performance measure

**Rationale**:
- **Multi-Metric Evaluation**: No single metric captures all aspects of performance. Multiple metrics provide comprehensive view.
- **Precision-Recall Tradeoff**: Phishing detection requires balancing false positives (user inconvenience) and false negatives (security risk). F1-score captures this balance.
- **ROC-AUC**: Threshold-independent measure allows comparison across different decision thresholds.

### Threshold Optimization

**Strategy**: Maximize 0.7 × balanced_accuracy + 0.3 × macro_F1 over thresholds [0.2, 0.9]

**Rationale**:
- **Balanced Accuracy**: Accounts for class imbalance, ensuring performance on both classes
- **Macro F1**: Balances precision and recall across classes
- **Weighted Combination**: 70% weight on accuracy (overall performance), 30% on F1 (precision-recall balance)
- **Threshold Range**: [0.2, 0.9] explores reasonable thresholds, avoiding extremes

---

## Fusion Testing Results

### Fusion vs Individual Modality Comparison

To validate the effectiveness of the multimodal fusion approach, comprehensive testing was conducted combining 2-3 input modalities and comparing fusion results against individual modality predictions.

**Testing Methodology**:
- 8 test cases covering various scenarios (all phishing, all legitimate, mixed signals)
- Combinations of URL, Email, and SMS modalities
- Comparison metrics: agreement rate, majority matching, probability analysis

**Test Results Summary**:
- **Total Test Cases**: 8
- **Fusion Matches Majority**: 7/8 (87.50%)
- **Average Agreement Rate**: 66.67%

**Detailed Test Cases**:

1. **Phishing URL + Phishing Email**
   - URL: Phishing (0.940), Email: Phishing (0.886)
   - Fusion: Phishing (0.917)
   - Agreement: 100%, Matches Majority: True
   - **Analysis**: Fusion successfully combines strong phishing signals from both modalities

2. **Legitimate URL + Legitimate Email**
   - URL: Legitimate (0.018), Email: Phishing (0.568)
   - Fusion: Legitimate (0.247)
   - Agreement: 50%, Matches Majority: False
   - **Analysis**: Email model false positive (legitimate email flagged as phishing), fusion correctly identifies as legitimate due to strong URL signal

3. **Phishing URL + Legitimate SMS**
   - URL: Phishing (0.780), SMS: Legitimate (0.434)
   - Fusion: Phishing (0.654)
   - Agreement: 50%, Matches Majority: True
   - **Analysis**: URL phishing signal dominates, fusion correctly identifies phishing despite legitimate SMS

4. **Phishing Email + Phishing SMS**
   - Email: Phishing (0.772), SMS: Legitimate (0.496)
   - Fusion: Phishing (0.649)
   - Agreement: 50%, Matches Majority: True
   - **Analysis**: Email phishing signal dominates, fusion correctly identifies phishing despite legitimate SMS classification

5. **Mixed Signals (Phishing URL + Legitimate Email)**
   - URL: Phishing (0.620), Email: Legitimate (0.398)
   - Fusion: Phishing (0.527)
   - Agreement: 50%, Matches Majority: True
   - **Analysis**: URL phishing signal slightly stronger, fusion correctly identifies phishing

6. **Three Modalities - All Phishing**
   - URL: Phishing (0.760), Email: Phishing (0.859), SMS: Legitimate (0.384)
   - Fusion: Phishing (0.697)
   - Agreement: 66.67%, Matches Majority: True
   - **Analysis**: Strong phishing signals from URL and Email override SMS false negative, fusion correctly identifies phishing

7. **Three Modalities - All Legitimate**
   - URL: Legitimate (0.018), Email: Legitimate (0.077), SMS: Legitimate (0.147)
   - Fusion: Legitimate (0.069)
   - Agreement: 100%, Matches Majority: True
   - **Analysis**: All modalities correctly identify legitimate, fusion maintains accuracy

8. **Three Modalities - Mixed Signals**
   - URL: Legitimate (0.300), Email: Phishing (0.869), SMS: Phishing (0.614)
   - Fusion: Phishing (0.556)
   - Agreement: 66.67%, Matches Majority: True
   - **Analysis**: Email and SMS phishing signals override URL false negative, fusion correctly identifies phishing

**Key Findings**:

1. **Fusion Improves Robustness**: When individual modalities disagree (mixed signals), fusion tends to make the correct decision by weighing multiple signals
2. **Error Correction**: Fusion can correct individual model errors (e.g., Test Case 2 where email model had false positive)
3. **Signal Dominance**: Strong signals from one modality can override weaker conflicting signals (security-first approach)
4. **Consensus Agreement**: When modalities agree (100% agreement), fusion maintains accuracy
5. **Weighted Fusion Benefits**: The weighted fusion approach (URL: 0.35, Email: 0.25, SMS: 0.20) effectively prioritizes more reliable modalities

**Fusion Weights Rationale**:
- **URL (0.35)**: Highest weight because URL analysis examines actual domain content, most reliable for phishing detection
- **Email (0.25)**: Second highest weight due to rich textual patterns and proven effectiveness
- **SMS (0.20)**: Lower weight due to shorter text length and higher variability
- **QR (0.15)**: Lower weight because visual patterns are less reliable than content analysis
- **Deepfake (0.05)**: Lowest weight as it's a specialized attack vector with different detection characteristics

**Conclusion**: The fusion approach demonstrates 87.50% accuracy in matching majority decisions and 66.67% average agreement with individual modalities. The fusion mechanism effectively combines multiple signals to improve overall detection robustness and can correct individual model errors, validating the multimodal approach for phishing detection.

---

## API Response Structure and Research-Focused Debug Information

### Research-Focused API Design Rationale

Since this system is designed for research purposes, all API endpoints return comprehensive debug information alongside core detection results. This design decision was made to support:

1. **Model Explainability**: Researchers can understand why specific predictions were made
2. **Performance Analysis**: Detailed timing and model parameters enable performance optimization studies
3. **Reproducibility**: Complete model information ensures research reproducibility
4. **Decision Transparency**: Fusion weights and individual probabilities reveal multimodal decision logic
5. **Model Comparison**: Detailed parameters allow comparison with alternative approaches

### Standard API Response Structure

All detection endpoints follow this response format:

```json
{
  "source": "modality_type",
  "label": "Phishing/Legitimate",
  "confidence": 0.0-1.0,
  "phishing_probability": 0.0-1.0,
  "debug": {
    "response_time_ms": float,
    "model_type": "string",
    "decision_threshold": float,
    // ... additional modality-specific debug info
  }
}
```

### Debug Information by Modality

#### URL Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **model_type**: "XGBoost" or "Heuristic" (fallback)
- **decision_threshold**: 0.5 (standard classification threshold)
- **original_url**: Input URL before processing
- **url_redirected**: Boolean indicating if URL was redirected
- **feature_count**: 63 (number of URL features extracted)
- **model_loaded**: Boolean indicating if ML model is available
- **model_parameters**: XGBoost hyperparameters (n_estimators, max_depth, learning_rate)
- **fallback_mode**: "Heuristic analysis" if ML model unavailable

**Rationale**: URL detection combines ML model predictions with heuristic boosts. Debug information reveals whether the ML model was used, which features were extracted, and if URL redirection occurred, all critical for understanding detection behavior.

#### Email Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **model_type**: "Logistic Regression with TF-IDF"
- **decision_threshold**: 0.5
- **text_length**: Character count of input email
- **feature_extraction**: "TF-IDF"
- **max_features**: 10000 (maximum vocabulary size)
- **ngram_range**: "(1, 2)" (unigrams and bigrams)
- **model_parameters**: Logistic Regression hyperparameters (max_iter, class_weight)
- **model_loaded**: Boolean indicating if model is available

**Rationale**: Email detection uses text classification with TF-IDF feature extraction. Debug information reveals text processing parameters, vocabulary size, and n-gram configuration, enabling researchers to understand feature engineering decisions.

#### SMS Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **model_type**: "Logistic Regression with TF-IDF (Balanced)"
- **decision_threshold**: 0.5
- **text_length**: Character count of input SMS
- **feature_extraction**: "TF-IDF"
- **max_features**: 8000 (smaller vocabulary for shorter texts)
- **ngram_range**: "(1, 2)"
- **class_weighting**: "balanced" (handles class imbalance)
- **model_parameters**: Logistic Regression hyperparameters
- **model_loaded**: Boolean indicating if model is available

**Rationale**: SMS detection uses similar text classification as email but with balanced class weighting due to different data distribution. Debug information highlights the class imbalance handling strategy.

#### QR Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **file_size_bytes**: Size of uploaded QR image
- **model_type**: "XGBoost with Multimodal Features"
- **decoded_success**: Boolean indicating if QR was successfully decoded
- **url_probability**: URL model probability (if QR contains URL)
- **qr_model_probability**: QR visual model probability
- **fused_probability**: Combined probability from URL + QR models
- **decision_threshold**: 0.5
- **fusion_weights**: URL: 0.7, QR: 0.3 (URL dominates for security)
- **model_status**: Which model(s) were used (url_qr_fusion, url_model_only, qr_model_only)

**Rationale**: QR detection uses multimodal fusion combining visual QR analysis with URL content analysis. Debug information reveals the fusion strategy, individual model contributions, and why specific weights were chosen (URL dominance for security).

#### Deepfake Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **model_type**: "EfficientNet-B0 CNN"
- **decision_threshold**: 0.44 (optimized threshold from training)
- **file_size_bytes**: Size of uploaded media file
- **file_name**: Original filename
- **frame_processing**: "Frame-level classification"
- **image_size**: 160 (reduced from standard 224 for efficiency)
- **frames_per_video**: 4 (number of frames extracted per video)
- **transfer_learning**: "ImageNet pretrained weights"
- **fine_tuning**: "Last 2 feature blocks unfrozen"

**Rationale**: Deepfake detection uses transfer learning with EfficientNet-B0. Debug information reveals the optimization strategy (reduced image size, limited frames), transfer learning approach, and fine-tuning strategy, all critical for understanding the trade-off between accuracy and efficiency.

#### Voice Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **model_type**: "Random Forest Classifier"
- **decision_threshold**: 0.5
- **file_size_bytes**: Size of uploaded audio file
- **file_name**: Original filename
- **feature_extraction**: "Hand-crafted audio features (MFCC, spectral)"
- **n_estimators**: 100 (number of trees in Random Forest)
- **class_weighting**: "balanced_subsample"
- **parallel_processing**: "All CPU cores"

**Rationale**: Voice detection uses traditional machine learning with hand-crafted audio features rather than deep learning. Debug information reveals the feature extraction approach (MFCC, spectral features), model configuration, and parallel processing strategy.

#### Fusion Detection Debug Information
- **response_time_ms**: Processing time in milliseconds
- **fusion_method**: "Weighted probability fusion"
- **decision_threshold**: 0.5
- **modalities_used**: List of modalities included in fusion
- **fusion_weights**: URL: 0.35, Email: 0.25, SMS: 0.20, QR: 0.15, Deepfake: 0.05
- **individual_probabilities**: Probability from each modality
- **weighted_calculation**: "Sum of (probability * weight) / total_weight"

**Rationale**: Fusion combines multiple modalities using weighted probability fusion. Debug information reveals the fusion strategy, weight rationale (URL highest due to content analysis reliability), and individual modality contributions, enabling researchers to understand multimodal decision logic.

### Why This Approach for Research

**Comprehensive Debug Information Benefits**:

1. **Model Explainability**: Researchers can trace exactly how each prediction was made
2. **Parameter Transparency**: All model hyperparameters and configuration choices are exposed
3. **Performance Analysis**: Response times and processing details enable optimization studies
4. **Decision Logic**: Fusion weights and individual probabilities reveal multimodal reasoning
5. **Reproducibility**: Complete information ensures research can be reproduced
6. **Comparative Analysis**: Detailed parameters enable comparison with alternative approaches
7. **Debugging**: Comprehensive information aids in identifying and fixing issues
8. **Documentation**: Debug information serves as self-documenting API behavior

**Alternative Approaches Considered and Rejected**:

1. **Minimal API Responses**: Only return label and probability
   - **Rejected**: Insufficient for research analysis and model explainability

2. **Optional Debug Parameter**: Include debug info only when requested
   - **Rejected**: Adds complexity, research use case always needs debug info

3. **Separate Debug Endpoint**: Separate endpoint for detailed information
   - **Rejected**: Requires additional API calls, less convenient for research

4. **Conditional Debug Info**: Only include debug for certain modalities
   - **Rejected**: Inconsistent API behavior, harder to use programmatically

**Selected Approach**: Always include comprehensive debug information in all API responses. This provides maximum research utility while maintaining consistent API behavior across all modalities.

---

## Conclusion

### Model Selection Summary

| Modality | Model | Accuracy | Rationale |
|----------|-------|----------|-----------|
| URL | XGBoost | 99.74% | Best performance on tabular URL features, interpretable, efficient |
| Email | Logistic Regression + TF-IDF | 99.30% | Strong baseline, interpretable, fast training |
| SMS | Logistic Regression + TF-IDF (Balanced) | 97.85% | Handles short texts, class imbalance, proven effectiveness |
| QR | XGBoost (Multimodal) | 90.65% | Combines visual + URL features, handles missing data |
| Deepfake | EfficientNet-B0 | 83.50% | Efficient CNN, transfer learning, laptop-friendly |
| Voice | Random Forest | 98.51% | Robust to feature scaling, interpretable, excellent on audio features |

### Key Design Principles

1. **Modality-Specific Optimization**: Each modality uses model architecture best suited to its data characteristics
2. **Interpretability Priority**: Models chosen for explainability (feature importance, coefficients) where possible
3. **Computational Efficiency**: Models selected for real-time inference capability
4. **Transfer Learning**: Leveraging pretrained models (ImageNet) where applicable to improve performance with limited data
5. **Regularization**: Strong regularization (early stopping, weight decay, class weighting) to prevent overfitting on limited datasets
6. **Fusion Strategy**: Weighted fusion with modality-specific weights based on reliability and attack scenario

### Future Improvements

1. **Deepfake Dataset Expansion**: Increase from 2,000 to 10,000 videos to improve accuracy to 90%+
2. **Advanced Fusion**: Implement neural network-based fusion for automatic weight learning
3. **Real-Time Adaptation**: Online learning to adapt to new phishing patterns
4. **Explainability Enhancements**: SHAP values, attention mechanisms for deeper interpretability
5. **Multilingual Support**: Extend models to handle non-English text and URLs

---

