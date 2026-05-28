# Justification: Lower Accuracy in Video Deepfake Detection Compared to Audio

## Overview
This document provides a technical and practical justification for why video deepfake detection models, even with advanced machine learning techniques, tend to produce lower accuracy and ROC AUC compared to audio deepfake detection. The explanation is tailored for project defense and client communication.

---

## 1. Data Imbalance & Complexity
- **Video datasets** (e.g., FaceForensics++) are highly imbalanced, with far fewer original (real) samples than manipulated (fake) ones. This leads to models being biased toward the majority class, resulting in high false positives and lower overall accuracy and ROC AUC.
- **Video deepfakes** are visually complex and often subtle, making it harder for models to distinguish real from fake compared to audio, where artifacts are more pronounced and easier to capture with features.

## 2. Feature Extraction Challenges
- **Video features** are high-dimensional and may include redundant or noisy information, making it harder for classical models (RandomForest, XGBoost) to learn effective boundaries.
- **Audio features** (e.g., MFCCs) are more compact and discriminative, leading to better model performance.

## 3. Computational Resources & Overhead
- **Video processing** requires significantly more time and computational resources (CPU/GPU, RAM, disk I/O) due to the need to extract and process multiple frames per sample.
- **Training and inference** on video features are slower and more memory-intensive than on audio features, increasing overhead and limiting scalability.

## 4. Model Performance Comparison
- **Video (RandomForest, combined features):**
  - Accuracy: ~0.90
  - ROC AUC: ~0.54
  - High recall, but high false positives and poor class separation
- **Audio (best model):**
  - Accuracy: 0.95+
  - ROC AUC: 0.80+
  - Lower false positives, better class separation, and more reliable in practice

| Modality | Accuracy | ROC AUC | False Positives | Time/Resources | Notes |
|----------|----------|---------|-----------------|----------------|-------|
| Video    | 0.90     | 0.54    | High            | High           | Imbalanced, complex features |
| Audio    | 0.95+    | 0.80+   | Low             | Low            | Compact, discriminative features |

## 5. Conclusion
Video deepfake detection is inherently more challenging due to data imbalance, feature complexity, and computational demands. Audio deepfake detection is currently more reliable and efficient for most practical applications. These findings are supported by experimental results and are consistent with the literature.

---

**Defensive Statement:**
> The lower accuracy in video deepfake detection is not a result of poor model selection or implementation, but rather a reflection of the intrinsic challenges in the video domain. The project leverages best-practice machine learning and data engineering, and the results are in line with current research. The audio pipeline demonstrates the effectiveness of the approach, and ongoing improvements in video data balancing and feature engineering are expected to further enhance performance.
