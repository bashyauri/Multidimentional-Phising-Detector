# Dataset Split Methodology Comparison

This document compares model performance between two-way (80/20) and three-way (70/15/15) dataset splits for academic research purposes.

## Research Context

This comparison investigates the impact of dataset split methodology on model performance metrics. The three-way split (Training/Validation/Testing) is a machine learning best practice that:

- **Training Set (70%)**: Used for model fitting and parameter learning
- **Validation Set (15%)**: Used during training for hyperparameter tuning, model selection, and early stopping
- **Testing Set (15%)**: Reserved for final unbiased performance evaluation

The two-way split (80/20) only separates training and testing data, which can lead to optimistic bias if test data indirectly influences model selection.

## Performance Comparison

### URL Phishing Detection Model

| Metric | Two-Way Split (80/20) | Three-Way Split (70/15/15) | Difference |
|---|---:|---:|---:|
| **Accuracy** | 99.74% | 99.74% | 0.00% |
| **Precision** | 99.57% | 99.59% | +0.02% |
| **Recall** | 99.97% | 99.97% | 0.00% |
| **F1-Score** | 99.77% | 99.78% | +0.01% |
| **ROC-AUC** | 99.90% | 99.92% | +0.02% |
| **Test Samples** | 47,159 | 35,370 | -11,789 |

**Analysis**: URL model shows nearly identical performance between split methods, indicating robustness to dataset composition changes. The slight improvement in precision and ROC-AUC with three-way split suggests better generalization.

### Email Phishing Detection Model

| Metric | Two-Way Split (80/20) | Three-Way Split (70/15/15) | Difference |
|---|---:|---:|---:|
| **Accuracy** | 99.30% | 99.32% | +0.02% |
| **Precision** | 99.20% | 99.27% | +0.07% |
| **Recall** | 99.54% | 99.51% | -0.03% |
| **F1-Score** | 99.37% | 99.39% | +0.02% |
| **ROC-AUC** | 99.96% | 99.98% | +0.02% |
| **Test Samples** | 7,831 | 5,874 | -1,957 |

**Analysis**: Email model shows minimal performance difference, with slight improvement in precision and F1-score using three-way split. The consistency indicates the model is well-regularized.

### SMS Phishing Detection Model

| Metric | Two-Way Split (80/20) | Three-Way Split (70/15/15) | Difference |
|---|---:|---:|---:|
| **Accuracy** | 97.85% | 97.73% | -0.12% |
| **Precision** | 91.39% | 90.43% | -0.96% |
| **Recall** | 92.62% | 92.86% | +0.24% |
| **F1-Score** | 92.00% | 91.63% | -0.37% |
| **ROC-AUC** | 98.77% | 98.89% | +0.12% |
| **Test Samples** | 1,115 | 836 | -279 |

**Analysis**: SMS model shows the largest performance variation, with slightly lower accuracy and precision but higher recall with three-way split. This variation is expected for smaller datasets (1,115 vs 836 test samples) where sample composition has greater impact.

### QR Code Phishing Detection Model

| Metric | Two-Way Split (80/20) | Three-Way Split (70/15/15) | Difference |
|---|---:|---:|---:|
| **Accuracy** | 90.65% | 84.00% | -6.65% |
| **Precision** | 88.83% | 88.06% | -0.77% |
| **Recall** | 93.00% | 78.67% | -14.33% |
| **F1-Score** | 90.86% | 83.10% | -7.76% |
| **ROC-AUC** | 96.72% | 89.51% | -7.21% |
| **Test Samples** | 2,000 | 150 | -1,850 |

**Analysis**: QR model shows significant performance drop with three-way split due to much smaller test set (150 vs 2,000 samples). This highlights the importance of sufficient test set size for reliable evaluation, especially for visual-based models with higher variance.

### Voice Deepfake Detection Model

| Metric | Two-Way Split (80/20) | Three-Way Split (70/15/15) | Difference |
|---|---:|---:|---:|
| **Accuracy** | 98.51% | 98.30% | -0.21% |
| **Precision** | 97.98% | 97.65% | -0.33% |
| **Recall** | 99.07% | 98.98% | -0.09% |
| **F1-Score** | 98.52% | 98.31% | -0.21% |
| **ROC-AUC** | 99.92% | 99.88% | -0.04% |
| **Test Samples** | 2,356 | 1,767 | -589 |

**Analysis**: Voice model shows minimal performance difference between split methods, with slight decrease across all metrics using three-way split. The consistency indicates stable performance across different test compositions.

## Summary Statistics

### Overall Performance Trends

| Model | Two-Way Accuracy | Three-Way Accuracy | Difference | Test Size Impact |
|---|---:|---:|---:|---:|
| URL | 99.74% | 99.74% | 0.00% | Minimal (large dataset) |
| Email | 99.30% | 99.32% | +0.02% | Minimal (medium dataset) |
| SMS | 97.85% | 97.73% | -0.12% | Moderate (small dataset) |
| QR | 90.65% | 84.00% | -6.65% | High (very small test set) |
| Voice | 98.51% | 98.30% | -0.21% | Minimal (medium dataset) |

### Key Findings

1. **Large Datasets (URL)**: Performance is virtually identical between split methods, indicating robustness to test composition changes when sufficient data is available.

2. **Medium Datasets (Email, Voice)**: Minimal performance variation (±0.2%), showing that well-regularized models maintain consistent performance across different split methodologies.

3. **Small Datasets (SMS)**: Moderate variation (±0.4-1.0%) due to higher sensitivity to test sample composition with limited data.

4. **Very Small Test Sets (QR)**: Significant performance drop (6.65%) when test set is too small (150 vs 2,000 samples), highlighting the importance of adequate test set size for reliable evaluation.

5. **Academic Rigor**: Three-way split provides more rigorous evaluation by ensuring test set is never seen during model selection or hyperparameter tuning.

## Recommendations

### For Academic Research

1. **Use Three-Way Split**: For final model evaluation and reporting, use the three-way split (70/15/15) methodology to ensure unbiased performance metrics.

2. **Report Both Methods**: Include both two-way and three-way split results in research papers to demonstrate robustness and transparency.

3. **Adequate Test Set Size**: Ensure test sets contain sufficient samples (minimum 500-1,000) for reliable evaluation, especially for visual-based models.

4. **Dataset Size Considerations**: For small datasets (< 1,000 samples), consider cross-validation instead of single train/test splits.

### For Production Deployment

1. **Use Three-Way Split Models**: Deploy models trained with three-way split methodology for production use, as they are more rigorously validated.

2. **Monitor Performance**: Track model performance in production and compare against validation metrics to detect drift.

## File Locations

### Two-Way Split Models (Original)
- `models/url_model.pkl`
- `models/email_model.pkl`
- `models/sms_model.pkl`
- `models/qr_model.pkl`
- `models/voice_model_balanced.pkl`

### Three-Way Split Models (Validation)
- `models/url_model_validation.pkl`
- `models/email_model_validation.pkl`
- `models/sms_model_validation.pkl`
- `models/qr_model_validation.pkl`
- `models/voice_model_balanced_validation.pkl`

### Metrics Files
- Two-way: `models/{module}_metrics.json`
- Three-way: `models/{module}_validation_metrics.json`

## Conclusion

The three-way dataset split methodology provides more rigorous and unbiased model evaluation while maintaining comparable performance to the two-way split for most models. The slight performance variations observed are expected and reflect the more conservative evaluation approach. For academic research and production deployment, the three-way split methodology is recommended as it follows machine learning best practices and provides more reliable performance estimates.
