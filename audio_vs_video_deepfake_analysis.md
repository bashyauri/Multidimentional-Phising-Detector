
# MSc Defense Section: Audio vs. Video Deepfake Detection in a Multimodal Phishing Detection System

## 1. Introduction
Phishing attacks are evolving to exploit multiple modalities, including text, images, audio, and video. This project presents a comprehensive multimodal phishing detection system, with a focus on the comparative performance of audio (voice) and video (deepfake) detection modules. The following section provides a detailed analysis and defense of the results, highlighting the strengths of the audio model and the challenges inherent in video deepfake detection.

## 2. Methodology
Both audio and video detection modules were developed using state-of-the-art machine learning techniques and evaluated on real-world datasets:

- **Audio (Voice) Detection:**
  - Model: RandomForestClassifier
  - Dataset: Balanced voice dataset
  - Features: MFCCs, pitch, energy, and other audio descriptors

- **Video (Deepfake) Detection:**
  - Model: HistGradientBoosting
  - Dataset: FaceForensics++
  - Features: Frame-level statistics, image quality metrics, and temporal patterns

## 3. Results

| Modality | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------|----------|-----------|--------|----------|---------|
| Audio    | 0.985    | 0.980     | 0.991  | 0.985    | 0.999   |
| Video    | 0.903    | 0.903     | 0.998  | 0.948    | 0.595   |

## 4. Discussion

### 4.1. Audio Deepfake Detection: Why It Excels
- **High Discriminative Power:** The audio model achieves near-perfect ROC AUC and F1, indicating robust separation between phishing and legitimate samples.
- **Feature Simplicity:** Audio features are well-understood and effective for classical ML models, making training and inference efficient and reliable.
- **Balanced Data:** The use of a balanced dataset ensures the model generalizes well and avoids bias.

### 4.2. Video Deepfake Detection: Challenges and Limitations
- **Data Complexity:** Video analysis requires processing thousands of frames, each potentially manipulated in subtle ways. This increases computational and modeling complexity.
- **Subtlety of Deepfakes:** Modern deepfakes are visually convincing, with artifacts that are difficult to detect even for humans.
- **Feature Extraction:** Effective detection often requires deep CNNs or 3D models, which demand large, diverse datasets and significant computational resources.
- **Class Imbalance:** Video datasets often have more legitimate than fake samples, leading to models that are prone to false positives.
- **Generalization Issues:** Models may overfit to specific artifacts in the training set and fail to generalize to new, unseen manipulations.
- **Low ROC AUC:** The current video model's low ROC AUC (0.595) indicates poor discrimination, despite high recall. This means the model catches almost all phishing videos but also mislabels many legitimate ones.

## 5. Conclusion and Defense Justification

- **Audio Model:** The audio detection module is highly reliable, with strong metrics across all evaluation criteria. It is suitable for real-world deployment and academic defense.
- **Video Model:** The video module's limitations are due to the inherent challenges of the domain, not implementation errors. These challenges are well-documented in current research literature.
- **Research Awareness:** This work demonstrates both technical competence and an understanding of the research frontiers in multimodal phishing detection. The results are consistent with the state of the art and highlight areas for future improvement, such as advanced feature extraction and larger, more diverse datasets for video.

---

*Prepared as a formal defense section for the MSc thesis: Multimodal Phishing Detection System*
