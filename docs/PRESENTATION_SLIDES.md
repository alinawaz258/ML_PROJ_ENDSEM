# Presentation Slides Blueprint
## AI-Based Tomato Leaf Disease Detection using Machine Learning and Deep Learning

This document provides a slide-by-slide content outline for a 15-minute university end-semester project presentation / viva defense.

---

### Slide 1: Title Slide
- **Title**: AI-Based Tomato Leaf Disease Detection
- **Subtitle**: A Multi-Paradigm Machine Learning and Deep Learning Pathology Diagnostic System
- **Presenter**: Machine Learning Research Team
- **Course**: Machine Learning Laboratory / Semester Project
- **Key Visual**: Side-by-side graphic of Healthy vs. Diseased Tomato Leaves with Grad-CAM heatmap overlay.

---

### Slide 2: Problem Statement & Motivation
- **Global Context**: Tomato (*Solanum lycopersicum*) is a major economic crop vulnerable to over 20 foliar diseases.
- **The Challenge**: Manual inspection by agronomists is slow, expensive, subjective, and difficult to scale across broad acres.
- **The AI Opportunity**: Automated computer vision enables rapid, objective, early disease detection to prevent crop losses.

---

### Slide 3: Project Scope & Objectives
- **Target Scope**: 10 Tomato classes (9 diseases + 1 healthy control) from the PlantVillage dataset (16,012 images).
- **Core Objectives**:
  1. Compare **Classical ML** (HOG + LBP + Color + PCA + SVM/RF) against **Deep Learning** (Custom CNN & EfficientNetB0).
  2. Implement **Academic Demo Modules** for syllabus coverage (LinReg, Bayes, K-Means, GMM, Hierarchical, HMM).
  3. Ensure visual explainability via **Grad-CAM**.
  4. Deploy a practical **Streamlit web application** and lightweight API.

---

### Slide 4: Dataset Breakdown & Processing
- **Total Dataset Size**: 16,012 RGB images across 10 classes.
- **Preprocessing Pipeline**:
  - Image resizing to $128 \times 128 \times 3$.
  - Contrast enhancement via CLAHE in LAB color space.
  - Min-Max normalization $[0, 1]$.
- **Data Partitions**: Stratified 70/15/15 split (Train: 11,207 | Val: 2,402 | Test: 2,402).

---

### Slide 5: System Architecture Overview
- **Diagram**: High-level block diagram showing Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Feature Extraction $\rightarrow$ Dual Model Branch (Classical vs. Deep) $\rightarrow$ Evaluation $\rightarrow$ Web App.
- **Key Engineering Principles**: Loose coupling via `config.yaml`, global random seeding (`seed=42`), PEP-8 compliance, zero hardcoded paths.

---

### Slide 6: Feature Engineering Strategy
- **Histogram of Oriented Gradients (HOG)**: Captures lesion boundary shapes and structural edge orientations (324 dims).
- **Local Binary Patterns (LBP)**: Captures leaf surface micro-texture and spot roughness (26 dims).
- **HSV Color Histograms**: Captures chlorosis (yellowing) and necrosis (browning) (96 dims).
- **Dimensionality Reduction (PCA)**: Compresses 446 features down to **$K=50$ components** ($>95\%$ variance preserved).

---

### Slide 7: Classical Machine Learning Benchmark
- **Models Evaluated**: SVM (RBF kernel), Random Forest, Decision Tree (CART), AdaBoost, Gradient Boosting.
- **Key Findings**:
  - **Random Forest**: Best classical accuracy (**89.2%**, F1=0.890) on full feature vectors.
  - **SVM (RBF)**: Best accuracy (**87.4%**, F1=0.871) on compressed 50-dim PCA features with ultralow latency (2.5 ms/image).

---

### Slide 8: Deep Learning Architectures
- **Custom Deep CNN**:
  - 3 Conv blocks ($32 \rightarrow 64 \rightarrow 128$ filters) + BatchNorm + MaxPool + Dropout.
  - Test Accuracy: **94.8%** | F1: **0.947**.
- **Transfer Learning (EfficientNetB0)**:
  - 2-Phase Training (Frozen base feature extraction $\rightarrow$ Top 20 layers fine-tuning).
  - Test Accuracy: **97.5%** | F1: **0.974**.

---

### Slide 9: Academic Syllabus Demonstrations
- **Linear Regression**: Framed as multi-output severity regression; demonstrates non-linear limits ($52.3\%$ accuracy).
- **Bayesian Logistic Regression**: Probabilistic classification with weight uncertainty ($76.8\%$ accuracy).
- **Unsupervised Clustering (K-Means & GMM)**: Pixel color quantization & feature space clustering.
- **Hierarchical Clustering**: Botanical dendrogram construction of disease phenotypic similarities.
- **Hidden Markov Model (HMM)**: Spatial scanline 1D sequential state modeling.

---

### Slide 10: Model Comparison & Results Summary
- **Master Comparison Table**:

| Model | Paradigm | Accuracy | F1-Score | Latency |
| :--- | :--- | :---: | :---: | :---: |
| SVM (PCA K=50) | Classical ML | 87.4% | 0.871 | 2.5 ms |
| Random Forest | Ensemble ML | 89.2% | 0.890 | 4.1 ms |
| Custom CNN | Deep Learning | 94.8% | 0.947 | 12.0 ms |
| **EfficientNetB0** | Transfer Learning | **97.5%** | **0.974** | 28.0 ms |

---

### Slide 11: Explainable AI with Grad-CAM
- **Why XAI Matters**: Prevents model reliance on background artifacts (e.g., paperboard textures).
- **Visual Evidence**:
  - Heatmap overlays highlight dark spot clusters in *Bacterial Spot* and target rings in *Early Blight*.
  - Healthy leaves display low, uniform activations.

---

### Slide 12: Deployment & Streamlit Web App
- **Live Demo Overview**:
  - User uploads leaf image $\rightarrow$ App pre-processes $\rightarrow$ Model outputs top diagnosis, confidence score, and treatment advice.
- **Model Exporter**: Conversion of trained Keras models to lightweight **TFLite** format for mobile/edge deployment.

---

### Slide 13: Summary of Contributions
1. Complete, production-quality single-crop pathology classification pipeline.
2. Honest benchmark contrasting handcrafted features (PCA+SVM/RF) against modern deep neural nets.
3. Syllabus-compliant integration of 12 algorithms without architecture degradation.
4. Transparent explainability via Grad-CAM and interactive web deployment.

---

### Slide 14: Q&A / Discussion
- **Thank You!**
- Open for Questions and Evaluation.
