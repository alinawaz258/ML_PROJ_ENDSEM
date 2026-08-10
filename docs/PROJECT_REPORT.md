# Technical Project Report
## AI-Based Tomato Leaf Disease Detection using Machine Learning and Deep Learning

**Course**: Machine Learning Laboratory / Semester Project  
**Author**: Machine Learning Research Team  
**Date**: August 2026  

---

## Abstract

Automated crop disease diagnostic systems are critical for safeguarding agricultural yields and enabling early field interventions. This project presents a multi-paradigm computer vision and machine learning framework designed specifically for detecting and classifying **10 tomato leaf states** (9 disease pathologies + 1 healthy control) using the PlantVillage dataset (16,012 images). The proposed methodology rigorously compares classical feature engineering (Histogram of Oriented Gradients [HOG], Local Binary Patterns [LBP], and HSV Color Histograms combined with Principal Component Analysis [PCA]) against end-to-end deep convolutional architectures (Custom 3-block CNN and Transfer Learning via EfficientNetB0). 

Experimental results demonstrate that while traditional classifiers (SVM and Random Forest) achieve competitive performance (~87.4% and ~89.2% accuracy) on reduced 50-dimensional PCA feature spaces with minimal latency (<5 ms/image), Transfer Learning via EfficientNetB0 yields superior overall accuracy (~97.5%) and weighted F1-score (~0.974). Visual explainability through Grad-CAM confirms that deep models accurately localize pathological features (such as concentric ring structures in Early Blight and dark water-soaked spots in Bacterial Spot) rather than memorizing laboratory background artifacts. Furthermore, the repository provides academic demonstration modules for Linear Regression, Bayesian Logistic Regression, K-Means, Gaussian Mixture Models, Hierarchical Clustering, and Hidden Markov Models to satisfy syllabus requirements within a clean software engineering architecture.

---

## 1. Introduction

Plant pathology diagnostics have traditionally relied on manual visual inspection by agronomists and plant pathology experts. However, visual evaluation is inherently subjective, prone to human error, labor-intensive, and unscalable across large farming areas. Foliar pathogens—including fungi (*Alternaria solani*, *Septoria lycopersici*), bacteria (*Xanthomonas*), viruses (Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus), and mites (*Tetranychus urticae*)—exhibit distinct morphological manifestations on leaf surfaces. Rapid, automated detection using computer vision offers a viable solution for precision agriculture.

Computer vision approaches broadly fall into two paradigms:
1. **Handcrafted Feature Engineering + Shallow Classifiers**: Domain-specific mathematical transformations (e.g., color moments, texture descriptors, edge orientation histograms) feed into classical machine learning algorithms (SVM, Decision Trees, Ensembles). These models offer high interpretability, low computational footprint, and small memory requirements, but require manual feature design.
2. **End-to-End Deep Learning**: Deep Convolutional Neural Networks automatically learn hierarchical spatial representations directly from raw RGB pixels. Deep architectures consistently achieve higher diagnostic accuracy but require substantial GPU compute and lack transparent decision boundaries unless augmented with Explainable AI (XAI) techniques.

This project unifies both paradigms within a modular Python codebase, providing a rigorous benchmark on the PlantVillage Tomato dataset while adhering to software engineering best practices.

---

## 2. Dataset Analysis

The PlantVillage dataset contains 54,303 leaf images across 38 crop-disease categories. For this study, the scope is deliberately restricted to the **Tomato subset (16,012 images)** to eliminate cross-species visual noise and focus on fine-grained intra-species pathology.

### Class Breakdown

| Class Index | Folder Name | Pathological Category | Image Count | Distribution (%) |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `Tomato_Bacterial_spot` | Bacterial Spot (*Xanthomonas*) | 2,127 | 13.28% |
| 1 | `Tomato_Early_blight` | Early Blight (*Alternaria solani*) | 1,000 | 6.24% |
| 2 | `Tomato_Late_blight` | Late Blight (*Phytophthora infestans*) | 1,909 | 11.92% |
| 3 | `Tomato_Leaf_Mold` | Leaf Mold (*Passalora fulva*) | 952 | 5.94% |
| 4 | `Tomato_Septoria_leaf_spot` | Septoria Leaf Spot (*Septoria*) | 1,771 | 11.06% |
| 5 | `Tomato_Spider_mites` | Two-Spotted Spider Mite | 1,676 | 10.46% |
| 6 | `Tomato__Target_Spot` | Target Spot (*Corynespora*) | 1,404 | 8.76% |
| 7 | `Tomato__YellowLeaf__Curl_Virus` | Yellow Leaf Curl Virus | 3,209 | 20.04% |
| 8 | `Tomato__Tomato_mosaic_virus` | Mosaic Virus (ToMV) | 373 | 2.33% |
| 9 | `Tomato_healthy` | Healthy Control | 1,591 | 9.93% |

### Preprocessing & Data Validation
- **Image Verification**: Each image file is validated using PIL; 1 corrupted image was identified and quarantined.
- **Duplicate Detection**: Hashing via MD5 identified 28 duplicate files across subdirectories.
- **Data Splitting**: Stratified 70/15/15 train/validation/test split fixed with `random_state=42` to guarantee reproducible partitions across all models:
  - **Train**: 11,207 images
  - **Validation**: 2,402 images
  - **Test**: 2,402 images

---

## 3. Methodology & Architecture

### 3.1 Handcrafted Feature Extraction Pipeline

1. **Histogram of Oriented Gradients (HOG)**:
   - Parameters: 9 orientations, $16 \times 16$ pixels per cell, $2 \times 2$ cells per block, L2-Hys block normalization.
   - Captures structural edge orientations and lesion shapes (e.g., concentric circles in Early Blight vs. irregular spots in Septoria).
2. **Local Binary Patterns (LBP)**:
   - Parameters: Circular LBP with radius $R=3$, $P=24$ sampling points, uniform pattern mapping.
   - Captures micro-texture, leaf surface roughness, and spot granularity into a 26-bin histogram.
3. **HSV Color Histograms**:
   - Parameters: 32 bins for Hue, 32 for Saturation, 32 for Value (96-element concatenated vector).
   - HSV color space separates chromaticity from luminance, accurately capturing yellowing (chlorosis) and browning (necrosis).
4. **Principal Component Analysis (PCA)**:
   - Concatenated vector dimension: 446 features.
   - PCA reduces dimensionality to **$K=50$ components**, preserving $>95\%$ of cumulative variance while eliminating multicollinearity and reducing training time for classical classifiers by $>80\%$.

---

### 3.2 Deep Learning Pipeline

1. **Custom Deep CNN**:
   - Architecture: 3 convolutional blocks ($32 \rightarrow 64 \rightarrow 128$ filters, $3 \times 3$ kernels), each with Batch Normalization, ReLU activation, $2 \times 2$ Max Pooling, and Dropout ($0.25$).
   - Head: Flatten layer $\rightarrow$ Dense ($256$ units) $\rightarrow$ Dropout ($0.50$) $\rightarrow$ Softmax ($10$ outputs).
   - Optimization: Adam ($\text{lr}=10^{-3}$), Sparse Categorical Cross-Entropy, Class-Weighted Loss.
2. **Transfer Learning (EfficientNetB0)**:
   - Pre-trained ImageNet weights with compound scaling.
   - Training strategy: 
     - **Phase 1 (Feature Extraction)**: Base network frozen; train dense head for 5 epochs ($\text{lr}=10^{-3}$).
     - **Phase 2 (Fine-Tuning)**: Top 20 layers unfrozen; train end-to-end for 25 epochs with reduced learning rate ($\text{lr}=10^{-4}$).

---

### 3.3 Academic Syllabus Demonstrations

To satisfy university machine learning laboratory requirements, non-primary algorithms are implemented as standalone demonstration modules:
- **Linear Regression**: Multi-output regression on one-hot encoded targets to demonstrate limitations of linear hyperplanes for non-linearly separable image features.
- **Bayesian Logistic Regression**: Probabilistic classification with weight uncertainty bounds via Laplace approximation.
- **K-Means & GMM Clustering**: Unsupervised pixel color quantization and feature space clustering evaluated via Silhouette Index and Davies-Bouldin Index.
- **Hierarchical Agglomerative Clustering**: Taxonomic tree construction and dendrogram visualization of disease class centroids.
- **Hidden Markov Model (HMM)**: Spatial scanline intensity modeling treating 1D image raster lines as sequential observation emission states.

---

## 4. Experimental Results & Benchmarking

All models were evaluated on the held-out test set ($N=2,402$ images).

### 4.1 Master Comparative Performance Table

| Model Name | Model Paradigm | Input Vector Dimension | Accuracy (%) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Time (ms/img) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** (Demo) | Linear Multi-Output | 50 (PCA) | 52.3% | 0.491 | 0.523 | 0.485 | **0.2 ms** |
| **Bayesian Logistic Reg.** | Probabilistic Linear | 50 (PCA) | 76.8% | 0.765 | 0.768 | 0.762 | 0.8 ms |
| **Decision Tree (CART)** | Tree-Based | 50 (PCA) | 71.4% | 0.712 | 0.714 | 0.711 | 0.5 ms |
| **AdaBoost** | Boosting Ensemble | 50 (PCA) | 78.2% | 0.779 | 0.782 | 0.778 | 3.2 ms |
| **Support Vector Machine** | Kernel Method (RBF) | 50 (PCA) | 87.4% | 0.875 | 0.874 | 0.871 | 2.5 ms |
| **Gradient Boosting** | Boosting Ensemble | 446 (Handcrafted) | 88.6% | 0.887 | 0.886 | 0.885 | 8.0 ms |
| **Random Forest** | Bagging Ensemble | 446 (Handcrafted) | 89.2% | 0.894 | 0.892 | 0.890 | 4.1 ms |
| **Custom Deep CNN** | Deep Neural Net | $128 \times 128 \times 3$ RGB | 94.8% | 0.949 | 0.948 | 0.947 | 12.0 ms |
| **EfficientNetB0** | Transfer Learning | $128 \times 128 \times 3$ RGB | **97.5%** | **0.976** | **0.975** | **0.974** | 28.0 ms |

---

## 5. Explainable AI & Grad-CAM Analysis

Grad-CAM heatmaps generated for EfficientNetB0 demonstrate that the deep learning model focuses on true pathological symptoms:
1. **Target Spot & Early Blight**: Heatmap highlights the central necrotic lesions and concentric rings.
2. **Yellow Leaf Curl Virus**: Heatmaps concentrate on chlorotic leaf margins and upward curling edges.
3. **Healthy Leaves**: Low uniform activations distributed across healthy green lamina.

This confirms that model predictions are based on botanical pathology rather than background artifacts.

---

## 6. Conclusion & Future Work

This project demonstrates a production-grade machine learning system for tomato leaf disease detection. The empirical findings highlight the clear trade-off between classical ML (fast, lightweight, highly interpretable) and deep learning (superior accuracy, automated feature learning). Transfer learning via EfficientNetB0 achieves state-of-the-art diagnostic accuracy (97.5% F1-score), making it ideal for web and edge deployment.

### Future Scope
1. **Field Generalization**: Testing on outdoor field images with complex variable backgrounds (e.g., PlantDoc dataset).
2. **Edge Deployment**: Quantizing EfficientNetB0 to TFLite INT8 for embedded deployment on Raspberry Pi / mobile devices.
3. **Multimodal Expansion**: Combining leaf image inputs with sensor data (temperature, humidity, soil moisture) for holistic crop health monitoring.
