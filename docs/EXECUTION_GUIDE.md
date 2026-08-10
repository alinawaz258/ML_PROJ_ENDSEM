# Execution Guide
## AI-Based Tomato Leaf Disease Detection System

This document outlines all methods for running, training, evaluating, and deploying the Tomato Leaf Disease Detection pipeline.

---

## 1. Unified Pipeline Execution (`main.py`)

The primary entry point is `main.py` located in the project root directory.

### Run All Pipeline Phases
To execute the complete end-to-end pipeline from raw data scanning to model evaluation:

```bash
python main.py
```

### Run a Specific Phase
You can run individual phases using the `--phase` flag:

```bash
python main.py --phase <PHASE_NUMBER>
```

| Phase Number | Phase Name | Description | Main Output Files |
| :---: | :--- | :--- | :--- |
| **1** | Dataset Preparation | Scans images, detects duplicates/corruption, performs stratified split | `dataset.csv`, `train.csv`, `val.csv`, `test.csv`, `class_distribution.png` |
| **2** | Image Preprocessing | Resizes images to $128 \times 128$, normalizes intensity, saves arrays | `X_train.npy`, `X_val.npy`, `X_test.npy`, `before_after_preprocessing.png` |
| **3** | Feature Extraction | Extracts HOG, LBP, Color Histograms; fits PCA ($K=50$) | `features_train.npy`, `pca_model.pkl`, `pca_variance.png` |
| **4** | Classical ML Training | Trains SVM, Random Forest, Decision Tree, AdaBoost, Gradient Boosting | `svm_model.pkl`, `random_forest_model.pkl`, `classical_ml_results.json` |
| **5** | Academic ML Demos | Runs Linear Regression, Bayesian, K-Means, GMM, Hierarchical, HMM | `linear_regression_demo.png`, `kmeans_clusters.png`, `dendrogram.png` |
| **6** | Deep Learning Training | Trains Custom CNN and fine-tunes EfficientNetB0 | `custom_cnn_final.keras`, `efficientnet_final.keras`, `gradcam_efficientnet.png` |
| **7** | Evaluation & Comparison | Aggregates all metrics, plots ROC curves and comparison bar charts | `final_evaluation_report.json`, `model_comparison_bar_chart.png` |

---

## 2. Phase-Specific Standalone Scripts

Each phase can also be executed directly via standalone Python scripts located in `scripts/`:

```bash
# Phase 1: Dataset scanning & splitting
python scripts/run_dataset.py

# Phase 2: Preprocessing & normalization
python scripts/run_preprocessing.py

# Phase 3: Feature extraction & PCA reduction
python scripts/run_features.py

# Phase 4: Classical ML model training
python scripts/run_classical_ml.py

# Phase 5: Academic demonstration algorithms
python scripts/run_demos.py

# Phase 6: Deep Learning model training & Grad-CAM
python scripts/run_deep_learning.py

# Phase 7: Comprehensive evaluation & comparison
python scripts/run_evaluation.py
```

---

## 3. Interactive Web Application (Streamlit)

Launch the Streamlit web application to test model predictions interactively via a browser interface:

```bash
streamlit run app/streamlit_app.py
```

### Features of the Web App:
- **Model Selection**: Choose between **EfficientNetB0**, **Custom CNN**, **Support Vector Machine (SVM)**, or **Random Forest**.
- **Drag-and-Drop Upload**: Upload any leaf image (`.jpg`, `.jpeg`, `.png`).
- **Real-Time Prediction**: Displays predicted disease name and confidence percentage bar.
- **Probability Distribution**: Bar chart showing probabilities across all 10 tomato classes.
- **Pathology Advice**: Displays disease description and treatment recommendations.

---

## 4. Single-Image CLI Prediction

To make a prediction on a single image file from the command line:

```bash
python scripts/predict.py --image archive/PlantVillage/Tomato_healthy/000bf832-da49-48b7-8772-603280936980___GH_HL Leaf 308.JPG --model efficientnet
```

### Supported `--model` Arguments:
- `efficientnet` (Default — Transfer Learning model)
- `cnn` (Custom CNN)
- `svm` (Support Vector Machine on PCA features)
- `random_forest` (Random Forest Classifier)

---

## 5. Configuration Customization

All pipeline hyperparameters are defined in `config/config.yaml`. To modify pipeline behavior without changing code:

- **Change Image Resolution**: Edit `preprocessing.image_size` (e.g., `[224, 224]`).
- **Adjust PCA Components**: Edit `features.pca.n_components` (e.g., `100`).
- **Modify Training Epochs**: Edit `deep_learning.epochs` (e.g., `50`).
- **Tune SVM Hyperparameters**: Edit `classical_ml.svm.C` or `kernel`.
