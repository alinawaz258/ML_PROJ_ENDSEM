# Installation Guide
## AI-Based Tomato Leaf Disease Detection System

This document provides detailed instructions for setting up the environment, installing dependencies, and verifying your installation.

---

## 1. System Requirements

### Hardware Requirements
- **CPU**: Intel Core i5 / AMD Ryzen 5 or better (8+ cores recommended).
- **RAM**: Minimum 8 GB (16 GB recommended for feature extraction & training).
- **Disk Space**: Minimum 5 GB free disk space (excluding dataset).
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA support (GTX 1060 / RTX 2060 or better) with 4+ GB VRAM for faster deep learning training.

### Operating System Support
- Windows 10 / 11 (64-bit)
- Ubuntu 20.04 / 22.04 LTS (64-bit)
- macOS 12+ (Apple Silicon or Intel)

---

## 2. Prerequisites

### Python Installation
Ensure Python **3.10** or higher is installed:
```bash
python --version
```

### Git
Ensure Git is installed to clone the repository:
```bash
git --version
```

---

## 3. Environment Setup

### Method A: Using Standard Python Virtual Environment (`venv`)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/tomato-disease-detection.git
   cd tomato-disease-detection
   ```

2. **Create the virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```

### Method B: Using Conda

1. **Create Conda environment**:
   ```bash
   conda create -n tomato_env python=3.10 -y
   ```

2. **Activate environment**:
   ```bash
   conda activate tomato_env
   ```

---

## 4. Dependency Installation

Upgrade `pip` and install all required packages specified in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Required Packages Summary

| Package Category | Libraries | Purpose |
| :--- | :--- | :--- |
| **Core Scientific** | `numpy`, `pandas` | Data structures, array operations, CSV I/O |
| **Machine Learning** | `scikit-learn`, `joblib` | Classical classifiers, metrics, PCA, model persistence |
| **Deep Learning** | `tensorflow` | Keras API, Custom CNN, EfficientNetB0, Grad-CAM |
| **Computer Vision** | `opencv-python`, `scikit-image`, `Pillow` | Image loading, resizing, HOG, LBP extraction |
| **Visualization** | `matplotlib`, `seaborn` | Loss curves, confusion matrices, ROC plots |
| **Configuration & Utils** | `pyyaml`, `tqdm` | Config parsing, progress bars |
| **Sequential ML** | `hmmlearn` | Hidden Markov Model demonstration |
| **Web Deployment** | `streamlit` | Interactive web application dashboard |

---

## 5. Dataset Verification

Ensure the PlantVillage Tomato dataset is present at `archive/PlantVillage/`:

```text
archive/PlantVillage/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites_Two_spotted_spider_mite/
├── Tomato__Target_Spot/
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Tomato_mosaic_virus/
└── Tomato_healthy/
```

If your dataset is located elsewhere, update `paths.dataset_root` in `config/config.yaml`.

---

## 6. Installation Verification

Verify that all key packages import cleanly by running:

```bash
python -c "import tensorflow as tf; import sklearn; import cv2; import streamlit; print('Installation Successful! TF Version:', tf.__version__)"
```

If this command outputs `Installation Successful!` without errors, your environment is ready.
