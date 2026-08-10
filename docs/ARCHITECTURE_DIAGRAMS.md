# Architecture Diagrams & Software Specifications
## AI-Based Tomato Leaf Disease Detection System

This document contains the complete visual software architecture, data flow diagrams, pipeline diagrams, module dependency charts, and execution flow charts for the Tomato Leaf Disease Detection system.

---

## 1. System Architecture Diagram

High-level modular architecture showing client interactions, configuration management, data ingestion, feature engineering, model suites, evaluation engine, and deployment interfaces.

```mermaid
graph TD
    subgraph Client / User Layer
        CLI[Main CLI Runner main.py]
        ScriptRunner[Phase Scripts scripts/run_*.py]
        WebUI[Streamlit Web App app/streamlit_app.py]
        PredictCLI[Prediction CLI scripts/predict.py]
    end

    subgraph Orchestration & Config Layer
        Config[Central Config config/config.yaml]
        Logger[Logging Factory src/utils/logger.py]
        Seed[Seed Manager src/utils/seed.py]
    end

    subgraph Data & Feature Pipeline Layer
        Loader[Dataset Manager src/data/dataset.py]
        Prep[Preprocessor Engine src/preprocessing/preprocessor.py]
        FE[Feature Extractor HOG+LBP+ColorHist src/features/]
        PCA[PCA Reducer src/features/pca_reducer.py]
    end

    subgraph Machine Learning & Deep Learning Layer
        PrimaryML[Classical ML Suite SVM, RF, CART, Boosting]
        DemoML[Academic Demos LinReg, Bayes, K-Means, GMM, Hierarchical, HMM]
        DeepNet[Deep Learning Suite Custom CNN & EfficientNetB0]
    end

    subgraph Evaluation & Deployment Layer
        Eval[Metrics Calculator & Visualizer src/evaluation/]
        XAI[Grad-CAM Engine src/models/deep_learning/gradcam.py]
        Deploy[Predictor & Model Exporter src/deployment/]
    end

    CLI & ScriptRunner --> Config
    CLI & ScriptRunner --> Loader
    Loader --> Prep
    Prep --> FE
    FE --> PCA
    
    PCA --> PrimaryML
    FE --> DemoML
    Prep --> DeepNet
    
    PrimaryML & DeepNet --> Eval
    DeepNet --> XAI
    Eval --> Deploy
    WebUI & PredictCLI --> Deploy
```

---

## 2. Data Flow Diagram (DFD)

Data transformations from raw PlantVillage image files to preprocessed tensors, extracted features, model logits, and final diagnostic classifications.

```mermaid
flowchart TD
    A[Raw Image Files PlantVillage Tomato] -->|Directory Scan & PIL Validation| B[Structured DataFrame & Manifests dataset.csv, train.csv]
    
    B -->|Batch Read & BGR-to-RGB| C[Raw Color Image Array 256x256x3]
    C -->|Bilinear Resize & CLAHE| D[Normalized Image Tensor 128x128x3]
    
    D -->|Skimage HOG| E1[HOG Feature Vector 324-dim]
    D -->|Skimage LBP| E2[LBP Texture Histogram 26-dim]
    D -->|OpenCV HSV| E3[3D Color Histogram 96-dim]
    
    E1 & E2 & E3 -->|Concatenate| F[Combined Feature Matrix 446-dim]
    F -->|Scikit-Learn PCA| G[PCA Reduced Feature Matrix K=50]
    
    G -->|Fit SVC & RandomForest| H1[Classical Predictions & Probabilities]
    D -->|Forward Pass Keras CNN & EfficientNet| H2[Deep Learning Predictions & Probabilities]
    
    H1 & H2 -->|Scikit-Learn Metrics| I[Performance Metrics & Visualizations]
```

---

## 3. Complete Processing Pipeline Diagram

Sequential data processing flow demarcating primary classification paths versus academic syllabus demonstration paths.

```mermaid
graph LR
    subgraph Ingestion & Preprocessing
        P1[Raw Images] --> P2[Resize 128x128]
        P2 --> P3[MinMax Normalization]
    end
    
    subgraph Feature Engineering Branch
        P3 --> F1[HOG Extraction]
        P3 --> F2[LBP Extraction]
        P3 --> F3[HSV Histogram]
        F1 & F2 & F3 --> F4[Combined Vector]
        F4 --> F5[PCA Compression K=50]
    end
    
    subgraph Primary Prediction Path
        F5 --> M1[SVM Classifier]
        F5 --> M2[Random Forest]
        P3 --> M3[Custom CNN]
        P3 --> M4[EfficientNetB0]
    end

    subgraph Academic Demonstration Path
        F5 --> D1[Linear Regression]
        F5 --> D2[Bayesian Logistic]
        P3 --> D3[K-Means / GMM]
        F5 --> D4[Hierarchical Dendrogram]
        F4 --> D5[Spatial HMM]
    end
    
    M1 & M2 & M3 & M4 --> E[Evaluation & Grad-CAM]
```

---

## 4. Module Dependency Diagram

Inter-module code dependencies showing clean separation between data loaders, feature extractors, model implementations, and deployment services.

```mermaid
graph LR
    Utils[src/utils/common.py & logger.py] --> Data[src/data/dataset.py]
    Utils --> Prep[src/preprocessing/preprocessor.py]
    Utils --> Feat[src/features/]
    Utils --> Models[src/models/]
    Utils --> Eval[src/evaluation/]
    
    Data --> Scripts[scripts/run_*.py]
    Prep --> Scripts
    Feat --> Models
    Models --> Eval
    Eval --> Deploy[src/deployment/]
    Deploy --> Streamlit[app/streamlit_app.py]
```

---

## 5. Execution Flow Diagram

Control flow sequence when executing `main.py` or individual phase scripts.

```mermaid
sequenceDiagram
    autonumber
    participant User as User / CLI
    participant Main as main.py
    participant Loader as DatasetManager
    participant Prep as Preprocessor
    participant FE as FeatureExtractor
    participant ML as ModelTrainer
    participant Eval as Evaluator

    User->>Main: python main.py
    Main->>Main: Load config.yaml & Set Seed(42)
    Main->>Loader: Scan & Split Dataset
    Loader-->>Main: Save dataset.csv, train.csv, val.csv, test.csv
    Main->>Prep: Preprocess Image Splits
    Prep-->>Main: Save X_train.npy, X_val.npy, X_test.npy
    Main->>FE: Extract HOG+LBP+Color & Fit PCA
    FE-->>Main: Save features_train.npy & pca_model.pkl
    Main->>ML: Train SVM, RF, CNN, EfficientNetB0
    ML-->>Main: Save *.pkl and *.keras models
    Main->>Eval: Compute Metrics & Generate Plots
    Eval-->>Main: Save final_evaluation_report.json
    Main-->>User: Execution Complete! Outputs in artifacts/
```
