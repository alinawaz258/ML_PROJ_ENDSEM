# Design Improvements Report
## Technical Review of the Original Software Design Document (SDD)

**Project**: AI-Based Tomato Leaf Disease Detection  
**Date**: 2026-08-10  
**Reviewer Role**: Principal ML Engineer & Software Architect

---

## Improvement 1 — Scope Reduction from 38 to 10 Classes

| Aspect | Detail |
|---|---|
| **Original Design** | Full PlantVillage dataset with 38 crop-disease classes across 14 species (54,303 images). |
| **Improved Design** | Tomato-only subset: 10 classes (9 diseases + 1 healthy), 16,012 images. |
| **Technical Justification** | The original design introduced unnecessary cross-species variance that dilutes disease-specific feature learning. A single-crop focus produces tighter decision boundaries, reduces training time by ~70%, and aligns with real-world deployment scenarios where a model targets one crop. The 10-class tomato subset remains sufficiently complex for a university-level multi-class classification project. |
| **Expected Benefit** | Faster iteration cycles, more meaningful per-class analysis, stronger class-specific feature engineering, and a more deployable end product. |

---

## Improvement 2 — HOG Features Replace GLCM

| Aspect | Detail |
|---|---|
| **Original Design** | GLCM (Gray-Level Co-occurrence Matrix) as the primary texture descriptor alongside LBP. |
| **Improved Design** | Replace GLCM with HOG (Histogram of Oriented Gradients) as the primary structural feature. Retain LBP for micro-texture capture. |
| **Technical Justification** | GLCM is a second-order statistical measure that captures texture co-occurrence but is computationally expensive for large image sets and produces a relatively small feature vector. HOG captures edge orientation distributions and structural patterns (lesion boundaries, vein structures, spot shapes) which are more discriminative for disease morphology. HOG is also the industry-standard feature for object and pattern recognition in computer vision (Dalal & Triggs, 2005). LBP remains for fine-grained texture (surface roughness, spot granularity). The combination of HOG (global structure) + LBP (local texture) + Color Histograms (chromatic shifts) provides a more complementary feature set. |
| **Expected Benefit** | Higher discriminative power for classical ML classifiers, better capture of disease boundary patterns, and alignment with standard CV literature. |

---

## Improvement 3 — Separation of Primary Pipeline from Academic Demonstrations

| Aspect | Detail |
|---|---|
| **Original Design** | All 12 algorithms integrated into a single monolithic pipeline with adapted justifications for non-applicable algorithms (e.g., Linear Regression for severity scoring, HMM for spatial scanlines). |
| **Improved Design** | Clear architectural separation: (1) **Primary Prediction Pipeline** — SVM, Random Forest, Ensemble methods, Custom CNN, EfficientNetB0. (2) **Academic Demonstration Modules** — Linear Regression, K-Means, GMM, Hierarchical Clustering, HMM, Bayesian Logistic Regression each in self-contained scripts with educational commentary. |
| **Technical Justification** | Forcing unsuitable algorithms (Linear Regression, HMM, K-Means) into the main classification pipeline introduces engineering debt, misleading benchmark comparisons, and fragile integration points. Separating them as self-contained demonstrations satisfies syllabus requirements without contaminating the production prediction path. Each demo module includes: problem framing, data adaptation, execution, visualization, and a printed explanation of why the algorithm behaves as it does on image data. |
| **Expected Benefit** | Cleaner architecture, honest benchmarking, easier maintenance, and transparent academic compliance. |

---

## Improvement 4 — Augmentation Strategy Moved to Deep Learning Only

| Aspect | Detail |
|---|---|
| **Original Design** | Data augmentation applied globally in the preprocessing stage before feature extraction. |
| **Improved Design** | Augmentation applied exclusively within the Deep Learning training pipeline via `tf.keras.preprocessing` or `tf.image` transformations. Classical ML models operate on original (non-augmented) extracted features. |
| **Technical Justification** | Augmenting images before handcrafted feature extraction (HOG/LBP/Color) artificially inflates the training set in a way that breaks the i.i.d. assumption of classical statistical models. Augmented copies of the same leaf produce near-identical feature vectors, causing data leakage in cross-validation folds. Deep learning models, by contrast, benefit from augmentation because the stochastic transformations act as implicit regularization during gradient-based optimization. |
| **Expected Benefit** | Eliminates data leakage risk in classical ML evaluation, produces honest cross-validation scores, and correctly leverages augmentation where it is theoretically justified. |

---

## Improvement 5 — Configuration-Driven Architecture

| Aspect | Detail |
|---|---|
| **Original Design** | Configuration file mentioned but not central to the architecture. Module behavior partially hardcoded. |
| **Improved Design** | Single `config/config.yaml` file drives ALL pipeline behavior: dataset paths, image dimensions, feature extraction parameters, model hyperparameters, training settings, output directories. Every module reads from this config. Zero hardcoded paths or magic numbers in source code. |
| **Technical Justification** | Configuration-driven design follows the Dependency Inversion Principle. It enables reproducibility (commit the config alongside results), hyperparameter experimentation (change YAML, re-run), and portability (different machines, different paths). |
| **Expected Benefit** | Full reproducibility, easy experimentation, zero-touch portability across environments. |

---

## Improvement 6 — Streamlit Replaces Flask for Deployment

| Aspect | Detail |
|---|---|
| **Original Design** | Generic "lightweight inferencing API" with TFLite/ONNX export. No specific web interface. |
| **Improved Design** | Streamlit-based interactive web application with drag-and-drop image upload, real-time prediction display, confidence visualization, and Grad-CAM overlay. |
| **Technical Justification** | Streamlit provides a zero-boilerplate web UI framework purpose-built for ML demos. It requires no HTML/CSS/JS knowledge, integrates natively with Python data science libraries, and produces visually polished interfaces in ~100 lines of code. For a university project, Streamlit dramatically reduces deployment complexity while producing a more impressive demonstration than a raw REST API. |
| **Expected Benefit** | Professional demo-ready deployment, faster development, interactive visualization for evaluators. |

---

## Improvement 7 — Stratified Split with Deterministic Seeding

| Aspect | Detail |
|---|---|
| **Original Design** | Stratified 70/15/15 train/val/test split mentioned but implementation details unspecified. |
| **Improved Design** | Stratified splitting using `sklearn.model_selection.train_test_split` with a fixed `random_state=42` across all components. Splits saved as CSV manifests (`train.csv`, `val.csv`, `test.csv`) with columns: `filepath`, `label`, `label_index`, `split`. All modules load from these CSVs rather than re-splitting. |
| **Technical Justification** | Saving split manifests as CSV files ensures that classical ML feature extraction, deep learning data generators, and evaluation scripts all operate on exactly the same data partitions. This eliminates subtle split-drift bugs where different random states produce different partitions across modules. |
| **Expected Benefit** | Guaranteed partition consistency, reproducible results, debuggable data lineage. |

---

## Improvement 8 — EfficientNetB0 as Primary Transfer Architecture

| Aspect | Detail |
|---|---|
| **Original Design** | Three transfer learning architectures: MobileNetV2, ResNet50V2, EfficientNetB0. |
| **Improved Design** | Single transfer learning architecture: EfficientNetB0 with compound scaling. Custom CNN as the lightweight baseline. |
| **Technical Justification** | Training three transfer architectures triples GPU compute time without proportional academic value. EfficientNetB0 (Tan & Le, 2019) achieves superior accuracy-to-parameter ratio compared to both MobileNetV2 and ResNet50V2. It is the optimal single-architecture choice for demonstrating transfer learning. The Custom CNN provides the necessary baseline comparison. Two models (Custom CNN vs. EfficientNetB0) produce a clean, interpretable comparison without redundancy. |
| **Expected Benefit** | Reduced compute budget, cleaner comparison narrative, focus on quality over quantity. |

---

## Improvement 9 — Script-per-Phase Execution Model

| Aspect | Detail |
|---|---|
| **Original Design** | Single `main.py` orchestrator running the entire pipeline end-to-end. |
| **Improved Design** | Individual `scripts/run_*.py` files for each phase (dataset, preprocessing, features, classical ML, deep learning, evaluation, demos) PLUS a unified `main.py` that calls them sequentially. Each script is independently executable. |
| **Technical Justification** | A monolithic orchestrator fails on partial execution (e.g., re-running only evaluation after changing a metric). Independent scripts allow selective re-execution, easier debugging, and phase-level testing. The unified `main.py` provides a single-command full-pipeline option for convenience. |
| **Expected Benefit** | Granular control, faster debugging, selective re-runs, cleaner logging per phase. |

---

## Summary of Changes

| # | Change | Impact | Risk |
|---|---|---|---|
| 1 | Scope to tomato-only | High | None |
| 2 | HOG replaces GLCM | Medium | None |
| 3 | Pipeline vs. demo separation | High | None |
| 4 | Augmentation in DL only | Medium | None |
| 5 | Config-driven architecture | High | None |
| 6 | Streamlit deployment | Medium | None |
| 7 | CSV-based split manifests | Medium | None |
| 8 | Single transfer architecture | Low | None |
| 9 | Script-per-phase execution | Medium | None |
