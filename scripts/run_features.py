import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
from src.utils.common import load_config, get_project_root, resolve_path, ensure_dirs, timer
from src.utils.logger import get_logger
from src.features.hog_extractor import HOGExtractor
from src.features.lbp_extractor import LBPExtractor
from src.features.color_histogram import ColorHistogramExtractor
from src.features.pca_reducer import PCAReducer

logger = get_logger(__name__)

@timer
def main():
    root = get_project_root()
    cfg = load_config(root / "config" / "config.yaml")
    
    features_dir = root / cfg['paths']['features_dir']
    models_dir = root / cfg['paths']['models_dir']
    figures_dir = root / cfg['paths']['figures_dir']
    
    import os
    os.makedirs(str(features_dir), exist_ok=True)
    os.makedirs(str(models_dir), exist_ok=True)
    os.makedirs(str(figures_dir), exist_ok=True)
    
    class_names = cfg['dataset']['class_labels']
    
    logger.info("Loading preprocessed arrays...")
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    X_val = np.load(features_dir / "X_val.npy")
    y_val = np.load(features_dir / "y_val.npy")
    X_test = np.load(features_dir / "X_test.npy")
    y_test = np.load(features_dir / "y_test.npy")
    
    logger.info(f"X_train shape: {X_train.shape}")
    
    hog_ext = HOGExtractor(cfg)
    lbp_ext = LBPExtractor(cfg)
    ch_ext = ColorHistogramExtractor(cfg)
    
    # Generate visualizations on first train image
    sample_img = X_train[0]
    hog_ext.visualize(sample_img, str(figures_dir / "hog_visualization.png"))
    lbp_ext.visualize(sample_img, str(figures_dir / "lbp_visualization.png"))
    ch_ext.visualize(sample_img, str(figures_dir / "color_histogram_visualization.png"))
    
    def extract_all_features(X):
        f_hog = hog_ext.extract_batch(X)
        f_lbp = lbp_ext.extract_batch(X)
        f_ch = ch_ext.extract_batch(X)
        return np.hstack([f_hog, f_lbp, f_ch])
    
    logger.info("Extracting features for training set...")
    features_train = extract_all_features(X_train)
    logger.info("Extracting features for validation set...")
    features_val = extract_all_features(X_val)
    logger.info("Extracting features for test set...")
    features_test = extract_all_features(X_test)
    
    logger.info(f"Combined features shape: {features_train.shape}")
    
    pca_reducer = PCAReducer(cfg)
    logger.info("Fitting PCA on training features...")
    features_train_pca = pca_reducer.fit_transform(features_train)
    features_val_pca = pca_reducer.transform(features_val)
    features_test_pca = pca_reducer.transform(features_test)
    
    logger.info(f"PCA reduced features shape: {features_train_pca.shape}")
    
    pca_reducer.save(str(models_dir / "pca_model.pkl"))
    pca_reducer.visualize_variance(str(figures_dir / "pca_variance.png"))
    pca_reducer.visualize_2d(features_train_pca, y_train, class_names, str(figures_dir / "pca_2d_scatter.png"))
    
    logger.info("Saving processed features and labels...")
    np.save(features_dir / "features_train.npy", features_train_pca)
    np.save(features_dir / "features_val.npy", features_val_pca)
    np.save(features_dir / "features_test.npy", features_test_pca)
    
    np.save(features_dir / "labels_train.npy", y_train)
    np.save(features_dir / "labels_val.npy", y_val)
    np.save(features_dir / "labels_test.npy", y_test)
    
    logger.info("Feature engineering phase completed successfully.")

if __name__ == "__main__":
    main()
