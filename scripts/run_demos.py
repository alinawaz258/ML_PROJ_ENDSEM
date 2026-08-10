import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
from src.utils.common import load_config, get_project_root, ensure_dirs
from src.utils.logger import get_logger
from src.models.classical import BayesianLogisticModel, LinearRegressionDemo
from src.models.clustering import KMeansDemo, GMMDemo, HierarchicalDemo
from src.models.sequential import HMMDemo

logger = get_logger(__name__)

def main():
    cfg = load_config()
    root = get_project_root()
    class_names = cfg.get('dataset', {}).get('class_labels', [f"Class {i}" for i in range(10)])
    
    feat_dir = os.path.join(root, 'artifacts', 'features')
    fig_dir = os.path.join(root, 'artifacts', 'figures')
    report_dir = os.path.join(root, 'artifacts', 'reports')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    logger.info("Loading data for demos...")
    X_train = np.load(os.path.join(feat_dir, 'features_train.npy'))
    y_train = np.load(os.path.join(feat_dir, 'labels_train.npy'))
    X_test = np.load(os.path.join(feat_dir, 'features_test.npy'))
    y_test = np.load(os.path.join(feat_dir, 'labels_test.npy'))
    
    # We can use test set for clustering demos to keep them fast, or a subset
    X_demo = X_test[:1000] if len(X_test) > 1000 else X_test
    y_demo = y_test[:1000] if len(y_test) > 1000 else y_test
    
    # 1. Linear Regression Demo
    lr_demo = LinearRegressionDemo(cfg)
    lr_demo.run_demo(X_train, y_train, X_test, y_test, class_names, fig_dir)
    
    # 2. Bayesian Logistic
    bayesian = BayesianLogisticModel(cfg)
    bayesian.train(X_train, y_train)
    bayesian.evaluate(X_test, y_test)
    
    # 3. KMeans Demo
    kmeans = KMeansDemo(cfg)
    kmeans.run_demo(X_demo, y_demo, class_names, fig_dir)
    
    # 4. GMM Demo
    gmm = GMMDemo(cfg)
    gmm.run_demo(X_demo, y_demo, class_names, fig_dir)
    
    # 5. Hierarchical Demo
    hc = HierarchicalDemo(cfg)
    hc.run_demo(X_demo, y_demo, class_names, fig_dir)
    
    # 6. HMM Demo
    hmm = HMMDemo(cfg)
    hmm.run_demo(X_demo, y_demo, class_names, fig_dir)
    
    logger.info("All academic demonstrations completed.")

if __name__ == "__main__":
    main()
