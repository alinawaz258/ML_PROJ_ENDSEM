import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
from src.utils.common import load_config, get_project_root, ensure_dirs, save_json
from src.utils.logger import get_logger
from src.models.classical import SVMClassifier, RandomForestModel, DecisionTreeModel, AdaBoostModel, GradientBoostingModel

logger = get_logger(__name__)

def main():
    cfg = load_config()
    root = get_project_root()
    
    # Paths
    feat_dir = os.path.join(root, 'artifacts', 'features')
    model_dir = os.path.join(root, 'artifacts', 'models')
    report_dir = os.path.join(root, 'artifacts', 'reports')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    logger.info("Loading features...")
    X_train = np.load(os.path.join(feat_dir, 'features_train.npy'))
    y_train = np.load(os.path.join(feat_dir, 'labels_train.npy'))
    X_test = np.load(os.path.join(feat_dir, 'features_test.npy'))
    y_test = np.load(os.path.join(feat_dir, 'labels_test.npy'))
    
    models = {
        'SVM': SVMClassifier(cfg),
        'RandomForest': RandomForestModel(cfg),
        'DecisionTree': DecisionTreeModel(cfg),
        'AdaBoost': AdaBoostModel(cfg),
        'GradientBoosting': GradientBoostingModel(cfg)
    }
    
    results = {}
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 45)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.train(X_train, y_train)
        
        logger.info(f"Evaluating {name}...")
        eval_res = model.evaluate(X_test, y_test)
        results[name] = eval_res
        
        # Save model
        model.save(os.path.join(model_dir, f"{name.lower()}.joblib"))
        
        print(f"{name:<20} | {eval_res['accuracy']:.4f}     | {eval_res['f1']:.4f}")
        
    save_json(results, os.path.join(report_dir, 'classical_ml_results.json'))
    logger.info("Classical ML pipeline completed.")

if __name__ == "__main__":
    main()
