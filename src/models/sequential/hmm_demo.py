import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger
from src.utils.common import timer, save_json
import os
import numpy as np
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

logger = get_logger(__name__)

class HMMDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        
    @timer
    def run_demo(self, X, y, class_names, save_dir):
        logger.info("Running HMMDemo...")
        print("HMM Demo: Treating spatial feature sequences as Markov processes.")
        os.makedirs(save_dir, exist_ok=True)
        
        if hmm is None:
            logger.error("hmmlearn not installed. Skipping HMM demo.")
            return {}
            
        models = {}
        log_likelihoods = {}
        
        for c_idx, c_name in enumerate(class_names):
            X_c = X[y == c_idx]
            if len(X_c) == 0: continue
            
            model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=10)
            try:
                # Treat each row as a single length sequence for simplicity in this demo
                lengths = [1] * len(X_c)
                model.fit(X_c, lengths)
                models[c_name] = model
                log_likelihoods[c_name] = float(model.score(X_c, lengths))
                
                # Visualize transition matrix for one of them
                if c_idx == 0:
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(model.transmat_, annot=True, cmap="YlGnBu")
                    plt.title(f"State Transition Matrix ({c_name})")
                    plt.savefig(os.path.join(save_dir, "hmm_transmat.png"), dpi=150)
                    plt.close()
                    
            except Exception as e:
                logger.warning(f"HMM fitting failed for class {c_name}: {e}")
                
        save_json(log_likelihoods, os.path.join(save_dir, "hmm_results.json"))
        return log_likelihoods
