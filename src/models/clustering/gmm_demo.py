import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from src.utils.logger import get_logger
from src.utils.common import timer, save_json
import numpy as np
import os

logger = get_logger(__name__)

class GMMDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        
    @timer
    def run_demo(self, X, y, class_names, save_dir):
        logger.info("Running GMMDemo...")
        print("GMM Demo: Modeling clusters as probabilistic Gaussian distributions.")
        os.makedirs(save_dir, exist_ok=True)
        
        k = len(class_names)
        gmm = GaussianMixture(n_components=k, random_state=42)
        preds = gmm.fit_predict(X)
        probas = gmm.predict_proba(X)
        
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        sil = float(silhouette_score(X, preds))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Soft assignment visualization based on max probability
        uncertainty = 1.0 - np.max(probas, axis=1)
        
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preds, s=50, alpha=0.7, cmap='Set2', 
                         edgecolors='k', linewidths=uncertainty*2) # Thicker edges for more uncertain points
        plt.title("GMM Clusters (Edge thickness indicates uncertainty)")
        plt.savefig(os.path.join(save_dir, "gmm_clusters.png"), dpi=150)
        plt.close()
        
        res = {'bic': bic, 'aic': aic, 'silhouette': sil}
        save_json(res, os.path.join(save_dir, "gmm_results.json"))
        return res
