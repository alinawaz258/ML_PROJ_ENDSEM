import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from src.utils.logger import get_logger
from src.utils.common import timer, save_json
from sklearn.decomposition import PCA
import os
import numpy as np

logger = get_logger(__name__)

class HierarchicalDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        
    @timer
    def run_demo(self, X, y, class_names, save_dir):
        logger.info("Running HierarchicalDemo...")
        print("Hierarchical Demo: Building a tree of clusters using bottom-up approach.")
        os.makedirs(save_dir, exist_ok=True)
        
        # Downsample for dendrogram if too large
        idx = np.random.choice(len(X), min(500, len(X)), replace=False)
        X_sub = X[idx]
        
        linked = linkage(X_sub, 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, truncate_mode='lastp', p=30)
        plt.title("Hierarchical Clustering Dendrogram (Subset)")
        plt.savefig(os.path.join(save_dir, "hierarchical_dendrogram.png"), dpi=150)
        plt.close()
        
        k = len(class_names)
        hc = AgglomerativeClustering(n_clusters=k)
        preds = hc.fit_predict(X)
        
        sil = float(silhouette_score(X, preds))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=preds, palette='Set3', legend=False)
        plt.title("Hierarchical Clusters")
        plt.savefig(os.path.join(save_dir, "hierarchical_clusters.png"), dpi=150)
        plt.close()
        
        res = {'silhouette': sil}
        save_json(res, os.path.join(save_dir, "hierarchical_results.json"))
        return res
