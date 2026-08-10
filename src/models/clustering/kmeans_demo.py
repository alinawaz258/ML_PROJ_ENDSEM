import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from src.utils.logger import get_logger
from src.utils.common import timer, save_json
import os

logger = get_logger(__name__)

class KMeansDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        
    @timer
    def run_demo(self, X, y, class_names, save_dir):
        logger.info("Running KMeansDemo...")
        print("KMeans Demo: Partitioning data into k clusters based on distance to centroids.")
        os.makedirs(save_dir, exist_ok=True)
        
        # Elbow method
        inertias = []
        for k in range(2, min(16, len(X))):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
            
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, min(16, len(X))), inertias, marker='o')
        plt.title("Elbow Curve")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.savefig(os.path.join(save_dir, "kmeans_elbow.png"), dpi=150)
        plt.close()
        
        k = len(class_names)
        kmeans = KMeans(n_clusters=k, random_state=42)
        preds = kmeans.fit_predict(X)
        
        sil = float(silhouette_score(X, preds))
        db = float(davies_bouldin_score(X, preds))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', ax=ax1, legend=False)
        ax1.set_title("True Labels")
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=preds, palette='Set2', ax=ax2, legend=False)
        ax2.set_title("KMeans Clusters")
        plt.savefig(os.path.join(save_dir, "kmeans_clusters.png"), dpi=150)
        plt.close()
        
        res = {'silhouette': sil, 'davies_bouldin': db}
        save_json(res, os.path.join(save_dir, "kmeans_results.json"))
        return res
