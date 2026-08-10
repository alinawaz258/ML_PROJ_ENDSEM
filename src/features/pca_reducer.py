import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
from src.utils.logger import get_logger
from src.utils.common import timer

logger = get_logger(__name__)

class PCAReducer:
    """Dimensionality reduction using Principal Component Analysis."""
    
    def __init__(self, cfg):
        pca_cfg = cfg.get("features", {}).get("pca", {})
        self.n_components_config = pca_cfg.get("n_components", 50)
        self.whiten = pca_cfg.get("whiten", False)
        self.pca = None
        logger.info(f"Initialized PCAReducer with n_components={self.n_components_config}, whiten={self.whiten}")

    @timer
    def fit(self, X: np.ndarray) -> 'PCAReducer':
        n_samples, n_features = X.shape
        n_components = min(self.n_components_config, n_samples, n_features)
        logger.info(f"Fitting PCA with n_components={n_components} (samples={n_samples}, features={n_features})")
        
        self.pca = PCA(n_components=n_components, whiten=self.whiten)
        self.pca.fit(X)
        logger.info(f"PCA fitted. Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        return self

    @timer
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise ValueError("PCA is not fitted yet. Call fit() first.")
        logger.info(f"Transforming features using PCA: {X.shape}")
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        if self.pca is None:
            raise ValueError("PCA is not fitted. Cannot save.")
        joblib.dump(self.pca, path)
        logger.info(f"Saved PCA model to {path}")

    def load(self, path: str):
        self.pca = joblib.load(path)
        logger.info(f"Loaded PCA model from {path}")

    def visualize_variance(self, save_path: str):
        if self.pca is None:
            raise ValueError("PCA is not fitted. Cannot visualize variance.")
            
        plt.figure(figsize=(8, 6))
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved PCA variance visualization to {save_path}")

    def visualize_2d(self, X_reduced: np.ndarray, y: np.ndarray, class_names: list, save_path: str):
        if X_reduced.shape[1] < 2:
            logger.warning("Need at least 2 components to visualize 2D PCA. Skipping.")
            return
            
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1],
            hue=[class_names[int(label)] for label in y],
            palette='tab10',
            alpha=0.7
        )
        plt.title('2D PCA Scatter Plot')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved PCA 2D scatter visualization to {save_path}")
