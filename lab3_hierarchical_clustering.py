"""
Lab 3: Hierarchical Clustering (Agglomerative)
Unit 3 Lab Exercise

Goal:
Perform Hierarchical Clustering on feature representations and visualize
the hierarchy using a Dendrogram and 2D cluster projections.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# 1. Load dataset features
print("Loading data...")
X_train = np.load("artifacts/features/features_train.npy")
y_train = np.load("artifacts/features/labels_train.npy")

# Sample a subset for clean, readable visualization and quick execution
np.random.seed(42)
sample_size = 300
idx = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train[idx]
y_sample = y_train[idx]

print(f"Dataset loaded. Using {sample_size} samples for demonstration.")

# 2. Hierarchical Linkage and Dendrogram
print("Computing Ward linkage for dendrogram...")
linked = linkage(X_sample, method='ward')

# 3. Fit Agglomerative Clustering
k = 10  # 10 plant disease/healthy classes
model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
cluster_labels = model.fit_predict(X_sample)

# 4. Evaluation
score = silhouette_score(X_sample, cluster_labels)
print(f"\n--- Results ---")
print(f"Number of clusters: {k}")
print(f"Silhouette Score  : {score:.4f}")

# 5. Plot Dendrogram & 2D PCA Cluster visualization side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Dendrogram
dendrogram(linked, truncate_mode='lastp', p=25, ax=axes[0], leaf_rotation=45)
axes[0].set_title("Hierarchical Dendrogram (Ward Linkage)")
axes[0].set_xlabel("Cluster / Sample Index")
axes[0].set_ylabel("Distance (Dissimilarity)")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Reduce to 2D for scatter plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=35, alpha=0.8)
axes[1].set_title("Agglomerative Clusters (2D PCA Projection)")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("lab3_output.png", dpi=150)
print("\nPlot saved: lab3_output.png")
