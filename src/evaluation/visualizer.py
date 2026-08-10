"""
Publication-quality visualization generator for model evaluation.

Generates confusion matrices, ROC curves, training curves, feature importance
plots, comparison bar charts, and model comparison dashboards.

Usage
-----
    from src.evaluation.visualizer import Visualizer
    viz = Visualizer(cfg)
    viz.plot_confusion_matrix(cm, class_names, "SVM", save_path)
    viz.plot_roc_curves(y_true, y_proba, class_names, "SVM", save_path)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """
    Visualization engine for model evaluation results.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from config.yaml.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.vis_cfg = cfg["visualization"]
        self.fig_size = tuple(self.vis_cfg["figure_size"])
        self.dpi = self.vis_cfg["dpi"]
        self.save_fmt = self.vis_cfg["save_format"]

        # Set global style
        try:
            plt.style.use(self.vis_cfg["style"])
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str,
        save_path: str,
        normalize: bool = True,
    ) -> None:
        """
        Plot and save a confusion matrix heatmap.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix, shape (num_classes, num_classes).
        class_names : list
            Class name labels.
        model_name : str
            Model name for the title.
        save_path : str
            Path to save the figure.
        normalize : bool
            If True, normalize the confusion matrix by row.
        """
        if isinstance(cm, list):
            cm = np.array(cm)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], linewidths=0.5, square=True,
        )
        axes[0].set_title(f"{model_name} — Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Predicted Label", fontsize=10)
        axes[0].set_ylabel("True Label", fontsize=10)
        axes[0].tick_params(axis="both", labelsize=7, rotation=45)

        # Normalized
        if normalize:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
            sns.heatmap(
                cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, square=True, vmin=0, vmax=1,
            )
            axes[1].set_title(
                f"{model_name} — Confusion Matrix (Normalized)",
                fontsize=12, fontweight="bold",
            )
            axes[1].set_xlabel("Predicted Label", fontsize=10)
            axes[1].set_ylabel("True Label", fontsize=10)
            axes[1].tick_params(axis="both", labelsize=7, rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved: %s", save_path)

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        model_name: str,
        save_path: str,
    ) -> None:
        """
        Plot multi-class ROC curves (one-vs-rest) with macro and micro averages.

        Parameters
        ----------
        y_true : np.ndarray
            True labels, shape (N,).
        y_proba : np.ndarray
            Predicted probabilities, shape (N, num_classes).
        class_names : list
            Class name labels.
        model_name : str
            Model name for the title.
        save_path : str
            Path to save the figure.
        """
        if isinstance(y_proba, list):
            y_proba = np.array(y_proba)

        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        # Per-class ROC curves
        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.8,
                    label=f"{cls_name} (AUC={roc_auc:.3f})")

        # Macro average
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2.5,
                label=f"Macro Avg (AUC={macro_auc:.3f})")

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curves saved: %s", save_path)

    def plot_comparison_bar_chart(
        self,
        all_results: List[Dict[str, Any]],
        save_path: str,
    ) -> None:
        """
        Plot a grouped bar chart comparing all models across metrics.

        Parameters
        ----------
        all_results : list of dict
            List of results from MetricsCalculator.compute_all.
        save_path : str
            Path to save the figure.
        """
        models = [r["model_name"] for r in all_results]
        metrics = ["accuracy", "precision", "recall", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

        x = np.arange(len(models))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(12, len(models) * 2), 7))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            values = [r.get(metric, 0) for r in all_results]
            bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Comparison bar chart saved: %s", save_path)

    def plot_per_class_f1(
        self,
        all_results: List[Dict[str, Any]],
        class_names: List[str],
        save_path: str,
    ) -> None:
        """
        Plot per-class F1 scores for each model as a grouped heatmap.

        Parameters
        ----------
        all_results : list of dict
            Results with 'per_class_f1' key.
        class_names : list
            Class name labels.
        save_path : str
            Path to save the figure.
        """
        # Filter results that have per_class_f1
        valid_results = [r for r in all_results if "per_class_f1" in r]
        if not valid_results:
            logger.warning("No per-class F1 data available for heatmap.")
            return

        models = [r["model_name"] for r in valid_results]
        data = np.array([r["per_class_f1"] for r in valid_results])

        fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(4, len(models) * 0.6)))
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="RdYlGn",
            xticklabels=class_names, yticklabels=models,
            ax=ax, linewidths=0.5, vmin=0, vmax=1,
        )
        ax.set_title("Per-Class F1-Score by Model", fontsize=13, fontweight="bold")
        ax.set_xlabel("Disease Class", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Per-class F1 heatmap saved: %s", save_path)

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        model_name: str,
        save_path: str,
        top_n: int = 20,
    ) -> None:
        """
        Plot feature importance as a horizontal bar chart.

        Parameters
        ----------
        importances : np.ndarray
            Feature importance values.
        feature_names : list
            Feature names.
        model_name : str
            Model name.
        save_path : str
            Path to save the figure.
        top_n : int
            Number of top features to display.
        """
        # Get top N features
        indices = np.argsort(importances)[-top_n:]
        top_importances = importances[indices]
        top_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                     for i in indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
        ax.barh(range(len(top_importances)), top_importances, color=colors)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{model_name} — Top {top_n} Feature Importances",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature importance saved: %s", save_path)
