"""
Metrics computation for model evaluation.

Computes accuracy, precision, recall, F1-score, confusion matrix,
classification report, and ROC-AUC across all models.

Usage
-----
    from src.evaluation.metrics import MetricsCalculator
    calc = MetricsCalculator(cfg)
    results = calc.compute_all(y_true, y_pred, y_proba, model_name="SVM")
"""

import numpy as np
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Unified metrics computation for all models.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from config.yaml.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.class_names = cfg["dataset"]["class_labels"]
        self.num_classes = cfg["dataset"]["num_classes"]
        self.average = cfg["evaluation"]["average"]

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Compute all classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True class labels, shape (N,).
        y_pred : np.ndarray
            Predicted class labels, shape (N,).
        y_proba : np.ndarray, optional
            Predicted probabilities, shape (N, num_classes).
        model_name : str
            Name of the model for logging.

        Returns
        -------
        dict
            Dictionary containing all computed metrics.
        """
        results = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(
                y_true, y_pred, average=self.average, zero_division=0,
            )),
            "recall": float(recall_score(
                y_true, y_pred, average=self.average, zero_division=0,
            )),
            "f1": float(f1_score(
                y_true, y_pred, average=self.average, zero_division=0,
            )),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.class_names, zero_division=0,
            ),
        }

        # Per-class metrics
        per_class_precision = precision_score(
            y_true, y_pred, average=None, zero_division=0,
        ).tolist()
        per_class_recall = recall_score(
            y_true, y_pred, average=None, zero_division=0,
        ).tolist()
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, zero_division=0,
        ).tolist()

        results["per_class_precision"] = per_class_precision
        results["per_class_recall"] = per_class_recall
        results["per_class_f1"] = per_class_f1

        # ROC-AUC (if probabilities are available)
        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                roc_auc = roc_auc_score(
                    y_true_bin, y_proba, average=self.average, multi_class="ovr",
                )
                results["roc_auc"] = float(roc_auc)
            except Exception as e:
                logger.warning("ROC-AUC computation failed for %s: %s", model_name, e)
                results["roc_auc"] = None

            try:
                results["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception as e:
                logger.warning("Log-loss computation failed for %s: %s", model_name, e)
                results["log_loss"] = None

        logger.info(
            "%s — Acc: %.4f, P: %.4f, R: %.4f, F1: %.4f",
            model_name, results["accuracy"], results["precision"],
            results["recall"], results["f1"],
        )

        return results

    @staticmethod
    def create_comparison_table(all_results: List[Dict[str, Any]]) -> str:
        """
        Create a formatted comparison table from multiple model results.

        Parameters
        ----------
        all_results : list of dict
            List of result dictionaries from ``compute_all``.

        Returns
        -------
        str
            Formatted comparison table string.
        """
        header = (
            f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
            f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for r in all_results:
            roc = f"{r.get('roc_auc', 'N/A'):>10.4f}" if r.get("roc_auc") else f"{'N/A':>10}"
            lines.append(
                f"{r['model_name']:<25} {r['accuracy']:>10.4f} "
                f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1']:>10.4f} {roc}"
            )

        lines.append(sep)
        return "\n".join(lines)
