"""
Comprehensive Evaluation & Comparison Script.

Loads all saved model results (classical ML and deep learning), aggregates
metrics, generates publication-quality comparison visualizations, and
produces the final evaluation report.

Usage
-----
    python scripts/run_evaluation.py
    python -m scripts.run_evaluation
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.utils.logger import get_logger
from src.utils.common import load_config, get_project_root, ensure_dirs, save_json, load_json, timer
from src.utils.seed import set_global_seed
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualizer import Visualizer

logger = get_logger(__name__)


@timer
def main() -> None:
    """Run the comprehensive evaluation and comparison pipeline."""
    logger.info("=" * 60)
    logger.info("EVALUATION & COMPARISON PIPELINE")
    logger.info("=" * 60)

    cfg = load_config()
    set_global_seed(cfg["dataset"]["random_state"])
    ensure_dirs(cfg)

    root = get_project_root()
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    features_dir = root / cfg["paths"]["features_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    class_names = cfg["dataset"]["class_labels"]
    viz = Visualizer(cfg)
    calc = MetricsCalculator(cfg)

    # Load test data
    logger.info("Loading test data...")
    y_test = np.load(str(features_dir / "y_test.npy"))

    # Collect all results
    all_results = []

    # ---- Load Classical ML Results ----
    classical_results_path = reports_dir / "classical_ml_results.json"
    if classical_results_path.exists():
        logger.info("Loading classical ML results...")
        classical_data = load_json(str(classical_results_path))

        for model_name, metrics in classical_data.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                result = {
                    "model_name": model_name,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1": metrics.get("f1", 0),
                    "roc_auc": metrics.get("roc_auc"),
                }
                if "per_class_f1" in metrics:
                    result["per_class_f1"] = metrics["per_class_f1"]
                if "confusion_matrix" in metrics:
                    result["confusion_matrix"] = metrics["confusion_matrix"]
                    # Generate confusion matrix plot
                    viz.plot_confusion_matrix(
                        np.array(metrics["confusion_matrix"]),
                        class_names, model_name,
                        str(figures_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"),
                    )
                all_results.append(result)
    else:
        logger.warning("Classical ML results not found at %s", classical_results_path)

    # ---- Load Deep Learning Results ----
    for dl_name, dl_file in [("Custom CNN", "cnn_results.json"), ("EfficientNetB0", "efficientnet_results.json")]:
        dl_path = reports_dir / dl_file
        if dl_path.exists():
            logger.info("Loading %s results...", dl_name)
            dl_data = load_json(str(dl_path))

            result = {
                "model_name": dl_data.get("model_name", dl_name),
                "accuracy": dl_data.get("accuracy", 0),
                "precision": dl_data.get("precision", 0),
                "recall": dl_data.get("recall", 0),
                "f1": dl_data.get("f1", 0),
                "roc_auc": dl_data.get("roc_auc"),
            }
            if "per_class_f1" in dl_data:
                result["per_class_f1"] = dl_data["per_class_f1"]
            if "confusion_matrix" in dl_data:
                result["confusion_matrix"] = dl_data["confusion_matrix"]
                viz.plot_confusion_matrix(
                    np.array(dl_data["confusion_matrix"]),
                    class_names, dl_name,
                    str(figures_dir / f"cm_{dl_name.lower().replace(' ', '_')}.png"),
                )

            # ROC curves from probabilities
            if "probabilities" in dl_data:
                y_proba = np.array(dl_data["probabilities"])
                viz.plot_roc_curves(
                    y_test[:len(y_proba)], y_proba, class_names, dl_name,
                    str(figures_dir / f"roc_{dl_name.lower().replace(' ', '_')}.png"),
                )

            all_results.append(result)
        else:
            logger.warning("%s results not found at %s", dl_name, dl_path)

    if not all_results:
        logger.error("No model results found. Run training scripts first.")
        return

    # ---- Generate Comparison Visualizations ----
    logger.info("Generating comparison visualizations...")

    # Comparison bar chart
    viz.plot_comparison_bar_chart(
        all_results,
        str(figures_dir / "model_comparison_bar_chart.png"),
    )

    # Per-class F1 heatmap
    viz.plot_per_class_f1(
        all_results, class_names,
        str(figures_dir / "per_class_f1_heatmap.png"),
    )

    # ---- Print Comparison Table ----
    table = calc.create_comparison_table(all_results)
    print("\n" + table)

    # ---- Save Final Report ----
    final_report = {
        "num_models_evaluated": len(all_results),
        "class_names": class_names,
        "results": [],
    }

    for r in all_results:
        report_entry = {
            "model_name": r["model_name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r.get("roc_auc"),
        }
        final_report["results"].append(report_entry)

    # Sort by F1 score
    final_report["results"].sort(key=lambda x: x["f1"], reverse=True)
    final_report["best_model"] = final_report["results"][0]["model_name"]
    final_report["best_f1"] = final_report["results"][0]["f1"]

    save_json(final_report, str(reports_dir / "final_evaluation_report.json"))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Models evaluated: {len(all_results)}")
    print(f"Best model: {final_report['best_model']} (F1={final_report['best_f1']:.4f})")
    print(f"\nReport saved: {reports_dir / 'final_evaluation_report.json'}")
    print(f"Figures saved: {figures_dir}")
    print("=" * 60)

    logger.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Evaluation pipeline failed: %s", e, exc_info=True)
        raise
