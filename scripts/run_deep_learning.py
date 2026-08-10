"""
Deep Learning Training & Evaluation Script.

Trains the Custom CNN and EfficientNetB0 transfer learning models on the
preprocessed tomato leaf disease dataset. Generates training curves,
Grad-CAM visualizations, and evaluation metrics.

Usage
-----
    python scripts/run_deep_learning.py
    python -m scripts.run_deep_learning
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
from src.utils.common import load_config, get_project_root, ensure_dirs, save_json, timer
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def plot_training_curves(history: dict, model_name: str, save_path: str) -> None:
    """Plot training and validation loss/accuracy curves."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss curve
    ax1.plot(epochs, history["loss"], "b-o", label="Training Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=3)
    ax1.set_title(f"{model_name} — Loss Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["accuracy"], "b-o", label="Training Accuracy", markersize=3)
    ax2.plot(epochs, history["val_accuracy"], "r-o", label="Validation Accuracy", markersize=3)
    ax2.set_title(f"{model_name} — Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved: %s", save_path)


@timer
def train_custom_cnn(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate the Custom CNN."""
    from src.models.deep_learning.custom_cnn import CustomCNN

    root = get_project_root()
    figures_dir = root / cfg["paths"]["figures_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]

    logger.info("=" * 60)
    logger.info("TRAINING CUSTOM CNN")
    logger.info("=" * 60)

    model = CustomCNN(cfg)
    model.build()
    history = model.train(X_train, y_train, X_val, y_val)

    # Save training curves
    plot_training_curves(history, "Custom CNN", str(figures_dir / "cnn_training_curves.png"))

    # Evaluate on test set
    results = model.evaluate(X_test, y_test)

    # Save model
    model_path = str(models_dir / "custom_cnn_final.keras")
    model.save(model_path)

    # Save results
    results_save = {k: v for k, v in results.items() if k not in ("predictions", "probabilities")}
    save_json(results_save, str(reports_dir / "cnn_results.json"))

    # Grad-CAM visualization
    try:
        from src.models.deep_learning.gradcam import GradCAM

        last_conv = model.get_last_conv_layer_name()
        gradcam = GradCAM(model.model, last_conv)

        # Select diverse samples for Grad-CAM (one per class if possible)
        sample_indices = []
        for cls_idx in range(cfg["dataset"]["num_classes"]):
            cls_indices = np.where(y_test == cls_idx)[0]
            if len(cls_indices) > 0:
                sample_indices.append(cls_indices[0])

        sample_images = X_test[sample_indices]
        sample_true = y_test[sample_indices]
        sample_pred = np.array(results["predictions"])[sample_indices]

        gradcam.generate_grid(
            sample_images, sample_true, sample_pred,
            cfg["dataset"]["class_labels"],
            str(figures_dir / "gradcam_cnn.png"),
            num_images=len(sample_indices),
        )
    except Exception as e:
        logger.warning("Grad-CAM failed for Custom CNN: %s", e)

    # Print results
    logger.info("Custom CNN Results:")
    logger.info("  Accuracy:  %.4f", results["accuracy"])
    logger.info("  Precision: %.4f", results["precision"])
    logger.info("  Recall:    %.4f", results["recall"])
    logger.info("  F1-Score:  %.4f", results["f1"])
    print("\n" + results["classification_report"])

    return results, history


@timer
def train_efficientnet(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate the EfficientNetB0 transfer learning model."""
    from src.models.deep_learning.efficientnet_transfer import EfficientNetTransfer

    root = get_project_root()
    figures_dir = root / cfg["paths"]["figures_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]

    logger.info("=" * 60)
    logger.info("TRAINING EFFICIENTNETB0 TRANSFER LEARNING")
    logger.info("=" * 60)

    model = EfficientNetTransfer(cfg)
    model.build()
    history = model.train(X_train, y_train, X_val, y_val)

    # Save training curves
    plot_training_curves(
        history, "EfficientNetB0",
        str(figures_dir / "efficientnet_training_curves.png"),
    )

    # Evaluate on test set
    results = model.evaluate(X_test, y_test)

    # Save model
    model_path = str(models_dir / "efficientnet_final.keras")
    model.save(model_path)

    # Save results
    results_save = {k: v for k, v in results.items() if k not in ("predictions", "probabilities")}
    save_json(results_save, str(reports_dir / "efficientnet_results.json"))

    # Grad-CAM visualization
    try:
        from src.models.deep_learning.gradcam import GradCAM

        last_conv = model.get_last_conv_layer_name()
        gradcam = GradCAM(model.model, last_conv)

        sample_indices = []
        for cls_idx in range(cfg["dataset"]["num_classes"]):
            cls_indices = np.where(y_test == cls_idx)[0]
            if len(cls_indices) > 0:
                sample_indices.append(cls_indices[0])

        sample_images = X_test[sample_indices]
        sample_true = y_test[sample_indices]
        sample_pred = np.array(results["predictions"])[sample_indices]

        gradcam.generate_grid(
            sample_images, sample_true, sample_pred,
            cfg["dataset"]["class_labels"],
            str(figures_dir / "gradcam_efficientnet.png"),
            num_images=len(sample_indices),
        )
    except Exception as e:
        logger.warning("Grad-CAM failed for EfficientNet: %s", e)

    # Print results
    logger.info("EfficientNetB0 Results:")
    logger.info("  Accuracy:  %.4f", results["accuracy"])
    logger.info("  Precision: %.4f", results["precision"])
    logger.info("  Recall:    %.4f", results["recall"])
    logger.info("  F1-Score:  %.4f", results["f1"])
    print("\n" + results["classification_report"])

    return results, history


def main() -> None:
    """Main entry point for deep learning training."""
    logger.info("=" * 60)
    logger.info("DEEP LEARNING PIPELINE — Tomato Leaf Disease Detection")
    logger.info("=" * 60)

    # Load config and set seed
    cfg = load_config()
    set_global_seed(cfg["dataset"]["random_state"])
    ensure_dirs(cfg)

    root = get_project_root()
    features_dir = root / cfg["paths"]["features_dir"]

    # Load preprocessed data
    logger.info("Loading preprocessed image data...")
    X_train = np.load(str(features_dir / "X_train.npy"))
    y_train = np.load(str(features_dir / "y_train.npy"))
    X_val = np.load(str(features_dir / "X_val.npy"))
    y_val = np.load(str(features_dir / "y_val.npy"))
    X_test = np.load(str(features_dir / "X_test.npy"))
    y_test = np.load(str(features_dir / "y_test.npy"))

    logger.info("Data shapes — Train: %s, Val: %s, Test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # Train Custom CNN
    cnn_results, cnn_history = train_custom_cnn(
        cfg, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Train EfficientNetB0
    eff_results, eff_history = train_efficientnet(
        cfg, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Save combined deep learning results
    dl_summary = {
        "Custom_CNN": {
            "accuracy": cnn_results["accuracy"],
            "precision": cnn_results["precision"],
            "recall": cnn_results["recall"],
            "f1": cnn_results["f1"],
        },
        "EfficientNetB0": {
            "accuracy": eff_results["accuracy"],
            "precision": eff_results["precision"],
            "recall": eff_results["recall"],
            "f1": eff_results["f1"],
        },
    }
    save_json(dl_summary, str(root / cfg["paths"]["reports_dir"] / "deep_learning_results.json"))

    # Print comparison
    print("\n" + "=" * 60)
    print("DEEP LEARNING COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, metrics in dl_summary.items():
        print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    print("=" * 60)

    logger.info("Deep learning pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Deep learning pipeline failed: %s", e, exc_info=True)
        raise
