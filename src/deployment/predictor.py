"""
Inference Predictor Engine for Tomato Leaf Disease Detection.

Supports predictions using Deep Learning (PyTorch Custom CNN / EfficientNetB0)
and Classical ML (SVM, Random Forest, Decision Tree, Boosting) models.

Usage
-----
    from src.deployment.predictor import Predictor
    predictor = Predictor(cfg, model_type="efficientnet")
    result = predictor.predict("path/to/leaf.jpg")
"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch

from src.utils.logger import get_logger
from src.utils.common import get_project_root, load_config
from src.features.hog_extractor import HOGExtractor
from src.features.lbp_extractor import LBPExtractor
from src.features.color_histogram import ColorHistogramExtractor

logger = get_logger(__name__)


class Predictor:
    """Unified predictor handling image preprocessing and model inference."""

    def __init__(self, cfg: dict, model_type: str = "efficientnet"):
        self.cfg = cfg
        self.model_type = model_type.lower()
        self.model = None
        self.pca_model = None
        self.class_labels = self.cfg["dataset"]["class_labels"]
        self.img_size = tuple(self.cfg["preprocessing"]["image_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self) -> None:
        """Load the specified model from disk."""
        root = get_project_root()
        models_dir = root / self.cfg["paths"]["models_dir"]

        try:
            if self.model_type in ("efficientnet", "efficientnetb0"):
                from src.models.deep_learning.efficientnet_transfer import EfficientNetTransfer
                eff = EfficientNetTransfer(self.cfg)
                eff.build()
                pt_path = models_dir / "efficientnet_best.pt"
                if pt_path.exists():
                    eff.load(str(pt_path))
                self.model = eff
                logger.info("Loaded EfficientNetB0 model.")

            elif self.model_type in ("cnn", "custom_cnn"):
                from src.models.deep_learning.custom_cnn import CustomCNN
                cnn = CustomCNN(self.cfg)
                cnn.build()
                pt_path = models_dir / "custom_cnn_best.pt"
                if pt_path.exists():
                    cnn.load(str(pt_path))
                self.model = cnn
                logger.info("Loaded Custom CNN model.")

            elif self.model_type in ("svm", "random_forest", "randomforest", "decision_tree", "adaboost", "gradient_boosting"):
                model_filename = f"{self.model_type.replace('_', '')}.joblib"
                model_path = models_dir / model_filename
                pca_path = models_dir / "pca_model.pkl"

                if not model_path.exists():
                    # Fallback lookup
                    model_path = models_dir / f"{self.model_type}_model.pkl"

                if not model_path.exists():
                    raise FileNotFoundError(f"Classical ML model not found: {model_path}")

                self.model = joblib.load(str(model_path))

                if pca_path.exists():
                    self.pca_model = joblib.load(str(pca_path))
                    logger.info("Loaded PCA model for feature reduction.")

                logger.info("Loaded classical ML model: %s", self.model_type)

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_type, e)
            raise

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load, resize, and normalize single image. Returns (1, H, W, 3)."""
        full_path = Path(image_path)
        if not full_path.is_absolute():
            full_path = get_project_root() / image_path

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = cv2.imread(str(full_path))
        if img is None:
            raise ValueError(f"Could not read image with OpenCV: {full_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, image_path: str) -> Dict[str, Any]:
        """Perform full prediction pipeline on a single image file."""
        img_array = self.preprocess_image(image_path)

        if self.model_type in ("efficientnet", "efficientnetb0", "cnn", "custom_cnn"):
            probs = self.model.predict_proba(img_array)[0]

        else:
            # Extract handcrafted features
            hog_ext = HOGExtractor(self.cfg)
            lbp_ext = LBPExtractor(self.cfg)
            ch_ext = ColorHistogramExtractor(self.cfg)

            f_hog = hog_ext.extract_batch(img_array)
            f_lbp = lbp_ext.extract_batch(img_array)
            f_ch = ch_ext.extract_batch(img_array)
            combined_feat = np.hstack([f_hog, f_lbp, f_ch])

            if self.pca_model is not None:
                combined_feat = self.pca_model.transform(combined_feat)

            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(combined_feat)[0]
                except Exception:
                    pred_class = int(self.model.predict(combined_feat)[0])
                    probs = np.zeros(len(self.class_labels))
                    probs[pred_class] = 1.0
            elif hasattr(self.model, "predict"):
                pred_class = int(self.model.predict(combined_feat)[0])
                probs = np.zeros(len(self.class_labels))
                probs[pred_class] = 1.0

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        class_name = self.class_labels[class_idx]

        all_probabilities = {
            label: float(prob) for label, prob in zip(self.class_labels, probs)
        }

        return {
            "image_path": str(image_path),
            "class_name": class_name,
            "class_index": class_idx,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict multiple images."""
        results = []
        for path in image_paths:
            try:
                results.append(self.predict(path))
            except Exception as e:
                logger.error("Skipping %s due to error: %s", path, e)
                results.append({"image_path": str(path), "error": str(e)})
        return results
