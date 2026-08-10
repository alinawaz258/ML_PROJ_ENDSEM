"""
EfficientNetB0 Transfer Learning for Tomato Leaf Disease Classification (PyTorch Engine).

Architecture
------------
Pre-trained EfficientNetB0 (ImageNet weights via torchvision.models)
with fine-tuned linear classifier head.

Usage
-----
    from src.models.deep_learning.efficientnet_transfer import EfficientNetTransfer
    model = EfficientNetTransfer(cfg)
    history = model.train(X_train, y_train, X_val, y_val)
    results = model.evaluate(X_test, y_test)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models

from src.utils.logger import get_logger
from src.utils.common import get_project_root, timer

logger = get_logger(__name__)


class EfficientNetTransfer:
    """
    EfficientNetB0 transfer learning model using PyTorch torchvision.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.dl_cfg = cfg["deep_learning"]
        self.eff_cfg = self.dl_cfg["efficientnet"]
        self.num_classes = cfg["dataset"]["num_classes"]
        self.image_size = tuple(self.dl_cfg["image_size"])
        self.batch_size = self.dl_cfg["batch_size"]
        self.epochs = self.dl_cfg["epochs"]
        self.learning_rate = self.dl_cfg["learning_rate"]
        self.fine_tune_lr = self.eff_cfg["fine_tune_lr"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = None
        logger.info("EfficientNetTransfer (PyTorch) initialized on device: %s", self.device)

    def build(self) -> None:
        """Build the EfficientNetB0 transfer learning model."""
        try:
            # Use torchvision pre-trained EfficientNetB0
            weights = models.EfficientNet_B0_Weights.DEFAULT
            eff_model = models.efficientnet_b0(weights=weights)
        except Exception as e:
            logger.warning("Could not download weights (%s), using uninitialized EfficientNetB0", e)
            eff_model = models.efficientnet_b0(weights=None)

        # Freeze early features initially
        for param in eff_model.features.parameters():
            param.requires_grad = False

        # Replace classification head
        in_features = eff_model.classifier[1].in_features
        eff_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, self.num_classes),
        )

        self.model = eff_model.to(self.device)
        logger.info("EfficientNetB0 PyTorch model built successfully.")

    def _unfreeze_features(self) -> None:
        """Unfreeze top feature layers for fine-tuning."""
        if self.model is not None:
            for param in self.model.features[-3:].parameters():
                param.requires_grad = True
            logger.info("Unfroze top feature blocks for fine-tuning.")

    @timer
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Train model in 2 phases: Feature Extraction + Fine-Tuning."""
        if self.model is None:
            self.build()

        X_tr = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_va = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
        y_va = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0.0
        root = get_project_root()
        ckpt_path = root / self.cfg["paths"]["models_dir"] / "efficientnet_best.pt"

        fe_epochs = 1 if self.device.type == "cpu" else max(2, min(self.epochs // 3, 5))
        ft_epochs = 2 if self.device.type == "cpu" else max(3, min(self.epochs - fe_epochs, 5))

        logger.info("PHASE 1: Feature Extraction (%d epochs)...", fe_epochs)
        for epoch in range(1, fe_epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            val_loss, val_acc = self._eval_loader(val_loader, criterion)
            history["loss"].append(running_loss / total)
            history["accuracy"].append(correct / total)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            logger.info("FE Epoch %d/%d — Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
                        epoch, fe_epochs, running_loss / total, correct / total, val_loss, val_acc)

        logger.info("PHASE 2: Fine-Tuning (%d epochs)...", ft_epochs)
        self._unfreeze_features()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.fine_tune_lr)

        for epoch in range(1, ft_epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            val_loss, val_acc = self._eval_loader(val_loader, criterion)
            history["loss"].append(running_loss / total)
            history["accuracy"].append(correct / total)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

            logger.info("FT Epoch %d/%d — Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
                        epoch, ft_epochs, running_loss / total, correct / total, val_loss, val_acc)

        self.history = history
        return history

    def _eval_loader(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        return val_loss_sum / val_total, val_correct / val_total

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.build()
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)

        all_probs = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for (images,) in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = softmax(outputs).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        class_names = self.cfg["dataset"]["class_labels"]

        results = {
            "model_name": "EfficientNetB0",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }

        logger.info("EfficientNetB0 Evaluation — Accuracy: %.4f, F1: %.4f", results["accuracy"], results["f1"])
        return results

    def save(self, path: str) -> None:
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            logger.info("EfficientNetB0 saved to %s", path)

    def load(self, path: str) -> None:
        if self.model is None:
            self.build()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("EfficientNetB0 loaded from %s", path)
