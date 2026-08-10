"""
Custom Convolutional Neural Network for Tomato Leaf Disease Classification (PyTorch Engine).

Architecture
------------
A 3-block CNN (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D -> Dropout)
followed by Linear classification layers.

Usage
-----
    from src.models.deep_learning.custom_cnn import CustomCNN
    model = CustomCNN(cfg)
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

from src.utils.logger import get_logger
from src.utils.common import get_project_root, timer

logger = get_logger(__name__)


class PyTorchCNN(nn.Module):
    """PyTorch 3-block CNN architecture."""

    def __init__(self, num_classes: int = 10, input_size: int = 128) -> None:
        super().__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (32, 64, 64)
            nn.Dropout(0.25),
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (64, 32, 32)
            nn.Dropout(0.25),
        )
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (128, 16, 16)
            nn.Dropout(0.25),
        )
        # Spatial dimension after 3 maxpools: input_size // 8
        feat_dim = 128 * (input_size // 8) * (input_size // 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class CustomCNN:
    """
    Custom CNN wrapper managing model life-cycle, training, and evaluation.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.dl_cfg = cfg["deep_learning"]
        self.num_classes = cfg["dataset"]["num_classes"]
        self.image_size = tuple(self.dl_cfg["image_size"])
        self.batch_size = self.dl_cfg["batch_size"]
        self.epochs = self.dl_cfg["epochs"]
        self.learning_rate = self.dl_cfg["learning_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = None
        logger.info("CustomCNN (PyTorch) initialized on device: %s", self.device)

    def build(self) -> None:
        """Build the PyTorch CNN model."""
        self.model = PyTorchCNN(
            num_classes=self.num_classes,
            input_size=self.image_size[0],
        ).to(self.device)
        logger.info("CustomCNN PyTorch model built successfully.")

    @timer
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Train the CNN model."""
        if self.model is None:
            self.build()

        # Convert numpy NHWC [0,1] to torch NCHW float32
        X_tr = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_va = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
        y_va = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_tr, y_tr)
        val_dataset = TensorDataset(X_va, y_va)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0.0

        root = get_project_root()
        ckpt_path = root / self.cfg["paths"]["models_dir"] / "custom_cnn_best.pt"

        # Limit max training epochs to 3 for fast CPU execution
        max_epochs = 3 if self.device.type == "cpu" else self.epochs

        logger.info("Starting PyTorch CustomCNN training for %d epochs...", max_epochs)

        for epoch in range(1, max_epochs + 1):
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

            train_loss = running_loss / total
            train_acc = correct / total

            # Validation
            self.model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), ckpt_path)

            logger.info(
                "Epoch %2d/%2d — Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
                epoch, max_epochs, train_loss, train_acc, val_loss, val_acc,
            )

        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            self.build()
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

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
        """Evaluate on test set."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        class_names = self.cfg["dataset"]["class_labels"]

        results = {
            "model_name": "Custom CNN",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }

        logger.info("CustomCNN Evaluation — Accuracy: %.4f, F1: %.4f", results["accuracy"], results["f1"])
        return results

    def save(self, path: str) -> None:
        """Save state dict."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            logger.info("CustomCNN saved to %s", path)

    def load(self, path: str) -> None:
        """Load state dict."""
        if self.model is None:
            self.build()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("CustomCNN loaded from %s", path)
