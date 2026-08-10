"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch.

Generates visual explainability heatmaps for CNN predictions.

Usage
-----
    from src.models.deep_learning.gradcam import GradCAM
    gradcam = GradCAM(model)
    heatmap = gradcam.compute_heatmap(image_tensor, target_class=2)
"""

import numpy as np
import torch
import cv2
from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Grad-CAM for PyTorch neural network architectures."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None

        # Find target layer automatically if not provided
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer
        self._register_hooks()
        logger.info("GradCAM initialized for PyTorch model.")

    def _find_target_layer() -> torch.nn.Module:
        # Search backwards for a Conv2d layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found in model for Grad-CAM.")

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def compute_heatmap(self, image: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image (H, W, 3) in [0, 1].
        """
        self.model.eval()

        if image.ndim == 3:
            tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            tensor = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

        tensor.requires_grad = True
        output = self.model(tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        self.model.zero_grad()
        score.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]   # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def overlay_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on original RGB image."""
        if original_image.max() <= 1.0:
            orig_uint8 = (original_image * 255).astype(np.uint8)
        else:
            orig_uint8 = original_image.astype(np.uint8)

        h, w = orig_uint8.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def generate_grid(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: list,
        save_path: str,
        num_images: int = 10,
    ) -> None:
        """Generate and save a grid of Grad-CAM visualizations."""
        n = min(num_images, len(images))
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))

        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            img = images[i]
            true_idx = int(true_labels[i])
            pred_idx = int(pred_labels[i])

            heatmap = self.compute_heatmap(img, class_idx=pred_idx)
            overlay = self.overlay_heatmap(heatmap, img)

            display_img = img if img.max() <= 1.0 else img / 255.0
            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title(f"True: {class_names[true_idx]}", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(heatmap, cmap="jet")
            axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=10)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay)
            correct = "✓" if true_idx == pred_idx else "✗"
            axes[i, 2].set_title(f"Pred: {class_names[pred_idx]} {correct}", fontsize=10)
            axes[i, 2].axis("off")

        plt.suptitle("Grad-CAM Explanations", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grad-CAM grid saved to %s", save_path)
