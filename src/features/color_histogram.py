import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.common import timer

logger = get_logger(__name__)

class ColorHistogramExtractor:
    """Extract color histogram features."""
    
    def __init__(self, cfg):
        ch_cfg = cfg.get("features", {}).get("color_histogram", {})
        self.bins = ch_cfg.get("bins", [32, 32, 32])
        self.color_space = ch_cfg.get("color_space", "hsv").lower()
        logger.info(f"Initialized ColorHistogramExtractor with bins={self.bins}, color_space={self.color_space}")

    def extract_single(self, image: np.ndarray) -> np.ndarray:
        image_uint8 = (image * 255).astype(np.uint8)
        
        if self.color_space == "hsv":
            image_converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        elif self.color_space == "lab":
            image_converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        else:
            image_converted = image_uint8
            
        hist_features = []
        for i, b in enumerate(self.bins):
            hist, _ = np.histogram(image_converted[:, :, i], bins=b, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            hist_features.extend(hist)
            
        return np.array(hist_features, dtype=np.float32)

    @timer
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        logger.info(f"Extracting Color Histogram features for {len(images)} images.")
        features = []
        for img in tqdm(images, desc="Color Hist Extraction"):
            features.append(self.extract_single(img))
        return np.array(features, dtype=np.float32)

    def visualize(self, image: np.ndarray, save_path: str):
        image_uint8 = (image * 255).astype(np.uint8)
        
        if self.color_space == "hsv":
            image_converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
            colors = ('h', 's', 'v')
        elif self.color_space == "lab":
            image_converted = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            colors = ('l', 'a', 'b')
        else:
            image_converted = image_uint8
            colors = ('r', 'g', 'b')
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Image')
        
        plot_colors = ['r', 'g', 'b'] if self.color_space not in ["hsv", "lab"] else ['k', 'gray', 'lightgray']
        
        for i, (col, b) in enumerate(zip(colors, self.bins)):
            hist, bin_edges = np.histogram(image_converted[:, :, i], bins=b, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            ax2.plot(bin_edges[:-1], hist, color=plot_colors[i] if self.color_space == 'rgb' else None, label=f'Channel {col.upper()}')
            
        ax2.set_title(f'Color Histogram ({self.color_space.upper()})')
        ax2.set_xlim([0, 256])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Color Histogram visualization to {save_path}")
