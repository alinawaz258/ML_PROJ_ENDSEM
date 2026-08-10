import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.common import timer

logger = get_logger(__name__)

class LBPExtractor:
    """Extract Local Binary Pattern (LBP) features."""
    
    def __init__(self, cfg):
        lbp_cfg = cfg.get("features", {}).get("lbp", {})
        self.num_points = lbp_cfg.get("num_points", 24)
        self.radius = lbp_cfg.get("radius", 3)
        self.method = lbp_cfg.get("method", "uniform")
        logger.info(f"Initialized LBPExtractor with num_points={self.num_points}, radius={self.radius}, method={self.method}")

    def extract_single(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] == 3:
            image_gray = rgb2gray(image)
        else:
            image_gray = image
            
        image_uint8 = (image_gray * 255).astype(np.uint8)
            
        lbp = local_binary_pattern(image_uint8, self.num_points, self.radius, self.method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    @timer
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        logger.info(f"Extracting LBP features for {len(images)} images.")
        features = []
        for img in tqdm(images, desc="LBP Extraction"):
            features.append(self.extract_single(img))
        return np.array(features, dtype=np.float32)

    def visualize(self, image: np.ndarray, save_path: str):
        if image.ndim == 3 and image.shape[-1] == 3:
            image_gray = rgb2gray(image)
        else:
            image_gray = image
            
        image_uint8 = (image_gray * 255).astype(np.uint8)
        lbp = local_binary_pattern(image_uint8, self.num_points, self.radius, self.method)
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Image')
        
        ax2.imshow(lbp, cmap='gray')
        ax2.axis('off')
        ax2.set_title('LBP Image')
        
        ax3.bar(np.arange(n_bins), hist)
        ax3.set_title('LBP Histogram')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved LBP visualization to {save_path}")
