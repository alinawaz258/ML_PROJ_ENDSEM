import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.common import timer

logger = get_logger(__name__)

class HOGExtractor:
    """Extract Histogram of Oriented Gradients (HOG) features."""
    
    def __init__(self, cfg):
        hog_cfg = cfg.get("features", {}).get("hog", {})
        self.orientations = hog_cfg.get("orientations", 9)
        self.pixels_per_cell = tuple(hog_cfg.get("pixels_per_cell", (16, 16)))
        self.cells_per_block = tuple(hog_cfg.get("cells_per_block", (2, 2)))
        self.block_norm = hog_cfg.get("block_norm", "L2-Hys")
        logger.info(f"Initialized HOGExtractor with orientations={self.orientations}, pixels_per_cell={self.pixels_per_cell}")

    def extract_single(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] == 3:
            image_gray = rgb2gray(image)
        else:
            image_gray = image
            
        features = hog(
            image_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            channel_axis=None
        )
        return features

    @timer
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        logger.info(f"Extracting HOG features for {len(images)} images.")
        features = []
        for img in tqdm(images, desc="HOG Extraction"):
            features.append(self.extract_single(img))
        return np.array(features, dtype=np.float32)

    def visualize(self, image: np.ndarray, save_path: str):
        if image.ndim == 3 and image.shape[-1] == 3:
            image_gray = rgb2gray(image)
        else:
            image_gray = image
            
        features, hog_image = hog(
            image_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=True,
            channel_axis=None
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image)
        ax1.set_title('Input Image')
        
        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)
        ax2.set_title('HOG Features')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved HOG visualization to {save_path}")
