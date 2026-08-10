import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.common import get_project_root, save_json, timer, ensure_dirs

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prep_cfg = cfg.get('preprocessing', {})
        self.image_size = tuple(self.prep_cfg.get('image_size', [128, 128]))
        self.normalize = self.prep_cfg.get('normalize', True)
        self.normalization_method = self.prep_cfg.get('normalization_method', 'minmax')
        
        self.project_root = get_project_root()
        self.features_dir = self.project_root / "artifacts" / "features"
        self.figures_dir = self.project_root / "artifacts" / "figures"
        self.reports_dir = self.project_root / "artifacts" / "reports"
        
        import os
        os.makedirs(str(self.features_dir), exist_ok=True)
        os.makedirs(str(self.figures_dir), exist_ok=True)
        os.makedirs(str(self.reports_dir), exist_ok=True)
        
    def load_image(self, filepath: str) -> np.ndarray:
        full_path = self.project_root / filepath
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        img = cv2.imread(str(full_path))
        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w, target_h = self.image_size
        
        if h > target_h or w > target_w:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
            
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return resized
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return image
            
        if self.normalization_method == 'minmax':
            return image.astype(np.float32) / 255.0
        elif self.normalization_method == 'standardize':
            img_float = image.astype(np.float32)
            mean = np.mean(img_float, axis=(0, 1))
            std = np.std(img_float, axis=(0, 1))
            std[std == 0] = 1.0
            return (img_float - mean) / std
        else:
            logger.warning(f"Unknown normalization method: {self.normalization_method}. Skipping normalization.")
            return image
            
    def preprocess_single(self, filepath: str) -> np.ndarray:
        img = self.load_image(filepath)
        img = self.resize_image(img)
        img = self.normalize_image(img)
        return img
        
    def preprocess_batch(self, filepaths: List[str], labels: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = []
        y = []
        
        for i, filepath in enumerate(tqdm(filepaths, desc="Preprocessing images")):
            try:
                img = self.preprocess_single(filepath)
                X.append(img)
                if labels is not None:
                    y.append(labels[i])
            except Exception as e:
                logger.warning(f"Skipping {filepath} due to error: {e}")
                
        X_arr = np.array(X)
        y_arr = np.array(y) if y else None
        
        return X_arr, y_arr
        
    def preprocess_split(self, split_csv: str) -> Tuple[np.ndarray, np.ndarray]:
        csv_path = self.reports_dir / split_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        logger.info(f"Processing split from {split_csv} with {len(df)} images.")
        
        filepaths = df['filepath'].tolist()
        labels = df['class_index'].tolist()
        
        return self.preprocess_batch(filepaths, labels)
        
    def save_preprocessed(self, X: np.ndarray, y: np.ndarray, split_name: str):
        x_path = self.features_dir / f"X_{split_name}.npy"
        y_path = self.features_dir / f"y_{split_name}.npy"
        
        logger.info(f"Saving X_{split_name} shape {X.shape} to {x_path}")
        np.save(str(x_path), X)
        logger.info(f"Saving y_{split_name} shape {y.shape} to {y_path}")
        np.save(str(y_path), y)
        
    def generate_comparison(self, df: pd.DataFrame):
        logger.info("Generating preprocessing comparison figure.")
        if len(df) == 0:
            logger.warning("Empty dataframe, cannot generate comparison.")
            return
            
        sample_df = df.sample(min(5, len(df)), random_state=42)
        
        fig, axes = plt.subplots(len(sample_df), 2, figsize=(10, 3 * len(sample_df)))
        if len(sample_df) == 1:
            axes = [axes]
            
        for ax_row, (_, row) in zip(axes, sample_df.iterrows()):
            filepath = row['filepath']
            
            try:
                orig_img = self.load_image(filepath)
                prep_img = self.preprocess_single(filepath)
                
                ax_row[0].imshow(orig_img)
                ax_row[0].set_title(f"Original\nShape: {orig_img.shape}")
                ax_row[0].axis('off')
                
                # If preprocessed image is min-max normalized, it's [0,1], which imshow handles fine.
                # If standardized, we need to normalize it to [0,1] for visualization.
                display_img = prep_img
                if self.normalization_method == 'standardize':
                    display_img = (prep_img - prep_img.min()) / (prep_img.max() - prep_img.min())
                    
                ax_row[1].imshow(display_img)
                ax_row[1].set_title(f"Preprocessed\nShape: {prep_img.shape}")
                ax_row[1].axis('off')
                
            except Exception as e:
                logger.warning(f"Error generating comparison for {filepath}: {e}")
                
        plt.tight_layout()
        fig_path = self.figures_dir / "before_after_preprocessing.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Saved comparison figure to {fig_path}")

    @timer
    def run(self):
        logger.info("Starting preprocessing pipeline.")
        
        splits = ['train', 'val', 'test']
        report = {
            "preprocessing_settings": self.prep_cfg,
            "splits": {}
        }
        
        train_df_path = self.reports_dir / "train.csv"
        
        for split in splits:
            try:
                csv_file = f"{split}.csv"
                X, y = self.preprocess_split(csv_file)
                self.save_preprocessed(X, y, split)
                
                report["splits"][split] = {
                    "num_images": len(X),
                    "X_shape": list(X.shape),
                    "y_shape": list(y.shape),
                    "min_val": float(np.min(X)) if len(X) > 0 else 0,
                    "max_val": float(np.max(X)) if len(X) > 0 else 0
                }
            except Exception as e:
                logger.error(f"Error processing split {split}: {e}")
                
        if train_df_path.exists():
            df = pd.read_csv(train_df_path)
            self.generate_comparison(df)
            
        report_path = self.reports_dir / "preprocessing_report.json"
        save_json(report, str(report_path))
        logger.info(f"Saved preprocessing report to {report_path}")
        return report
