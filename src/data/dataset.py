import os
import hashlib
from typing import Tuple
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.common import get_project_root, resolve_path, ensure_dirs, save_json, timer

logger = get_logger(__name__)

class DatasetManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.project_root = get_project_root()
        
        # Dataset paths
        self.dataset_dir = resolve_path(cfg['paths']['dataset_root'])
        
        # Output paths
        self.reports_dir = resolve_path(cfg['paths']['reports_dir'])
        self.figures_dir = resolve_path(cfg['paths']['figures_dir'])
        
        os.makedirs(str(self.reports_dir), exist_ok=True)
        os.makedirs(str(self.figures_dir), exist_ok=True)

    def scan_dataset(self) -> pd.DataFrame:
        logger.info("Scanning dataset directory...")
        
        data = []
        classes = sorted(os.listdir(self.dataset_dir))
        
        # Mapping from config if available
        config_labels = self.cfg.get('dataset', {}).get('class_labels', classes)
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Map index to config label if it exists in config in the same order
            class_label = config_labels[class_idx] if class_idx < len(config_labels) else class_name
            
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                
                if not os.path.isfile(file_path):
                    continue
                
                # Store relative path from project root for portability
                try:
                    rel_path = os.path.relpath(file_path, str(self.project_root))
                except ValueError:
                    rel_path = file_path
                
                try:
                    with Image.open(file_path) as img:
                        img.convert("RGB")
                    
                    data.append({
                        "filepath": rel_path,
                        "class_name": class_name,
                        "class_label": class_label,
                        "class_index": class_idx
                    })
                except (UnidentifiedImageError, IOError):
                    logger.warning(f"Corrupted or invalid image found: {file_path}")
                    
        df = pd.DataFrame(data)
        logger.info(f"Total valid images found: {len(df)}")
        return df

    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting duplicates...")
        
        def compute_md5(file_path):
            hash_md5 = hashlib.md5()
            try:
                full_path = os.path.join(str(self.project_root), file_path)
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            except Exception as e:
                logger.error(f"Failed to hash {file_path}: {e}")
                return None
                
        df = df.copy()
        df['hash'] = df['filepath'].apply(compute_md5)
        duplicates_df = df[df.duplicated(subset=['hash'], keep=False)].sort_values(by='hash')
        
        logger.info(f"Total duplicate images found: {len(duplicates_df)}")
        return duplicates_df

    def generate_statistics(self, df: pd.DataFrame) -> dict:
        logger.info("Generating statistics...")
        
        class_counts = df['class_name'].value_counts().to_dict()
        total_count = len(df)
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        mean_count = sum(class_counts.values()) / len(class_counts)
        
        # Sample images for dim
        sample_df = df.sample(min(100, len(df)), random_state=42)
        total_w, total_h = 0, 0
        valid_samples = 0
        
        for file_path in sample_df['filepath']:
            try:
                full_path = os.path.join(str(self.project_root), file_path)
                with Image.open(full_path) as img:
                    w, h = img.size
                    total_w += w
                    total_h += h
                    valid_samples += 1
            except Exception as e:
                logger.warning(f"Error reading image dimensions {file_path}: {e}")
                
        avg_w = total_w / valid_samples if valid_samples > 0 else 0
        avg_h = total_h / valid_samples if valid_samples > 0 else 0
        
        stats = {
            "total_images": total_count,
            "class_counts": class_counts,
            "min_images_per_class": min_count,
            "max_images_per_class": max_count,
            "mean_images_per_class": mean_count,
            "avg_width": avg_w,
            "avg_height": avg_h
        }
        
        return stats

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting dataset...")
        
        train_ratio = self.cfg['dataset']['split_ratios']['train']
        val_ratio = self.cfg['dataset']['split_ratios']['val']
        test_ratio = self.cfg['dataset']['split_ratios']['test']
        
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_ratio, 
            stratify=df['class_name'], 
            random_state=self.cfg['dataset'].get('random_state', 42)
        )
        
        val_rel_ratio = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_rel_ratio, 
            stratify=train_val_df['class_name'], 
            random_state=self.cfg['dataset'].get('random_state', 42)
        )
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

    def save_artifacts(self, df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, stats: dict, duplicates_df: pd.DataFrame):
        logger.info("Saving artifacts...")
        
        dataset_csv = os.path.join(self.reports_dir, "dataset.csv")
        train_csv = os.path.join(self.reports_dir, "train.csv")
        val_csv = os.path.join(self.reports_dir, "val.csv")
        test_csv = os.path.join(self.reports_dir, "test.csv")
        stats_json = os.path.join(self.reports_dir, "dataset_statistics.json")
        duplicates_csv = os.path.join(self.reports_dir, "duplicates.csv")
        summary_csv = os.path.join(self.reports_dir, "dataset_summary.csv")
        
        df.to_csv(dataset_csv, index=False)
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        duplicates_df.to_csv(duplicates_csv, index=False)
        
        summary_df = pd.DataFrame(list(stats['class_counts'].items()), columns=['class_name', 'count'])
        summary_df.to_csv(summary_csv, index=False)
        
        save_json(stats, stats_json)

    def visualize_distribution(self, df: pd.DataFrame):
        logger.info("Visualizing distribution...")
        
        plt.figure(figsize=(10, 8))
        sns.countplot(y='class_name', data=df, order=df['class_name'].value_counts().index, palette='viridis')
        plt.title('Class Distribution')
        plt.xlabel('Count')
        plt.ylabel('Class Name')
        plt.tight_layout()
        
        out_path = os.path.join(self.figures_dir, "class_distribution.png")
        plt.savefig(out_path)
        plt.close()

    def visualize_samples(self, df: pd.DataFrame):
        logger.info("Visualizing samples...")
        
        classes = sorted(df['class_name'].unique())
        num_classes = len(classes)
        
        fig, axes = plt.subplots(num_classes, 2, figsize=(10, 3 * num_classes))
        
        if num_classes == 1:
            axes = [axes]
            
        for idx, class_name in enumerate(classes):
            class_df = df[df['class_name'] == class_name]
            sample_df = class_df.sample(min(2, len(class_df)), random_state=42)
            
            for col_idx, (_, row) in enumerate(sample_df.iterrows()):
                try:
                    full_path = os.path.join(str(self.project_root), row['filepath'])
                    img = Image.open(full_path)
                    axes[idx][col_idx].imshow(img)
                    axes[idx][col_idx].axis('off')
                    if col_idx == 0:
                        axes[idx][col_idx].set_title(class_name, loc='left', pad=10)
                except Exception as e:
                    logger.warning(f"Error loading image for visualization {row['filepath']}: {e}")
                    
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "sample_images.png")
        plt.savefig(out_path)
        plt.close()

    @timer
    def run(self):
        logger.info("Starting DatasetManager execution")
        
        df = self.scan_dataset()
        duplicates_df = self.detect_duplicates(df)
        stats = self.generate_statistics(df)
        train_df, val_df, test_df = self.split_dataset(df)
        
        self.save_artifacts(df, train_df, val_df, test_df, stats, duplicates_df)
        self.visualize_distribution(df)
        self.visualize_samples(df)
        
        logger.info("DatasetManager execution completed")
        return df, train_df, val_df, test_df, stats, duplicates_df
