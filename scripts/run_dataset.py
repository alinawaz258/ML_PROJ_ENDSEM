import sys
from pathlib import Path
import random
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.common import load_config
from src.data.dataset import DatasetManager

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def main():
    try:
        logger.info("Loading configuration...")
        cfg = load_config()
        
        # Set global seed
        seed = cfg.get('random_state', 42)
        set_seed(seed)
        logger.info(f"Global seed set to {seed}")
        
        # Create dataset manager and run
        manager = DatasetManager(cfg)
        
        logger.info("Running dataset processing pipeline...")
        df, train_df, val_df, test_df, stats, duplicates_df = manager.run()
        
        print("\n--- Dataset Summary ---")
        print(f"Total valid images: {len(df)}")
        print(f"Total duplicate images: {len(duplicates_df)}")
        print(f"Training set: {len(train_df)} images")
        print(f"Validation set: {len(val_df)} images")
        print(f"Testing set: {len(test_df)} images")
        print(f"Average image dimensions: {stats['avg_width']:.2f}x{stats['avg_height']:.2f}")
        print("-----------------------\n")
        
        logger.info("Dataset script completed successfully.")
        
    except Exception as e:
        logger.exception(f"An error occurred during dataset processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
