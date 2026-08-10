#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.common import load_config
from src.preprocessing.preprocessor import Preprocessor

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

def main():
    logger.info("Initializing preprocessing run...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        config = load_config()
        
        preprocessor = Preprocessor(config)
        
        report = preprocessor.run()
        
        logger.info("Preprocessing completed successfully.")
        logger.info("Preprocessing summary:")
        for split, stats in report.get("splits", {}).items():
            logger.info(f"Split {split}: shape {stats['X_shape']}")
            
    except Exception as e:
        logger.error(f"Preprocessing run failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
