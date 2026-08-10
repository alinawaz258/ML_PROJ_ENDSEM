"""
Global reproducibility seeding.

Sets deterministic seeds for Python, NumPy, and TensorFlow to ensure
identical results across runs. Must be called before any data loading
or model construction.

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed(42)
"""

import os
import random
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds across all random number generators.

    Parameters
    ----------
    seed : int
        The seed value (default: 42).
    """
    # Python built-in
    random.seed(seed)

    # Environment variable for hash-based randomness
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug("NumPy seed set to %d", seed)
    except ImportError:
        logger.warning("NumPy not available — skipping NumPy seed")

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Suppress TF info logs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logger.debug("TensorFlow seed set to %d", seed)
    except ImportError:
        logger.warning("TensorFlow not available — skipping TF seed")

    logger.info("Global seed set to %d", seed)
