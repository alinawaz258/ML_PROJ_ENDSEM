"""
Common utility functions shared across all pipeline modules.

Provides:
    - YAML configuration loading
    - Project root detection
    - Output directory creation
    - Timer decorator for profiling
    - JSON/CSV I/O helpers
"""

import json
import os
import time
import functools
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Walks upward from this file's location until it finds a directory
    containing ``config/config.yaml``, which marks the project root.

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If the project root cannot be determined.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety limit
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate project root (no config/config.yaml found)."
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and return the YAML configuration dictionary.

    Parameters
    ----------
    config_path : str, optional
        Absolute or relative path to the config file.
        If None, uses ``<project_root>/config/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = str(get_project_root() / "config" / "config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """
    Create all output directories specified in the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (from ``load_config``).
    """
    root = get_project_root()
    paths_section = cfg.get("paths", {})
    for key, rel_path in paths_section.items():
        if key == "dataset_root":
            continue  # don't create dataset dir
        full_path = root / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", full_path)
    logger.info("All output directories created.")


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.

    Parameters
    ----------
    relative_path : str
        Relative path from project root.

    Returns
    -------
    Path
        Absolute path.
    """
    return get_project_root() / relative_path


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    data : Any
        Data to serialize (must be JSON-serializable).
    filepath : str
        Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON saved: %s", filepath)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def timer(func):
    """
    Decorator that logs the execution time of a function.

    Usage:
        @timer
        def train_model(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("%s completed in %.2f seconds", func.__name__, elapsed)
        return result
    return wrapper
