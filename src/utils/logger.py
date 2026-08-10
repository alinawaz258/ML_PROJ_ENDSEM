"""
Centralized logging configuration for the pipeline.

Provides a factory function that creates consistently formatted loggers
for every module. Logs are written to both console (stdout) and a
persistent log file defined in config.yaml.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    log_file : str, optional
        Path to the log file. If None, logs go to console only.
    level : int
        Logging level (default: ``logging.INFO``).
    fmt : str, optional
        Custom format string. Uses a sensible default if not provided.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    if fmt is None:
        fmt = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — only if a path is provided
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
