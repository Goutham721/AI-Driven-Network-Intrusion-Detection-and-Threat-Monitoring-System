"""
utils.py
Shared helper functions used across the project: logging setup,
random IP/timestamp generation, and small formatting helpers.
"""

import logging
import random
import sys
from datetime import datetime

sys.path.append("..")
from config import LOG_FORMAT  # noqa: E402


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger so every module logs consistently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def random_ip() -> str:
    """Generate a random, non-routable-looking IPv4 address for simulation."""
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}." \
           f"{random.randint(0, 255)}.{random.randint(1, 254)}"


def random_internal_ip() -> str:
    """Generate an internal-looking private IPv4 address (destination side)."""
    return f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"


def current_timestamp() -> str:
    """Return the current time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_makedirs(path: str) -> None:
    """Create a directory if it does not already exist."""
    import os
    os.makedirs(path, exist_ok=True)
