"""
Utility functions and configuration management.

This module provides:
- Config class: loads all settings from environment variables (.env)
- Logger setup: structured JSON logging
- Helper functions: reproducibility seed, path utilities
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Config:
    """Central configuration loaded from environment variables."""

    # Application
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "True").lower() == "true"
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")

    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/model/model.joblib")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_THRESHOLD: float = float(os.getenv("MODEL_THRESHOLD", 0.5))

    # Data
    DATA_RAW_PATH: str = os.getenv("DATA_RAW_PATH", "data/raw/")
    DATA_PROCESSED_PATH: str = os.getenv("DATA_PROCESSED_PATH", "data/processed/")
    TRAIN_TEST_SPLIT: float = float(os.getenv("TRAIN_TEST_SPLIT", 0.2))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", 42))

    # Training
    CV_FOLDS: int = int(os.getenv("CV_FOLDS", 5))
    OPTIMIZE_HYPERPARAMS: bool = os.getenv("OPTIMIZE_HYPERPARAMS", "True").lower() == "true"
    N_TRIALS: int = int(os.getenv("N_TRIALS", 50))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")

    # Monitoring
    DRIFT_REPORT_PATH: str = os.getenv("DRIFT_REPORT_PATH", "monitoring/reports/")
    DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", 0.1))

    # Docker
    DOCKER_IMAGE_NAME: str = os.getenv("DOCKER_IMAGE_NAME", "passos-magicos-ml")
    DOCKER_TAG: str = os.getenv("DOCKER_TAG", "latest")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger: Configured logger with JSON or text format.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if Config.LOG_FORMAT == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    logger.addHandler(console_handler)

    # File handler (ensure directory exists)
    log_path = Path(Config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def set_seed(seed: int = Config.RANDOM_STATE) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """
    Ensures a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path: The directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """
    Returns the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).parent.parent


def get_timestamp() -> str:
    """
    Returns the current UTC timestamp as a string.

    Returns:
        str: ISO 8601 timestamp string.
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
