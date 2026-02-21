"""
Pytest configuration and shared fixtures.
"""

import os

import pytest

# Set test environment variables before any imports
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("MODEL_PATH", "app/model/model.joblib")
os.environ.setdefault("MODEL_VERSION", "1.0.0")
os.environ.setdefault("MODEL_THRESHOLD", "0.5")
os.environ.setdefault("RANDOM_STATE", "42")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # suppress logs during tests
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("CV_FOLDS", "3")
os.environ.setdefault("N_TRIALS", "5")
os.environ.setdefault("OPTIMIZE_HYPERPARAMS", "False")
os.environ.setdefault("DATA_RAW_PATH", "data/raw/")
os.environ.setdefault("DATA_PROCESSED_PATH", "data/processed/")
os.environ.setdefault("DRIFT_REPORT_PATH", "monitoring/reports/")
os.environ.setdefault("DRIFT_THRESHOLD", "0.1")
os.environ.setdefault("LOG_FILE", "logs/test.log")
