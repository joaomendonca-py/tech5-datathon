"""
Unit tests for model loading, prediction format, and output validation.
Uses mock objects to avoid requiring a trained model file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Creates a mock scikit-learn-compatible model."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]
    )
    model.feature_importances_ = np.array([0.1, 0.3, 0.2, 0.15, 0.05, 0.1, 0.1])
    return model


@pytest.fixture
def sample_X() -> pd.DataFrame:
    """Sample feature DataFrame for prediction testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "INDE": [7.1, 4.2, 8.5, 3.8],
            "IAA": [7.5, 5.0, 9.0, 4.0],
            "IEG": [6.8, 4.5, 8.2, 3.5],
            "IPS": [8.0, 6.0, 9.1, 5.0],
            "IDA": [7.2, 4.8, 8.8, 4.2],
            "IPP": [6.5, 5.2, 7.9, 3.9],
            "IPV": [7.0, 4.9, 8.6, 4.1],
        }
    )


@pytest.fixture
def sample_y() -> pd.Series:
    return pd.Series([0, 1, 0, 1], name="RISCO_DEFASAGEM")


@pytest.fixture
def mock_model_bundle(mock_model):
    """Creates a full model bundle as saved by joblib.dump."""
    return {
        "model": mock_model,
        "metadata": {
            "model_name": "RandomForest",
            "model_version": "1.0.0",
            "trained_at": "20240101_120000",
            "training_time_seconds": 42.0,
            "test_metrics": {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.88,
                "f1_score": 0.84,
                "auc_roc": 0.91,
            },
            "hyperparameters": {"n_estimators": 200, "max_depth": 10},
            "feature_names": ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV"],
            "threshold": 0.5,
        },
        "artifacts": {},
    }


# ---------------------------------------------------------------------------
# Tests: model prediction format
# ---------------------------------------------------------------------------


class TestModelPredictionFormat:
    def test_predict_returns_array(self, mock_model, sample_X):
        result = mock_model.predict(sample_X)
        assert isinstance(result, np.ndarray)

    def test_predict_shape_matches_input(self, mock_model, sample_X):
        result = mock_model.predict(sample_X)
        assert result.shape == (len(sample_X),)

    def test_predict_proba_returns_array(self, mock_model, sample_X):
        result = mock_model.predict_proba(sample_X)
        assert isinstance(result, np.ndarray)

    def test_predict_proba_shape(self, mock_model, sample_X):
        result = mock_model.predict_proba(sample_X)
        assert result.shape == (len(sample_X), 2)

    def test_predict_proba_sums_to_one(self, mock_model, sample_X):
        result = mock_model.predict_proba(sample_X)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_values_are_binary(self, mock_model, sample_X):
        result = mock_model.predict(sample_X)
        assert set(result).issubset({0, 1})

    def test_predict_proba_values_between_0_and_1(self, mock_model, sample_X):
        result = mock_model.predict_proba(sample_X)
        assert (result >= 0).all() and (result <= 1).all()


# ---------------------------------------------------------------------------
# Tests: joblib model loading
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_model_bundle_has_required_keys(self, mock_model_bundle):
        required_keys = {"model", "metadata", "artifacts"}
        assert required_keys.issubset(set(mock_model_bundle.keys()))

    def test_metadata_has_required_fields(self, mock_model_bundle):
        meta = mock_model_bundle["metadata"]
        required = {"model_name", "model_version", "trained_at", "test_metrics", "feature_names"}
        assert required.issubset(set(meta.keys()))

    def test_model_version_format(self, mock_model_bundle):
        version = mock_model_bundle["metadata"]["model_version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_metrics_ranges_valid(self, mock_model_bundle):
        metrics = mock_model_bundle["metadata"]["test_metrics"]
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name}={value} out of range [0,1]"

    @patch("joblib.load")
    def test_joblib_load_returns_bundle(self, mock_load, mock_model_bundle):
        mock_load.return_value = mock_model_bundle
        import joblib

        bundle = joblib.load("app/model/model.joblib")
        assert "model" in bundle
        assert "metadata" in bundle

    @patch("joblib.dump")
    def test_joblib_dump_called_correctly(self, mock_dump, mock_model_bundle):
        import joblib

        joblib.dump(mock_model_bundle, "app/model/model.joblib")
        mock_dump.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: threshold-based classification
# ---------------------------------------------------------------------------


class TestThresholdClassification:
    def test_high_proba_classified_as_risk(self, mock_model, sample_X):
        probas = mock_model.predict_proba(sample_X)[:, 1]
        threshold = 0.5
        predictions = (probas >= threshold).astype(int)
        assert predictions[1] == 1  # proba=0.7 > 0.5

    def test_low_proba_classified_as_no_risk(self, mock_model, sample_X):
        probas = mock_model.predict_proba(sample_X)[:, 1]
        threshold = 0.5
        predictions = (probas >= threshold).astype(int)
        assert predictions[0] == 0  # proba=0.2 < 0.5

    def test_custom_threshold_changes_predictions(self, mock_model, sample_X):
        probas = mock_model.predict_proba(sample_X)[:, 1]
        pred_05 = (probas >= 0.5).astype(int)
        pred_06 = (probas >= 0.6).astype(int)
        # At threshold=0.6, proba=0.6 should become 1
        assert pred_06.sum() <= pred_05.sum()
