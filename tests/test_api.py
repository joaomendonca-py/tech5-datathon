"""
Unit tests for the FastAPI endpoints.
Uses TestClient to test all routes without running a real server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture with mocked model
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def model_store_loaded(mock_model):
    return {
        "model": mock_model,
        "loaded": True,
        "metadata": {
            "model_name": "RandomForest",
            "model_version": "1.0.0",
            "trained_at": "20240101_120000",
            "training_time_seconds": 42.0,
            "test_metrics": {"recall": 0.88, "f1_score": 0.84, "auc_roc": 0.91},
            "hyperparameters": {"n_estimators": 200},
            "feature_names": ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV"],
            "threshold": 0.5,
        },
        "artifacts": {},
    }


@pytest.fixture
def model_store_empty():
    return {"model": None, "loaded": False, "metadata": {}, "artifacts": {}}


@pytest.fixture
def client(model_store_loaded):
    """TestClient with loaded model."""
    with patch("app.routes._get_model_store", return_value=model_store_loaded):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_no_model(model_store_empty):
    """TestClient without a loaded model."""
    with patch("app.routes._get_model_store", return_value=model_store_empty):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------


VALID_STUDENT = {
    "IDADE": 14,
    "FASE": "F6",
    "PEDRA": "Ametista",
    "IAA": 7.5,
    "IEG": 6.8,
    "IPS": 8.0,
    "IDA": 7.2,
    "IPP": 6.5,
    "IPV": 7.0,
    "IAN": 6.9,
    "INDE": 7.1,
}

INVALID_STUDENT_MISSING_FIELD = {
    "IDADE": 14,
    "FASE": "F6",
    # Missing PEDRA and all indicators
}

INVALID_STUDENT_OUT_OF_RANGE = {
    **VALID_STUDENT,
    "INDE": 15.0,  # > 10, invalid
}


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    def test_health_shows_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_model_not_loaded(self, client_no_model):
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Tests: GET /model-info
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_model_name(self, client):
        response = client.get("/model-info")
        data = response.json()
        assert "model_name" in data

    def test_model_info_has_metrics(self, client):
        response = client.get("/model-info")
        data = response.json()
        assert "test_metrics" in data

    def test_model_info_503_when_no_model(self, client_no_model):
        response = client_no_model.get("/model-info")
        assert response.status_code == 503

    def test_model_info_has_feature_names(self, client):
        response = client.get("/model-info")
        data = response.json()
        assert "feature_names" in data
        assert isinstance(data["feature_names"], list)


# ---------------------------------------------------------------------------
# Tests: POST /predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        assert response.status_code == 200

    def test_predict_has_student_risk_field(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        data = response.json()
        assert "student_risk" in data

    def test_predict_risk_is_alto_or_baixo(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        data = response.json()
        assert data["student_risk"] in {"alto", "baixo"}

    def test_predict_has_probability(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_has_recommendation(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        data = response.json()
        assert "recommendation" in data
        assert len(data["recommendation"]) > 0

    def test_predict_has_risk_score(self, client):
        response = client.post("/predict", json=VALID_STUDENT)
        data = response.json()
        assert "risk_score" in data
        assert data["risk_score"] in {0, 1}

    def test_predict_missing_field_returns_422(self, client):
        response = client.post("/predict", json=INVALID_STUDENT_MISSING_FIELD)
        assert response.status_code == 422

    def test_predict_out_of_range_returns_422(self, client):
        response = client.post("/predict", json=INVALID_STUDENT_OUT_OF_RANGE)
        assert response.status_code == 422

    def test_predict_503_when_no_model(self, client_no_model):
        response = client_no_model.post("/predict", json=VALID_STUDENT)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Tests: POST /predict/batch
# ---------------------------------------------------------------------------


class TestPredictBatchEndpoint:
    def test_batch_returns_200(self, client):
        payload = {"students": [VALID_STUDENT, VALID_STUDENT]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_total_matches_input(self, client):
        students = [VALID_STUDENT] * 3
        response = client.post("/predict/batch", json={"students": students})
        data = response.json()
        assert data["total"] == 3

    def test_batch_has_results_list(self, client):
        response = client.post("/predict/batch", json={"students": [VALID_STUDENT]})
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1

    def test_batch_risk_counts_sum_to_total(self, client):
        students = [VALID_STUDENT] * 4
        response = client.post("/predict/batch", json={"students": students})
        data = response.json()
        assert data["high_risk_count"] + data["low_risk_count"] == data["total"]

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"students": []})
        assert response.status_code == 422

    def test_batch_503_when_no_model(self, client_no_model):
        payload = {"students": [VALID_STUDENT]}
        response = client_no_model.post("/predict/batch", json=payload)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Tests: GET /
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self, client):
        response = client.get("/")
        data = response.json()
        assert "docs" in data
