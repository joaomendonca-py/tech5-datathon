"""
API routes for the Passos Mágicos ML API.

Endpoints:
- GET  /health         — API health check
- GET  /model-info     — Model version, metrics, training date
- POST /predict        — Single student risk prediction
- POST /predict/batch  — Batch prediction for multiple students
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.feature_engineering import run_feature_engineering
from src.utils import Config, get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class StudentFeatures(BaseModel):
    """Input features for a single student's risk prediction."""

    IDADE: float = Field(..., ge=5, le=25, description="Idade do estudante (anos)")
    FASE: str = Field(..., description="Fase atual do aluno (ex: F1 a F8)")
    PEDRA: str = Field(..., description="Categoria de desempenho (Quartzo, Ágata, Ametista, Topázio)")
    GENERO: str = Field(default="MENINA", description="Gênero do aluno (MENINO ou MENINA)")
    INSTITUICAO: str = Field(default="ESCOLA PÚBLICA", description="Instituição de ensino do aluno")
    IAA: float = Field(..., ge=0, le=10, description="Indicador de Autoavaliação do Aluno")
    IEG: float = Field(..., ge=0, le=10, description="Indicador de Engajamento")
    IPS: float = Field(..., ge=0, le=10, description="Indicador Psicossocial")
    IDA: float = Field(..., ge=0, le=10, description="Indicador de Aprendizagem")
    IPP: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Partida (Pedagógico)")
    IPV: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Virada")
    IAN: float = Field(..., ge=0, le=10, description="Indicador de Adequação ao Nível")
    INDE: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional")

    model_config = {"json_schema_extra": {
        "example": {
            "IDADE": 14, "FASE": "F6", "PEDRA": "Ametista",
            "GENERO": "MENINA", "INSTITUICAO": "ESCOLA PÚBLICA",
            "IAA": 7.5, "IEG": 6.8, "IPS": 8.0, "IDA": 7.2,
            "IPP": 6.5, "IPV": 7.0, "IAN": 6.9, "INDE": 7.1,
        }
    }}


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    student_risk: str = Field(..., description="Nível de risco: 'alto' ou 'baixo'")
    probability: float = Field(..., description="Probabilidade de risco (0.0 a 1.0)")
    risk_score: int = Field(..., description="Classe predita: 1=em risco, 0=sem risco")
    recommendation: str = Field(..., description="Recomendação pedagógica baseada no risco")
    threshold_used: float = Field(..., description="Threshold aplicado na classificação")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    students: list[StudentFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    total: int
    high_risk_count: int
    low_risk_count: int
    results: list[PredictionResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_model_store() -> dict[str, Any]:
    """Imports and returns the model store from main module."""
    from app.main import model_store
    return model_store


def _get_recommendation(risk_score: int, probability: float) -> str:
    """Returns a pedagogical recommendation based on risk level."""
    if risk_score == 1:
        if probability >= 0.8:
            return "⚠️ Acompanhamento pedagógico URGENTE recomendado. Acionar equipe psicopedagógica."
        return "📋 Acompanhamento pedagógico prioritário recomendado. Monitoramento mensal."
    return "✅ Estudante dentro do esperado. Manter acompanhamento regular."


def _predict_single(features: StudentFeatures, model_store: dict) -> PredictionResponse:
    """Runs prediction for a single student."""
    model = model_store["model"]
    threshold = float(model_store["metadata"].get("threshold", Config.MODEL_THRESHOLD))
    artifacts = model_store.get("artifacts", {})

    input_dict = features.model_dump()
    df = pd.DataFrame([input_dict])

    # Add columns expected by feature engineering that may not be in the API schema
    if "ANO" not in df.columns:
        df["ANO"] = 2024  # current year as default
    if "PONTO_VIRADA" not in df.columns:
        df["PONTO_VIRADA"] = 0  # conservative default
    if "ANOS_PM" not in df.columns:
        df["ANOS_PM"] = 1  # first year as default

    # Run the same feature engineering used during training
    df = run_feature_engineering(df)

    # Apply the same OneHotEncoder used during training (stored in artifacts)
    onehot_encoder = artifacts.get("encoders", {}).get("onehot")
    if onehot_encoder is not None:
        cat_cols = [c for c in onehot_encoder.feature_names_in_ if c in df.columns]
        if cat_cols:
            encoded = onehot_encoder.transform(df[cat_cols].astype(str))
            feat_names = onehot_encoder.get_feature_names_out(cat_cols)
            df = df.drop(columns=cat_cols)
            df = pd.concat(
                [df, pd.DataFrame(encoded, columns=feat_names, index=df.index)],
                axis=1,
            )
    else:
        # Fallback: label-encode categoricals if no fitted encoder is available
        cat_cols = [c for c in ["FASE", "PEDRA"] if c in df.columns]
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes

    # Apply the same scaler used during training (only on the columns it knows)
    scaler = artifacts.get("scaler")
    if scaler is not None:
        scaler_cols = list(getattr(scaler, "feature_names_in_", []))
        if scaler_cols:
            # Align to scaler columns, fill missing with 0
            df_scale = df.reindex(columns=scaler_cols, fill_value=0)
            scaled = scaler.transform(df_scale)
            df_scaled = pd.DataFrame(scaled, columns=scaler_cols, index=df.index)
            # Re-add any columns not seen by the scaler (e.g. ANO)
            for col in df.columns:
                if col not in df_scaled.columns:
                    df_scaled[col] = df[col]
            df = df_scaled

    # Align to the exact feature set the model was trained on
    feature_names = model_store["metadata"].get("feature_names", [])
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    proba = float(model.predict_proba(df)[0][1])
    risk_score = int(proba >= threshold)

    return PredictionResponse(
        student_risk="alto" if risk_score == 1 else "baixo",
        probability=round(proba, 4),
        risk_score=risk_score,
        recommendation=_get_recommendation(risk_score, proba),
        threshold_used=threshold,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", tags=["Monitoring"])
async def health_check() -> dict:
    """
    API health check endpoint.

    Returns:
        JSON with status, timestamp, and model availability.
    """
    model_store = _get_model_store()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": model_store["loaded"],
        "version": Config.MODEL_VERSION,
        "environment": Config.APP_ENV,
    }


@router.get("/model-info", tags=["Monitoring"])
async def model_info() -> dict:
    """
    Returns metadata about the currently loaded model.

    Returns:
        JSON with model name, version, metrics, training date, and hyperparameters.
    """
    model_store = _get_model_store()

    if not model_store["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run 'make train' to train and save the model.",
        )

    return {
        "model_name": model_store["metadata"].get("model_name", "unknown"),
        "model_version": model_store["metadata"].get("model_version", Config.MODEL_VERSION),
        "trained_at": model_store["metadata"].get("trained_at"),
        "training_time_seconds": model_store["metadata"].get("training_time_seconds"),
        "test_metrics": model_store["metadata"].get("test_metrics", {}),
        "hyperparameters": model_store["metadata"].get("hyperparameters", {}),
        "feature_names": model_store["metadata"].get("feature_names", []),
        "threshold": model_store["metadata"].get("threshold", Config.MODEL_THRESHOLD),
    }


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentFeatures) -> PredictionResponse:
    """
    Predicts the school lag risk for a single student.

    Args:
        student: StudentFeatures — all PEDE indicators for one student.

    Returns:
        PredictionResponse with risk level, probability, and recommendation.
    """
    model_store = _get_model_store()

    if not model_store["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Run 'make train' to generate the model.",
        )

    try:
        result = _predict_single(student, model_store)
        logger.info(
            f"Prediction: risk={result.student_risk} | proba={result.probability:.4f} | "
            f"INDE={student.INDE} | FASE={student.FASE}"
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predicts school lag risk for a batch of students.

    Args:
        request: BatchPredictionRequest with a list of StudentFeatures.

    Returns:
        BatchPredictionResponse with aggregated counts and individual predictions.
    """
    model_store = _get_model_store()

    if not model_store["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Run 'make train' to generate the model.",
        )

    results = []
    for student in request.students:
        try:
            result = _predict_single(student, model_store)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error for student: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}",
            )

    high_risk = sum(1 for r in results if r.risk_score == 1)
    logger.info(f"Batch prediction: {len(results)} students | {high_risk} at risk.")

    return BatchPredictionResponse(
        total=len(results),
        high_risk_count=high_risk,
        low_risk_count=len(results) - high_risk,
        results=results,
    )
