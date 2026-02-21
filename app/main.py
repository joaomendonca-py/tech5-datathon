"""
FastAPI application entry point for the Passos Mágicos ML API.

Loads the trained model at startup and exposes prediction endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from src.utils import Config, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

model_store: dict[str, Any] = {
    "model": None,
    "metadata": {},
    "artifacts": {},
    "loaded": False,
}


def load_model() -> None:
    """Loads the serialized model and metadata from disk."""
    model_path = Path(Config.MODEL_PATH)

    if not model_path.exists():
        logger.warning(
            f"Model file not found at '{model_path}'. "
            "API will start but /predict endpoints will return 503. "
            "Run 'make train' to train and save the model."
        )
        return

    try:
        bundle = joblib.load(model_path)
        model_store["model"] = bundle["model"]
        model_store["metadata"] = bundle.get("metadata", {})
        model_store["artifacts"] = bundle.get("artifacts", {})
        model_store["loaded"] = True
        logger.info(
            f"Model loaded successfully: {model_store['metadata'].get('model_name', 'unknown')} "
            f"v{model_store['metadata'].get('model_version', '?')}"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logger.info("Starting Passos Mágicos ML API...")
    load_model()
    yield
    logger.info("Shutting down Passos Mágicos ML API.")


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Passos Mágicos — API de Risco de Defasagem Escolar",
    description=(
        "API de Machine Learning para prever o risco de defasagem escolar "
        "de estudantes da Associação Passos Mágicos. "
        "Utiliza dados do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional)."
    ),
    version=Config.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"])
async def root() -> dict:
    """API root — redirects to /docs for interactive documentation."""
    return {
        "message": "Passos Mágicos — API de Risco de Defasagem Escolar",
        "docs": "/docs",
        "health": "/health",
        "version": Config.MODEL_VERSION,
    }
