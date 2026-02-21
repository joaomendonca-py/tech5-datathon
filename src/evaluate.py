"""
Model evaluation for the Passos Mágicos Datathon.

Computes and reports:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Classification report
- Confusion matrix
- ROC curve
- Precision-Recall curve

Metric rationale:
    The primary metric is **Recall** (Sensitivity). In the context of student
    risk prediction, it is more costly to miss a student who is truly at risk
    (False Negative) than to flag a student who is not at risk (False Positive).
    A high Recall ensures we capture as many at-risk students as possible,
    enabling timely pedagogical intervention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import Config, ensure_dir, get_logger, get_timestamp

logger = get_logger(__name__)

REPORTS_DIR = Path("monitoring/reports")

# Suppress common non-critical warnings
import warnings
warnings.filterwarnings("ignore", message=".*Matplotlib.*")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = Config.MODEL_THRESHOLD,
    save_plots: bool = True,
) -> dict[str, Any]:
    """
    Evaluates a trained model on the test set and generates all metrics and plots.

    Args:
        model: Trained scikit-learn compatible model with predict_proba().
        X_test: Test features.
        y_test: True test labels.
        threshold: Classification threshold for positive class.
        save_plots: Whether to save ROC, PR, and confusion matrix plots.

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "threshold": threshold,
        "n_samples": len(y_test),
        "n_positive": int(y_test.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }

    logger.info(
        f"Evaluation results: Accuracy={metrics['accuracy']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1_score']:.4f} | AUC-ROC={metrics['auc_roc']:.4f}"
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Sem Risco", "Em Risco"]))

    if save_plots:
        ts = get_timestamp()
        ensure_dir(REPORTS_DIR)
        _plot_confusion_matrix(y_test, y_pred, ts)
        _plot_roc_curve(y_test, y_proba, metrics["auc_roc"], ts)
        _plot_precision_recall(y_test, y_proba, ts)

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, ts: str) -> None:
    """Saves the confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sem Risco", "Em Risco"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusão — Risco de Defasagem", fontsize=13)
    plt.tight_layout()
    path = REPORTS_DIR / f"confusion_matrix_{ts}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def _plot_roc_curve(
    y_true: pd.Series, y_proba: np.ndarray, auc_score: float, ts: str
) -> None:
    """Saves the ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Linha de base")
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)", fontsize=12)
    ax.set_title("Curva ROC — Risco de Defasagem Escolar", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = REPORTS_DIR / f"roc_curve_{ts}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {path}")


def _plot_precision_recall(y_true: pd.Series, y_proba: np.ndarray, ts: str) -> None:
    """Saves the Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#DD8452", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Linha de base ({baseline:.2f})")
    ax.set_xlabel("Recall (Sensibilidade)", fontsize=12)
    ax.set_ylabel("Precisão", fontsize=12)
    ax.set_title("Curva Precision-Recall — Risco de Defasagem", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / f"precision_recall_{ts}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Precision-Recall curve saved: {path}")


# ---------------------------------------------------------------------------
# Model reliability justification
# ---------------------------------------------------------------------------


def print_reliability_justification(metrics: dict[str, Any]) -> None:
    """
    Prints a structured justification of why the model is reliable for production.

    Args:
        metrics: Dictionary returned by evaluate_model().
    """
    print("\n" + "=" * 60)
    print("MODEL RELIABILITY JUSTIFICATION")
    print("=" * 60)
    print(
        f"""
Primary Metric: Recall = {metrics.get('recall', 'N/A')}

Rationale:
  In the context of predicting student risk of educational lag,
  the cost of a False Negative (missing an at-risk student) far
  outweighs the cost of a False Positive (unnecessary intervention).
  Therefore, Recall is prioritized as the primary metric.

  A high Recall ensures that the model identifies the maximum number
  of students requiring pedagogical attention, enabling timely support
  from Passos Mágicos' educators and psychopedagogists.

Additional Metrics:
  - AUC-ROC = {metrics.get('auc_roc', 'N/A')} (discriminative power across all thresholds)
  - F1-Score = {metrics.get('f1_score', 'N/A')} (balance between precision and recall)
  - Accuracy = {metrics.get('accuracy', 'N/A')}

Validation:
  - Stratified K-Fold cross-validation (k={Config.CV_FOLDS}) ensures
    robustness across different data splits.
  - Hyperparameter optimization reduces overfitting risk.
  - Model is evaluated on a held-out test set never seen during training.
"""
    )
