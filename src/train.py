"""
Model training pipeline for the Passos Mágicos Datathon.

Trains multiple classifiers, optimizes hyperparameters with Optuna,
selects the best model by Recall (primary metric), and saves it with joblib.

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- SVM

Validation strategy: StratifiedKFold (k=5)
Optimization: Optuna (or GridSearchCV fallback)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.evaluate import evaluate_model, print_reliability_justification
from src.feature_engineering import run_feature_engineering
from src.preprocessing import encode_categoricals, run_preprocessing_pipeline, scale_features
from src.utils import Config, ensure_dir, get_logger, get_timestamp, set_seed

# Suppress harmless convergence warnings during CV
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_PATH = Path(Config.MODEL_PATH)
CV = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)


# ---------------------------------------------------------------------------
# Baseline models (no tuning)
# ---------------------------------------------------------------------------


def get_baseline_models() -> dict[str, Any]:
    """
    Returns a dictionary of baseline scikit-learn-compatible classifiers.

    Returns:
        dict: Model name → classifier instance.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=Config.RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
        ),
        "SVM": SVC(
            probability=True,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------


def evaluate_with_cv(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "recall",
) -> float:
    """
    Evaluates a model using stratified cross-validation.

    Args:
        model: Scikit-learn compatible classifier.
        X_train: Training features.
        y_train: Training labels.
        scoring: Metric to optimize (default: 'recall').

    Returns:
        float: Mean cross-validation score.
    """
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring=scoring, n_jobs=-1)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------


def optimize_with_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = Config.N_TRIALS,
) -> tuple[Any, dict]:
    """
    Optimizes hyperparameters for a given model using Optuna.

    Args:
        model_name: Name of the model to optimize ('RandomForest', 'XGBoost', 'LightGBM').
        X_train: Training features.
        y_train: Training labels.
        n_trials: Number of Optuna trials.

    Returns:
        tuple: (best_model, best_params)
    """

    def objective(trial: optuna.Trial) -> float:
        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            model = RandomForestClassifier(
                **params,
                random_state=Config.RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            model = XGBClassifier(
                **params,
                random_state=Config.RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = LGBMClassifier(
                **params,
                random_state=Config.RANDOM_STATE,
                class_weight="balanced",
                verbosity=-1,
            )
        else:
            raise ValueError(f"Unsupported model for Optuna optimization: {model_name}")

        return evaluate_with_cv(model, X_train, y_train, scoring="recall")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"[{model_name}] Best params: {best_params} | Recall={study.best_value:.4f}")

    # Reconstruct best model with best params
    if model_name == "RandomForest":
        best_model = RandomForestClassifier(
            **best_params,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        best_model = XGBClassifier(
            **best_params,
            random_state=Config.RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )
    else:
        best_model = LGBMClassifier(
            **best_params,
            random_state=Config.RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
        )

    return best_model, best_params


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def train(raw_path: str | None = None) -> None:
    """
    Runs the full training pipeline.

    Steps:
    1. Preprocessing (load, clean, encode, scale, split)
    2. Feature engineering
    3. Baseline cross-validation for all models
    4. Hyperparameter optimization for top candidate
    5. Final model training and evaluation on test set
    6. Save best model with joblib

    Args:
        raw_path: Path to raw data directory (overrides Config.DATA_RAW_PATH).
    """
    start_time = time.time()
    set_seed()
    ts = get_timestamp()
    logger.info(f"=== Training pipeline started [{ts}] ===")

    # --- Preprocessing ---
    logger.info("Step 1/5: Preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test, artifacts = run_preprocessing_pipeline(raw_path)

    # --- Feature engineering ---
    # Runs BEFORE encoding so that IDADE and FASE are still raw values.
    # Row-level features (agg, interaction, phase-age gap) are safe to apply
    # independently on each split — no cross-row information leakage.
    logger.info("Step 2/5: Feature engineering...")
    X_train = run_feature_engineering(X_train)
    X_val   = run_feature_engineering(X_val)
    X_test  = run_feature_engineering(X_test)

    # Align columns across splits before encoding
    all_cols = list(X_train.columns)
    X_val  = X_val.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

    # --- Encoding & Scaling (after feature engineering, fit only on train) ---
    X_train, encoders = encode_categoricals(X_train)
    ohe = encoders.get("onehot")
    cat_cols_for_encoding = list(ohe.feature_names_in_) if ohe is not None else []

    def _apply_ohe(df: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted OneHotEncoder to val/test splits."""
        if ohe is None:
            return df
        existing = [c for c in cat_cols_for_encoding if c in df.columns]
        if not existing:
            return df
        encoded = ohe.transform(df[existing].astype(str))
        feat_names = ohe.get_feature_names_out(existing)
        df = df.drop(columns=existing)
        return pd.concat(
            [df, pd.DataFrame(encoded, columns=feat_names, index=df.index)], axis=1
        )

    X_val  = _apply_ohe(X_val)
    X_test = _apply_ohe(X_test)

    X_train, scaler = scale_features(X_train)
    X_val,  _       = scale_features(X_val,  scaler=scaler)
    X_test, _       = scale_features(X_test, scaler=scaler)

    artifacts = {"encoders": encoders, "scaler": scaler}

    # --- Baseline CV evaluation ---
    logger.info("Step 3/5: Baseline cross-validation...")
    models = get_baseline_models()
    cv_results: dict[str, float] = {}

    for name, model in models.items():
        score = evaluate_with_cv(model, X_train, y_train, scoring="recall")
        cv_results[name] = score
        logger.info(f"  [{name}] CV Recall = {score:.4f}")

    best_name = max(cv_results, key=cv_results.__getitem__)
    logger.info(f"Best baseline model: {best_name} (Recall={cv_results[best_name]:.4f})")

    # --- Hyperparameter optimization ---
    logger.info("Step 4/5: Hyperparameter optimization...")
    optimizable = {"RandomForest", "XGBoost", "LightGBM"}

    if Config.OPTIMIZE_HYPERPARAMS and best_name in optimizable:
        best_model, best_params = optimize_with_optuna(best_name, X_train, y_train)
    else:
        best_model = models[best_name]
        best_params = {}
        logger.info(f"Skipping Optuna for {best_name}. Using baseline params.")

    # --- Training on full train set and evaluation ---
    logger.info("Step 5/5: Training final model & evaluating on test set...")
    best_model.fit(X_train, y_train)

    # Validation set evaluation
    val_metrics = evaluate_model(best_model, X_val, y_val, save_plots=False)
    logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")

    # Test set evaluation (final)
    test_metrics = evaluate_model(best_model, X_test, y_test, save_plots=True)
    print_reliability_justification(test_metrics)

    # --- Save model ---
    ensure_dir(MODEL_SAVE_PATH.parent)
    model_metadata = {
        "model_name": best_name,
        "model_version": Config.MODEL_VERSION,
        "trained_at": ts,
        "training_time_seconds": round(time.time() - start_time, 2),
        "hyperparameters": best_params,
        "cv_results": {k: round(v, 4) for k, v in cv_results.items()},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_names": list(X_train.columns),
        "threshold": Config.MODEL_THRESHOLD,
    }

    joblib.dump(
        {"model": best_model, "metadata": model_metadata, "artifacts": artifacts},
        MODEL_SAVE_PATH,
    )
    logger.info(f"Model saved: {MODEL_SAVE_PATH}")

    # Save metadata as JSON for easy access
    meta_path = MODEL_SAVE_PATH.parent / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved: {meta_path}")

    elapsed = time.time() - start_time
    logger.info(f"=== Training complete in {elapsed:.1f}s ===")
    logger.info(f"Test Recall={test_metrics['recall']:.4f} | AUC-ROC={test_metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")  # headless backend before any import uses pyplot
    train()
    sys.exit(0)
