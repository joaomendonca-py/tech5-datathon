"""
Feature engineering for the Passos Mágicos Datathon.

Creates derived features from the PEDE dataset:
- Trend features: INDE evolution across years
- Aggregation features: per-student statistics (mean, std, min, max)
- Interaction features: cross-variable combinations
- Phase-age gap: expected vs. actual school phase
- Feature importance: SHAP values
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import Config, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds INDE trend features when multi-year data is available.

    Creates:
    - INDE_DELTA: change in INDE between last two available years per student
    - INDE_TREND: direction (+1=improving, -1=worsening, 0=stable)

    Args:
        df: Input DataFrame. Must contain 'ANO' and an identifier column.

    Returns:
        pd.DataFrame: DataFrame with trend features added.
    """
    df = df.copy()

    if "ANO" not in df.columns or "INDE" not in df.columns:
        logger.warning("'ANO' or 'INDE' column not found. Skipping trend features.")
        df["INDE_DELTA"] = 0.0
        df["INDE_TREND"] = 0
        return df

    # Use NOME or row index as student identifier
    id_col = "NOME" if "NOME" in df.columns else None

    if id_col:
        sorted_df = df.sort_values([id_col, "ANO"])
        df["INDE_DELTA"] = sorted_df.groupby(id_col)["INDE"].diff().fillna(0)
    else:
        df["INDE_DELTA"] = df.groupby("ANO")["INDE"].transform(lambda x: x - x.shift(1)).fillna(0)

    df["INDE_TREND"] = np.sign(df["INDE_DELTA"]).astype(int)

    logger.info("Trend features added: INDE_DELTA, INDE_TREND")
    return df


# ---------------------------------------------------------------------------
# Aggregation features
# ---------------------------------------------------------------------------


def add_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aggregated statistics across educational indicators for each student.

    Creates per-row statistics (mean, std, min, max, range) across the
    indicator columns: IAA, IEG, IPS, IDA, IPP, IPV, IAN.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregation features added.
    """
    df = df.copy()

    indicator_cols = [c for c in ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"] if c in df.columns]

    if not indicator_cols:
        logger.warning("No indicator columns found. Skipping aggregation features.")
        return df

    df["INDICADORES_MEAN"] = df[indicator_cols].mean(axis=1)
    df["INDICADORES_STD"] = df[indicator_cols].std(axis=1).fillna(0)
    df["INDICADORES_MIN"] = df[indicator_cols].min(axis=1)
    df["INDICADORES_MAX"] = df[indicator_cols].max(axis=1)
    df["INDICADORES_RANGE"] = df["INDICADORES_MAX"] - df["INDICADORES_MIN"]

    # Engagement score: ratio of IPP (participation) to IDA (learning)
    if "IPP" in df.columns and "IDA" in df.columns:
        df["ENGAJAMENTO_APRENDIZADO"] = df["IPP"] / df["IDA"].replace(0, np.nan)
        df["ENGAJAMENTO_APRENDIZADO"] = df["ENGAJAMENTO_APRENDIZADO"].fillna(1.0)

    logger.info(f"Aggregation features added from {len(indicator_cols)} indicators.")
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interaction features between educational and socioeconomic indicators.

    Creates:
    - INDE_x_IPS: cognitive development × social-psychological support
    - IDA_x_IPP: learning × participation
    - IEG_x_IAN: general engagement × socioeconomic need

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with interaction features added.
    """
    df = df.copy()

    if "INDE" in df.columns and "IPS" in df.columns:
        df["INDE_x_IPS"] = df["INDE"] * df["IPS"]

    if "IDA" in df.columns and "IPP" in df.columns:
        df["IDA_x_IPP"] = df["IDA"] * df["IPP"]

    if "IEG" in df.columns and "IAN" in df.columns:
        df["IEG_x_IAN"] = df["IEG"] * df["IAN"]

    # Emotional vs. academic ratio
    if "IPS" in df.columns and "INDE" in df.columns:
        df["EMOCIONAL_ACADEMICO"] = df["IPS"] / df["INDE"].replace(0, np.nan)
        df["EMOCIONAL_ACADEMICO"] = df["EMOCIONAL_ACADEMICO"].fillna(1.0)

    logger.info("Interaction features added.")
    return df


# ---------------------------------------------------------------------------
# Phase-age gap
# ---------------------------------------------------------------------------

FASE_ORDER: dict[str, int] = {f"F{i}": i for i in range(1, 9)}
EXPECTED_FASE_BY_AGE: dict[int, str] = {
    6: "F1", 7: "F2", 8: "F3", 9: "F4", 10: "F5",
    11: "F6", 12: "F7", 13: "F7", 14: "F8", 15: "F8",
}


def add_phase_age_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the gap between expected and actual school phase based on age.

    A positive gap indicates the student is behind (older than expected for their phase).
    A negative gap means the student is ahead.

    Creates:
    - FASE_ESPERADA: expected phase for the student's age
    - FASE_GAP: difference (expected - actual) in phase levels

    Args:
        df: Input DataFrame with 'IDADE' and 'FASE' columns.

    Returns:
        pd.DataFrame: DataFrame with phase-age gap features added.
    """
    df = df.copy()

    if "IDADE" not in df.columns or "FASE" not in df.columns:
        logger.warning("'IDADE' or 'FASE' column not found. Skipping phase-age gap.")
        df["FASE_GAP"] = 0
        return df

    df["FASE_ESPERADA"] = df["IDADE"].map(
        lambda age: EXPECTED_FASE_BY_AGE.get(int(age), "F1") if pd.notna(age) else "F1"
    )

    expected_num = df["FASE_ESPERADA"].map(lambda f: FASE_ORDER.get(str(f), 1))
    actual_num = df["FASE"].map(lambda f: FASE_ORDER.get(str(f), 1))

    df["FASE_GAP"] = expected_num - actual_num
    df["DEFASAGEM_SEVERA"] = (df["FASE_GAP"] >= 2).astype(int)

    # Drop the temporary string helper column — not a model feature
    df = df.drop(columns=["FASE_ESPERADA"], errors="ignore")

    logger.info("Phase-age gap features added: FASE_GAP, DEFASAGEM_SEVERA")
    return df


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


def get_feature_importance(
    model: Any,
    X: pd.DataFrame,
    method: str = "shap",
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Computes feature importances using SHAP or built-in model importances.

    Args:
        model: Trained scikit-learn compatible model.
        X: Feature DataFrame (sample or full test set).
        method: 'shap' (preferred) or 'builtin' (uses model.feature_importances_).
        max_samples: Max rows to use for SHAP computation (for speed).

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    sample = X.sample(min(max_samples, len(X)), random_state=Config.RANDOM_STATE)

    if method == "shap":
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            # For binary classification, use class 1 SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            importance = np.abs(shap_values).mean(axis=0)
            df_imp = pd.DataFrame({"feature": X.columns, "importance": importance})
        except Exception as e:
            logger.warning(f"SHAP failed ({e}), falling back to built-in importance.")
            method = "builtin"

    if method == "builtin":
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model has no feature_importances_. Returning uniform importance.")
            df_imp = pd.DataFrame({"feature": X.columns, "importance": [1.0] * len(X.columns)})
        else:
            df_imp = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            )

    return df_imp.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline.

    Steps: trend → aggregation → interaction → phase-age gap.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        pd.DataFrame: DataFrame enriched with all engineered features.
    """
    logger.info("Running feature engineering pipeline...")
    df = add_trend_features(df)
    df = add_agg_features(df)
    df = add_interaction_features(df)
    df = add_phase_age_gap(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
