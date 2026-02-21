"""
Data preprocessing pipeline for the Passos Mágicos Datathon.

Provides modular, testable functions for:
- Loading raw data (2022, 2023, 2024)
- Cleaning: duplicates, inconsistencies
- Null handling: median (numeric) / mode (categorical)
- Encoding: OneHotEncoder / LabelEncoder
- Scaling: StandardScaler
- Target variable construction: RISCO_DEFASAGEM
- Train/validation/test split (70/15/15 stratified)
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils import Config, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected columns per PEDE dataset
NUMERIC_COLS = [
    "INDE",
    "IAA",
    "IEG",
    "IPS",
    "IDA",
    "IPP",
    "IPV",
    "IAN",
    "ANOS_PM",
    "IDADE",
]

CATEGORICAL_COLS = [
    "PEDRA",
    "FASE",
    "GENERO",
    "INSTITUICAO",
]

# Mapping: PEDRA category → numeric risk level
PEDRA_ORDER = {
    "Quartzo": 0,
    "Ágata": 1,
    "Ametista": 2,
    "Topázio": 3,
}

# Expected phase (FASE) for each age group
FASE_BY_AGE: dict[int, str] = {
    6: "F1",
    7: "F2",
    8: "F3",
    9: "F4",
    10: "F5",
    11: "F6",
    12: "F7",
    13: "F7",
    14: "F8",
    15: "F8",
}

FASE_ORDER = {f"F{i}": i for i in range(1, 9)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Loads the PEDE dataset.

    Priority:
    1. If path points to a specific file, load it directly.
    2. If data/processed/pede_unified.csv exists (ETL output), use it.
    3. Fallback: scan raw directory for CSV/XLSX files.

    Args:
        path: File or directory path. Defaults to Config.DATA_PROCESSED_PATH.

    Returns:
        pd.DataFrame: Combined dataset with an 'ANO' column.

    Raises:
        FileNotFoundError: If no data files are found.
    """
    # 1. Explicit file path
    if path and Path(path).is_file():
        logger.info(f"Loading file directly: {path}")
        p = Path(path)
        df = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p, encoding="utf-8-sig", sep=None, engine="python")
        logger.info(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    # 2. Prefer processed unified CSV
    processed_unified = Path(Config.DATA_PROCESSED_PATH) / "pede_unified.csv"
    if processed_unified.exists():
        logger.info(f"Loading unified dataset: {processed_unified}")
        df = pd.read_csv(processed_unified, encoding="utf-8-sig")
        logger.info(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    # 3. Fallback: raw directory
    raw_path = Path(path or Config.DATA_RAW_PATH)
    csv_files = list(raw_path.glob("*.csv")) + list(raw_path.glob("*.xlsx"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV/XLSX files found in '{raw_path}'. "
            "Please add your data files to data/raw/ before running."
        )

    dfs = []
    for f in sorted(csv_files):
        logger.info(f"Loading file: {f.name}")
        if f.suffix == ".xlsx":
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f, encoding="utf-8-sig", sep=None, engine="python")

        year_candidates = [p for p in f.stem.split("_") if p.isdigit() and len(p) == 4]
        if year_candidates:
            df["ANO"] = int(year_candidates[-1])

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Dataset loaded: {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed:,} duplicate rows.")
    return df


def remove_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with invalid/inconsistent data.

    Validation rules:
    - Scores (INDE, IAA, etc.) must be in range [0, 10]
    - IDADE (if present) must be > 0

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    score_cols = [c for c in NUMERIC_COLS if c in df.columns and c.startswith("I")]
    for col in score_cols:
        mask &= df[col].between(0, 10) | df[col].isna()

    if "IDADE" in df.columns:
        mask &= (df["IDADE"] > 0) | df["IDADE"].isna()

    removed = (~mask).sum()
    if removed:
        logger.info(f"Removed {removed:,} inconsistent rows.")

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.

    Strategy:
    - Columns 100% null: dropped (not useful for modeling)
    - Numeric columns: fill with column median
    - Categorical columns: fill with column mode

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with no missing values in known columns.
    """
    df = df.copy()

    # Drop columns that are 100% null (no signal)
    all_null_cols = [c for c in df.columns if df[c].isnull().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        logger.info(f"Dropped {len(all_null_cols)} all-null columns: {all_null_cols}")

    # Drop columns with >90% null values (insufficient signal)
    high_null_cols = [c for c in df.columns if df[c].isnull().mean() > 0.90]
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
        logger.info(f"Dropped {len(high_null_cols)} high-null (>90%) columns: {high_null_cols}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
            logger.debug(f"Filled '{col}' nulls with median={median_val}")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isna().any():
            mode_series = df[col].mode()
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series[0])
                logger.debug(f"Filled '{col}' nulls with mode='{mode_series[0]}'")

    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------


def build_target(df: pd.DataFrame, threshold: float = 5.5) -> pd.DataFrame:
    """
    Constructs the binary target variable 'RISCO_DEFASAGEM'.

    Priority (in order — NO data leakage with model features):

    1. DEFASAGEM field (official dataset): student behind their expected level.
       DEFASAGEM <= -1 means the student is at least 1 phase behind → at risk.

    2. PONTO_VIRADA == 0: did NOT reach the turning point → potential risk (secondary).

    3. Fallback (when DEFASAGEM missing, e.g. year 2020):
       PEDRA == 'Quartzo' (lowest category, <4.0 INDE equivalent) → at risk.

    NOTE: INDE is intentionally NOT used here to avoid data leakage
    (INDE would be a feature AND the target signal simultaneously).

    Args:
        df: Input DataFrame with DEFASAGEM, PONTO_VIRADA, and optionally PEDRA.
        threshold: Unused in main logic (kept for API compatibility).

    Returns:
        pd.DataFrame: DataFrame with 'RISCO_DEFASAGEM' column added.
    """
    df = df.copy()
    df["RISCO_DEFASAGEM"] = 0

    # Rule 1: Official DEFASAGEM field — student behind their expected phase
    # Negative values mean "atrás do nível ideal" (behind ideal level)
    if "DEFASAGEM" in df.columns:
        defasagem_num = pd.to_numeric(df["DEFASAGEM"], errors="coerce")
        # <= -1: at least one phase behind → at risk
        df.loc[defasagem_num <= -1, "RISCO_DEFASAGEM"] = 1
        has_defasagem = defasagem_num.notna()
    else:
        has_defasagem = pd.Series(False, index=df.index)

    # Rule 2 (fallback for rows without DEFASAGEM): PEDRA == Quartzo only
    # (Quartzo ≈ INDE < 5.5, the worst performance category)
    if "PEDRA" in df.columns:
        no_defasagem_mask = ~has_defasagem
        df.loc[no_defasagem_mask & (df["PEDRA"] == "Quartzo"), "RISCO_DEFASAGEM"] = 1

    risk_pct = df["RISCO_DEFASAGEM"].mean() * 100
    n_risk = df["RISCO_DEFASAGEM"].sum()
    logger.info(
        f"Target built: {n_risk:,} at-risk students ({risk_pct:.1f}% of {len(df):,} total)."
    )
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_categoricals(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    strategy: str = "onehot",
) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns.

    Args:
        df: Input DataFrame.
        cols: List of categorical columns to encode. Defaults to CATEGORICAL_COLS present in df.
        strategy: 'onehot' or 'label'.

    Returns:
        tuple[pd.DataFrame, dict]: Encoded DataFrame and a dict of fitted encoders.
    """
    df = df.copy()
    cols = cols or [c for c in CATEGORICAL_COLS if c in df.columns]
    encoders: dict = {}

    if strategy == "onehot":
        # Define all known categories explicitly to avoid vocabulary gaps between splits
        KNOWN_CATEGORIES = {
            "PEDRA": ["Ametista", "Desconhecido", "Quartzo", "Topázio", "Ágata"],
            "FASE": ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
            "GENERO": ["MENINA", "MENINO"],
            "INSTITUICAO": [
                "ESCOLA JP II", "ESCOLA JOÃO PAULO II", "ESCOLA PÚBLICA",
                "REDE DECISÃO", "REDE DECISÃO/UNIÃO", "EINSTEIN",
                "ESTÁCIO", "UNISA", "FIAP",
            ],
        }
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            categories = [KNOWN_CATEGORIES.get(c, "auto") for c in existing_cols]
            # If any col has no known mapping, fall back to "auto"
            if any(c == "auto" for c in categories):
                categories = "auto"
            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop="first",
                categories=categories,
            )
            encoded = enc.fit_transform(df[existing_cols].astype(str))
            feat_names = enc.get_feature_names_out(existing_cols)
            df = df.drop(columns=existing_cols)
            df = pd.concat([df, pd.DataFrame(encoded, columns=feat_names, index=df.index)], axis=1)
            encoders["onehot"] = enc
    else:
        for col in cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    return df, encoders


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def scale_features(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardizes numeric features using StandardScaler.

    Args:
        df: Input DataFrame.
        cols: Columns to scale. Defaults to all numeric columns (excluding target).
        scaler: Pre-fitted scaler (for transform-only use on test set).

    Returns:
        tuple[pd.DataFrame, StandardScaler]: Scaled DataFrame and the fitted scaler.
    """
    df = df.copy()
    exclude = {"RISCO_DEFASAGEM", "ANO", "NOME", "INSTITUICAO_ENSINO_ALUNO"}
    if cols is None:
        cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    target_col: str = "RISCO_DEFASAGEM",
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into train, validation, and test sets (stratified).

    Default split: 70% train / 15% validation / 15% test.

    Args:
        df: Input DataFrame with target column.
        target_col: Name of the target column.
        test_size: Proportion for the test set.
        val_size: Proportion for the validation set (from remaining after test split).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=Config.RANDOM_STATE,
        stratify=y,
    )

    # val_size relative to train+val set
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=Config.RANDOM_STATE,
        stratify=y_train_val,
    )

    logger.info(
        f"Split: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,} | "
        f"Target rate: train={y_train.mean():.2%}, val={y_val.mean():.2%}, test={y_test.mean():.2%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    raw_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Executes the complete preprocessing pipeline.

    Steps: load → clean → handle nulls → build target → split →
           feature_engineering (in train.py) → encode → scale.

    NOTE: Encoding and scaling are intentionally deferred to train.py so that
    feature engineering (which needs IDADE and FASE as raw values) runs first.
    This avoids leaking statistics from val/test into the encoder fit.

    Args:
        raw_path: Path to raw data directory.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, artifacts)
        where artifacts is an empty dict at this stage (filled in train.py after FE).
    """
    df = load_data(raw_path)
    df = remove_duplicates(df)
    df = remove_inconsistencies(df)
    df = handle_nulls(df)
    df = build_target(df)

    # Drop identifier and leaky columns before ML
    id_cols = ["NOME", "NIVEL_IDEAL", "DEFASAGEM", "ANO_NASC", "ANO_INGRESSO", "TURMA"]
    cols_to_drop = [c for c in id_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped non-feature columns: {cols_to_drop}")

    # Split here so feature engineering and encoding run per-split in train.py
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    artifacts: dict = {}
    return X_train, X_val, X_test, y_train, y_val, y_test, artifacts
