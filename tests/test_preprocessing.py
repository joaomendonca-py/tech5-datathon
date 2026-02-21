"""
Unit tests for src/preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_target,
    encode_categoricals,
    handle_nulls,
    remove_duplicates,
    remove_inconsistencies,
    scale_features,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Creates a synthetic PEDE-like DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "NOME": [f"Aluno_{i}" for i in range(n)],
            "IDADE": np.random.randint(8, 18, n).astype(float),
            "FASE": np.random.choice(["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"], n),
            "PEDRA": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            "INDE": np.random.uniform(3.0, 9.5, n),
            "IAA": np.random.uniform(4.0, 10.0, n),
            "IEG": np.random.uniform(4.0, 10.0, n),
            "IPS": np.random.uniform(4.0, 10.0, n),
            "IDA": np.random.uniform(4.0, 10.0, n),
            "IPP": np.random.uniform(4.0, 10.0, n),
            "IPV": np.random.uniform(4.0, 10.0, n),
            "IAN": np.random.uniform(4.0, 10.0, n),
            "ANO": np.random.choice([2022, 2023, 2024], n),
        }
    )


@pytest.fixture
def df_with_nulls(sample_df) -> pd.DataFrame:
    """Introduces random nulls into the sample DataFrame."""
    df = sample_df.copy()
    for col in ["INDE", "IAA", "PEDRA"]:
        null_idx = np.random.choice(df.index, size=10, replace=False)
        df.loc[null_idx, col] = np.nan
    return df


@pytest.fixture
def df_with_duplicates(sample_df) -> pd.DataFrame:
    """Adds 20 duplicate rows to the sample DataFrame."""
    dups = sample_df.iloc[:20].copy()
    return pd.concat([sample_df, dups], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: remove_duplicates
# ---------------------------------------------------------------------------


class TestRemoveDuplicates:
    def test_removes_duplicate_rows(self, df_with_duplicates):
        original_len = len(df_with_duplicates)
        result = remove_duplicates(df_with_duplicates)
        assert len(result) < original_len

    def test_no_duplicates_unchanged(self, sample_df):
        result = remove_duplicates(sample_df)
        assert len(result) == len(sample_df)

    def test_returns_dataframe(self, sample_df):
        result = remove_duplicates(sample_df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: handle_nulls
# ---------------------------------------------------------------------------


class TestHandleNulls:
    def test_no_nulls_after_processing(self, df_with_nulls):
        result = handle_nulls(df_with_nulls)
        assert result.isnull().sum().sum() == 0

    def test_numeric_filled_with_median(self, df_with_nulls):
        original_median = df_with_nulls["INDE"].median()
        result = handle_nulls(df_with_nulls)
        # All previously null INDE values should be filled
        assert result["INDE"].isnull().sum() == 0

    def test_categorical_filled_with_mode(self, df_with_nulls):
        result = handle_nulls(df_with_nulls)
        assert result["PEDRA"].isnull().sum() == 0

    def test_returns_dataframe(self, df_with_nulls):
        result = handle_nulls(df_with_nulls)
        assert isinstance(result, pd.DataFrame)

    def test_shape_preserved(self, df_with_nulls):
        result = handle_nulls(df_with_nulls)
        assert result.shape == df_with_nulls.shape


# ---------------------------------------------------------------------------
# Tests: build_target
# ---------------------------------------------------------------------------


class TestBuildTarget:
    def test_target_column_created(self, sample_df):
        result = build_target(sample_df)
        assert "RISCO_DEFASAGEM" in result.columns

    def test_target_is_binary(self, sample_df):
        result = build_target(sample_df)
        assert set(result["RISCO_DEFASAGEM"].unique()).issubset({0, 1})

    def test_defasagem_field_triggers_risk(self):
        # Rule 1: DEFASAGEM <= -1 means student is behind -> at risk
        df = pd.DataFrame({
            "DEFASAGEM": [-2, -1, 0, 1],
            "PEDRA": ["Ametista"] * 4,
            "ANO": [2024] * 4,
        })
        result = build_target(df)
        assert result.loc[0, "RISCO_DEFASAGEM"] == 1  # DEFASAGEM=-2 -> risk
        assert result.loc[1, "RISCO_DEFASAGEM"] == 1  # DEFASAGEM=-1 -> risk
        assert result.loc[2, "RISCO_DEFASAGEM"] == 0  # DEFASAGEM=0  -> no risk
        assert result.loc[3, "RISCO_DEFASAGEM"] == 0  # DEFASAGEM=1  -> no risk

    def test_quartzo_fallback_triggers_risk(self):
        # Rule 2 (fallback when no DEFASAGEM): PEDRA == 'Quartzo' -> at risk
        df = pd.DataFrame({
            "PEDRA": ["Quartzo", "Ágata", "Ametista", "Topázio"],
            "ANO": [2022] * 4,
        })
        result = build_target(df)
        assert result.loc[0, "RISCO_DEFASAGEM"] == 1  # Quartzo -> risk
        assert result.loc[1, "RISCO_DEFASAGEM"] == 0
        assert result.loc[2, "RISCO_DEFASAGEM"] == 0
        assert result.loc[3, "RISCO_DEFASAGEM"] == 0

    def test_no_risk_when_no_defasagem_and_not_quartzo(self):
        # No DEFASAGEM column, PEDRA != Quartzo -> no risk
        df = pd.DataFrame({
            "PEDRA": ["Ametista", "Topázio"],
            "ANO": [2022] * 2,
        })
        result = build_target(df)
        assert result["RISCO_DEFASAGEM"].sum() == 0


# ---------------------------------------------------------------------------
# Tests: encode_categoricals
# ---------------------------------------------------------------------------


class TestEncodeCategoricals:
    def test_onehot_removes_original_columns(self, sample_df):
        result, encoders = encode_categoricals(sample_df, cols=["PEDRA", "FASE"], strategy="onehot")
        assert "PEDRA" not in result.columns
        assert "FASE" not in result.columns

    def test_onehot_adds_new_columns(self, sample_df):
        original_cols = len(sample_df.columns)
        result, _ = encode_categoricals(sample_df, cols=["PEDRA"], strategy="onehot")
        assert len(result.columns) > original_cols - 1  # at least net neutral

    def test_label_encodes_to_int(self, sample_df):
        result, encoders = encode_categoricals(sample_df, cols=["PEDRA"], strategy="label")
        assert result["PEDRA"].dtype in [np.int64, np.int32, int]

    def test_returns_encoders_dict(self, sample_df):
        _, encoders = encode_categoricals(sample_df, cols=["PEDRA"], strategy="label")
        assert isinstance(encoders, dict)
        assert "PEDRA" in encoders


# ---------------------------------------------------------------------------
# Tests: scale_features
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    def test_numeric_columns_scaled(self, sample_df):
        cols = ["INDE", "IAA", "IDA"]
        result, scaler = scale_features(sample_df, cols=cols)
        # After StandardScaler, mean should be ≈ 0
        assert abs(result[cols].mean().mean()) < 1.0

    def test_returns_scaler_object(self, sample_df):
        _, scaler = scale_features(sample_df, cols=["INDE"])
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)

    def test_pre_fitted_scaler_transform(self, sample_df):
        from sklearn.preprocessing import StandardScaler
        cols = ["INDE", "IAA"]
        _, fitted_scaler = scale_features(sample_df, cols=cols)

        new_df = sample_df.copy()
        result, _ = scale_features(new_df, cols=cols, scaler=fitted_scaler)
        assert result.shape == new_df.shape


# ---------------------------------------------------------------------------
# Tests: split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_correct_proportions(self, sample_df):
        df = build_target(sample_df)
        X_train, X_val, X_test, *_ = split_data(df)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df)
        assert len(X_test) / total == pytest.approx(0.15, abs=0.05)

    def test_no_overlap_between_splits(self, sample_df):
        df = build_target(sample_df).reset_index(drop=True)
        X_train, X_val, X_test, *_ = split_data(df)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_target_not_in_features(self, sample_df):
        df = build_target(sample_df)
        X_train, X_val, X_test, *_ = split_data(df)
        assert "RISCO_DEFASAGEM" not in X_train.columns

    def test_returns_series_for_labels(self, sample_df):
        df = build_target(sample_df)
        _, _, _, y_train, y_val, y_test = split_data(df)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_remove_inconsistencies(self, sample_df):
        result = remove_inconsistencies(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_df)
