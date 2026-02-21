"""
Unit tests for src/feature_engineering.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_agg_features,
    add_interaction_features,
    add_phase_age_gap,
    add_trend_features,
    run_feature_engineering,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "NOME": [f"Aluno_{i}" for i in range(n)],
            "IDADE": np.random.randint(8, 16, n).astype(float),
            "FASE": np.random.choice(["F2", "F3", "F4", "F5", "F6", "F7"], n),
            "PEDRA": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
            "INDE": np.random.uniform(4.0, 9.5, n),
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


# ---------------------------------------------------------------------------
# Tests: add_trend_features
# ---------------------------------------------------------------------------


class TestAddTrendFeatures:
    def test_creates_inde_delta_column(self, sample_df):
        result = add_trend_features(sample_df)
        assert "INDE_DELTA" in result.columns

    def test_creates_inde_trend_column(self, sample_df):
        result = add_trend_features(sample_df)
        assert "INDE_TREND" in result.columns

    def test_trend_values_are_integers(self, sample_df):
        result = add_trend_features(sample_df)
        assert result["INDE_TREND"].dtype in [np.int64, np.int32, int]

    def test_trend_values_valid_range(self, sample_df):
        result = add_trend_features(sample_df)
        assert set(result["INDE_TREND"].unique()).issubset({-1, 0, 1})

    def test_returns_dataframe(self, sample_df):
        result = add_trend_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_shape_unchanged(self, sample_df):
        result = add_trend_features(sample_df)
        assert result.shape[0] == sample_df.shape[0]

    def test_missing_columns_handled_gracefully(self):
        """Should not raise even without ANO/INDE columns."""
        df = pd.DataFrame({"IDADE": [10, 11], "FASE": ["F3", "F4"]})
        result = add_trend_features(df)
        assert "INDE_DELTA" in result.columns


# ---------------------------------------------------------------------------
# Tests: add_agg_features
# ---------------------------------------------------------------------------


class TestAddAggFeatures:
    def test_creates_mean_column(self, sample_df):
        result = add_agg_features(sample_df)
        assert "INDICADORES_MEAN" in result.columns

    def test_creates_std_column(self, sample_df):
        result = add_agg_features(sample_df)
        assert "INDICADORES_STD" in result.columns

    def test_creates_range_column(self, sample_df):
        result = add_agg_features(sample_df)
        assert "INDICADORES_RANGE" in result.columns

    def test_mean_is_within_valid_range(self, sample_df):
        result = add_agg_features(sample_df)
        assert result["INDICADORES_MEAN"].between(0, 10).all()

    def test_std_nonnegative(self, sample_df):
        result = add_agg_features(sample_df)
        assert (result["INDICADORES_STD"] >= 0).all()

    def test_range_nonnegative(self, sample_df):
        result = add_agg_features(sample_df)
        assert (result["INDICADORES_RANGE"] >= 0).all()

    def test_no_nulls_in_agg_features(self, sample_df):
        result = add_agg_features(sample_df)
        agg_cols = ["INDICADORES_MEAN", "INDICADORES_STD", "INDICADORES_MIN", "INDICADORES_MAX"]
        assert result[agg_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Tests: add_interaction_features
# ---------------------------------------------------------------------------


class TestAddInteractionFeatures:
    def test_creates_inde_x_ips(self, sample_df):
        result = add_interaction_features(sample_df)
        assert "INDE_x_IPS" in result.columns

    def test_creates_ida_x_ipp(self, sample_df):
        result = add_interaction_features(sample_df)
        assert "IDA_x_IPP" in result.columns

    def test_creates_ieg_x_ian(self, sample_df):
        result = add_interaction_features(sample_df)
        assert "IEG_x_IAN" in result.columns

    def test_interaction_values_nonnegative(self, sample_df):
        result = add_interaction_features(sample_df)
        assert (result["INDE_x_IPS"] >= 0).all()

    def test_returns_dataframe(self, sample_df):
        result = add_interaction_features(sample_df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: add_phase_age_gap
# ---------------------------------------------------------------------------


class TestAddPhaseAgeGap:
    def test_creates_fase_gap_column(self, sample_df):
        result = add_phase_age_gap(sample_df)
        assert "FASE_GAP" in result.columns

    def test_creates_defasagem_severa_column(self, sample_df):
        result = add_phase_age_gap(sample_df)
        assert "DEFASAGEM_SEVERA" in result.columns

    def test_defasagem_severa_is_binary(self, sample_df):
        result = add_phase_age_gap(sample_df)
        assert set(result["DEFASAGEM_SEVERA"].unique()).issubset({0, 1})

    def test_gap_is_numeric(self, sample_df):
        result = add_phase_age_gap(sample_df)
        assert pd.api.types.is_numeric_dtype(result["FASE_GAP"])

    def test_missing_columns_handled(self):
        """Should not raise even without IDADE/FASE."""
        df = pd.DataFrame({"INDE": [7.0, 5.0], "ANO": [2024, 2024]})
        result = add_phase_age_gap(df)
        assert "FASE_GAP" in result.columns
        assert result["FASE_GAP"].sum() == 0

    def test_older_student_has_positive_gap(self):
        """A 14-year-old in F3 should have a positive (large) gap."""
        df = pd.DataFrame({"IDADE": [14.0], "FASE": ["F3"]})
        result = add_phase_age_gap(df)
        assert result["FASE_GAP"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Tests: run_feature_engineering (integration)
# ---------------------------------------------------------------------------


class TestRunFeatureEngineering:
    def test_returns_dataframe(self, sample_df):
        result = run_feature_engineering(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_multiple_features(self, sample_df):
        result = run_feature_engineering(sample_df)
        new_cols = set(result.columns) - set(sample_df.columns)
        assert len(new_cols) >= 5

    def test_no_extra_rows(self, sample_df):
        result = run_feature_engineering(sample_df)
        assert len(result) == len(sample_df)

    def test_original_columns_preserved(self, sample_df):
        result = run_feature_engineering(sample_df)
        for col in sample_df.columns:
            assert col in result.columns
