"""
Data drift detection for the Passos Mágicos ML API.

Computes:
- PSI (Population Stability Index) for numeric features
- KS test for distributional comparison
- Evidently AI report (if available)

Usage:
    python monitoring/drift_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import Config, ensure_dir, get_logger, get_timestamp

logger = get_logger(__name__)

REPORT_DIR = Path(Config.DRIFT_REPORT_PATH)
DRIFT_THRESHOLD = Config.DRIFT_THRESHOLD


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes the Population Stability Index (PSI) between two distributions.

    PSI Interpretation:
    - PSI < 0.10: No significant change
    - 0.10 ≤ PSI < 0.25: Moderate change (monitor)
    - PSI ≥ 0.25: Significant change (action required)

    Args:
        expected: Reference distribution (training data).
        actual: Current distribution (production data).
        n_bins: Number of bins for discretization.

    Returns:
        float: PSI value.
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Normalize and avoid zeros
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# KS Test
# ---------------------------------------------------------------------------


def compute_ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """
    Performs the Kolmogorov-Smirnov test between two distributions.

    Args:
        expected: Reference distribution.
        actual: Current distribution.

    Returns:
        dict with 'statistic' and 'p_value'.
    """
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return {"statistic": float(ks_stat), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Full drift report
# ---------------------------------------------------------------------------


def generate_drift_report(
    reference_path: str | Path,
    current_path: str | Path,
    feature_cols: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
) -> dict:
    """
    Generates a full data drift report comparing reference and current datasets.

    Args:
        reference_path: Path to reference CSV (training data).
        current_path: Path to current CSV (production data).
        feature_cols: List of columns to analyze. Defaults to all numeric columns.
        output_path: Path to save the JSON report.

    Returns:
        dict: Drift report with PSI and KS results per feature.
    """
    logger.info("Generating drift report...")

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    if feature_cols is None:
        feature_cols = ref_df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in feature_cols if c in cur_df.columns]

    report: dict = {
        "generated_at": get_timestamp(),
        "reference_samples": len(ref_df),
        "current_samples": len(cur_df),
        "drift_threshold": DRIFT_THRESHOLD,
        "features": {},
        "drift_detected": False,
        "drifted_features": [],
    }

    for col in feature_cols:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        ks = compute_ks_test(ref_vals, cur_vals)

        drifted = psi >= DRIFT_THRESHOLD or ks["p_value"] < 0.05

        feature_report = {
            "psi": round(psi, 4),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "drift_detected": drifted,
            "psi_interpretation": (
                "No change" if psi < 0.10
                else "Moderate change" if psi < 0.25
                else "Significant change"
            ),
        }

        report["features"][col] = feature_report

        if drifted:
            report["drifted_features"].append(col)
            report["drift_detected"] = True
            logger.warning(f"Drift detected in '{col}': PSI={psi:.4f}, KS p-value={ks['p_value']:.4f}")

    logger.info(
        f"Drift report complete: {len(report['drifted_features'])} / {len(feature_cols)} "
        f"features show drift."
    )

    # Try Evidently AI report
    _try_evidently_report(ref_df, cur_df, feature_cols)

    # Save JSON report
    output = output_path or (REPORT_DIR / f"drift_report_{get_timestamp()}.json")
    ensure_dir(Path(output).parent)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Drift report saved: {output}")

    return report


def _try_evidently_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Attempts to generate an Evidently AI HTML report if available."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df[feature_cols], current_data=cur_df[feature_cols])

        html_path = REPORT_DIR / f"evidently_drift_{get_timestamp()}.html"
        ensure_dir(REPORT_DIR)
        report.save_html(str(html_path))
        logger.info(f"Evidently report saved: {html_path}")

    except ImportError:
        logger.info("Evidently AI not available. Using manual PSI/KS drift detection.")
    except Exception as e:
        logger.warning(f"Evidently report generation failed: {e}")


if __name__ == "__main__":
    generate_drift_report(
        reference_path=f"{Config.DATA_PROCESSED_PATH}/train_reference.csv",
        current_path=f"{Config.DATA_PROCESSED_PATH}/current_production.csv",
    )
