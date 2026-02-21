"""
ETL script: normalizes the Passos Mágicos PEDE datasets into a unified format.

Input files:
  - data/raw/PEDE_PASSOS_DATASET_FIAP.csv  (historical: 2020, 2021, 2022 in wide format)
  - data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx  (2024, needs column mapping)

Output:
  - data/processed/pede_unified.csv  (long format: one row per student-year)
  - data/processed/pede_2024.csv     (2024 cleaned, INDE derived if missing)

Column mapping to standard names:
  NOME, ANO, FASE, PEDRA, INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN,
  PONTO_VIRADA, DEFASAGEM, IDADE, GENERO, INSTITUICAO_ENSINO
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import Config, ensure_dir, get_logger

logger = get_logger(__name__)

RAW = Path(Config.DATA_RAW_PATH)
PROCESSED = Path(Config.DATA_PROCESSED_PATH)

# ─────────────────────────────────────────────
# PEDE INDE formula (from DataDictionary)
# INDE = 0.1*IAA + 0.15*IEG + 0.2*IPS + 0.3*IDA + 0.1*IPP + 0.15*IPV
# (approximation — exact formula may vary by year)
# ─────────────────────────────────────────────
INDE_WEIGHTS = {
    "IAA": 0.10,
    "IEG": 0.15,
    "IPS": 0.20,
    "IDA": 0.30,
    "IPP": 0.10,
    "IPV": 0.15,
}

# PEDRA thresholds (from documentation)
def classify_pedra(inde: object) -> str:
    try:
        val = float(inde)
    except (TypeError, ValueError):
        return "Desconhecido"
    if pd.isna(val):
        return "Desconhecido"
    if val < 5.506:
        return "Quartzo"
    elif val < 6.868:
        return "Ágata"
    elif val < 8.230:
        return "Ametista"
    else:
        return "Topázio"


# ─────────────────────────────────────────────
# 1. Process FIAP historical dataset (wide → long)
# ─────────────────────────────────────────────

def process_fiap_historical() -> pd.DataFrame:
    """
    Transforms the wide-format FIAP dataset (2020-2022) into long format.
    Each output row = one student × one year.
    """
    logger.info("Processing FIAP historical dataset (2020-2022)...")
    fiap_path = RAW / "PEDE_PASSOS_DATASET_FIAP.csv"
    if not fiap_path.exists():
        logger.warning(f"FIAP CSV not found: {fiap_path}")
        return pd.DataFrame()

    df_wide = pd.read_csv(fiap_path, sep=None, engine="python", encoding="utf-8-sig")
    logger.info(f"FIAP wide shape: {df_wide.shape}")

    # Standard column mapping per year
    year_configs = {
        2020: {
            "FASE": "FASE_TURMA_2020",
            "PEDRA": "PEDRA_2020",
            "INDE": "INDE_2020",
            "IAA": "IAA_2020",
            "IEG": "IEG_2020",
            "IPS": "IPS_2020",
            "IDA": "IDA_2020",
            "IPP": "IPP_2020",
            "IPV": "IPV_2020",
            "IAN": "IAN_2020",
            "PONTO_VIRADA": "PONTO_VIRADA_2020",
            "IDADE": "IDADE_ALUNO_2020",
            "ANOS_PM": "ANOS_PM_2020",
            "INSTITUICAO": "INSTITUICAO_ENSINO_ALUNO_2020",
            "DEFASAGEM": None,
        },
        2021: {
            "FASE": "FASE_2021",
            "PEDRA": "PEDRA_2021",
            "INDE": "INDE_2021",
            "IAA": "IAA_2021",
            "IEG": "IEG_2021",
            "IPS": "IPS_2021",
            "IDA": "IDA_2021",
            "IPP": "IPP_2021",
            "IPV": "IPV_2021",
            "IAN": "IAN_2021",
            "PONTO_VIRADA": "PONTO_VIRADA_2021",
            "IDADE": None,
            "ANOS_PM": None,
            "INSTITUICAO": "INSTITUICAO_ENSINO_ALUNO_2021",
            "DEFASAGEM": "DEFASAGEM_2021",
        },
        2022: {
            "FASE": "FASE_2022",
            "PEDRA": "PEDRA_2022",
            "INDE": "INDE_2022",
            "IAA": "IAA_2022",
            "IEG": "IEG_2022",
            "IPS": "IPS_2022",
            "IDA": "IDA_2022",
            "IPP": "IPP_2022",
            "IPV": "IPV_2022",
            "IAN": "IAN_2022",
            "PONTO_VIRADA": "PONTO_VIRADA_2022",
            "IDADE": None,
            "ANOS_PM": None,
            "INSTITUICAO": "INSTITUICAO_ENSINO_ALUNO_2021",  # reuse 2021
            "DEFASAGEM": "DEFASAGEM_2021",  # reuse 2021
        },
    }

    dfs = []
    for year, mapping in year_configs.items():
        cols = {"NOME": df_wide.get("NOME", pd.Series(dtype="object"))}
        cols["ANO"] = year

        for std_col, src_col in mapping.items():
            if src_col and src_col in df_wide.columns:
                cols[std_col] = df_wide[src_col].values
            else:
                cols[std_col] = np.nan

        year_df = pd.DataFrame(cols)
        year_df = year_df[year_df["INDE"].notna() | year_df["IAA"].notna()]
        dfs.append(year_df)
        logger.info(f"  Year {year}: {len(year_df)} records")

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"FIAP processed long shape: {result.shape}")
    return result


# ─────────────────────────────────────────────
# 2. Process PEDE 2024
# ─────────────────────────────────────────────

def process_pede_2024() -> pd.DataFrame:
    """
    Processes and standardizes the 2024 PEDE dataset.
    Derives INDE from component indicators where missing.
    """
    logger.info("Processing PEDE 2024 dataset...")
    path_2024 = RAW / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    if not path_2024.exists():
        logger.warning(f"PEDE 2024 not found: {path_2024}")
        return pd.DataFrame()

    df = pd.read_excel(path_2024)
    logger.info(f"PEDE 2024 raw shape: {df.shape}")

    # Map 2024 columns → standard names
    col_map = {
        "Nome": "NOME",
        "RA": "RA",
        "Fase": "FASE",
        "Turma": "TURMA",
        "Ano nasc": "ANO_NASC",
        "Idade 22": "IDADE",   # age as of 2022 reference — proxy
        "Gênero": "GENERO",
        "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IDA": "IDA",
        "IPV": "IPV",
        "IAN": "IAN",
        "IPP": "Cg",       # Cg (nota cognitiva) used as proxy for IPP when IPP absent
        "Fase ideal": "NIVEL_IDEAL",
        "Defas": "DEFASAGEM",
        "Atingiu PV": "PONTO_VIRADA",
        "Pedra 20": "PEDRA_2020",
        "Pedra 21": "PEDRA_2021",
        "Pedra 22": "PEDRA_2022",
        "INDE 22": "INDE_2022_REF",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ANO"] = 2024

    # Derive INDE from indicators (weighted formula)
    indicator_cols = [c for c in INDE_WEIGHTS.keys() if c in df.columns]
    if indicator_cols:
        total_weight = sum(INDE_WEIGHTS[c] for c in indicator_cols)
        df["INDE"] = sum(df[c] * INDE_WEIGHTS[c] for c in indicator_cols) / total_weight
        logger.info(f"INDE derived from: {indicator_cols}")
    elif "INDE_2022_REF" in df.columns:
        df["INDE"] = df["INDE_2022_REF"]
        logger.info("Using INDE_2022_REF as INDE proxy.")

    # Derive PEDRA from INDE if not present directly
    if "PEDRA" not in df.columns:
        if "PEDRA_2022" in df.columns and df["PEDRA_2022"].notna().sum() > 0:
            df["PEDRA"] = df["PEDRA_2022"]
        elif "INDE" in df.columns:
            df["PEDRA"] = df["INDE"].apply(classify_pedra)
            logger.info("PEDRA derived from INDE thresholds.")

    # IPP: use Cg (cognitive grade) as proxy if present
    if "IPP" not in df.columns and "Cg" in df.columns:
        df["IPP"] = df["Cg"]
    elif "IPP" not in df.columns:
        df["IPP"] = np.nan

    # Keep only standard columns
    standard_cols = [
        "NOME", "ANO", "FASE", "PEDRA", "INDE",
        "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
        "PONTO_VIRADA", "DEFASAGEM", "IDADE", "GENERO", "INSTITUICAO",
        "NIVEL_IDEAL",
    ]
    existing = [c for c in standard_cols if c in df.columns]
    df = df[existing].copy()
    df = df[df["INDE"].notna() | df["IAA"].notna()]

    logger.info(f"PEDE 2024 processed shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. Unify and clean
# ─────────────────────────────────────────────

def unify_datasets(df_hist: pd.DataFrame, df_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates historical and 2024 datasets into a unified long-format DataFrame.
    Applies final cleaning and type corrections.
    """
    logger.info("Unifying all datasets...")
    frames = [f for f in [df_hist, df_2024] if len(f) > 0]
    df = pd.concat(frames, ignore_index=True)

    # Normalize string columns
    str_cols = ["NOME", "FASE", "PEDRA", "GENERO", "INSTITUICAO"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace("NAN", np.nan)

    # Normalize FASE: keep only F1-F8 format (handles "F5-A", "F5 A", integers like 7 etc.)
    if "FASE" in df.columns:
        def _normalize_fase(x: object) -> object:
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            # Handle plain number: "7" -> "F7"
            if s.isdigit():
                return f"F{s}"
            s_upper = s.upper()
            import re
            m = re.search(r"F(\d)", s_upper)
            return f"F{m.group(1)}" if m else np.nan

        df["FASE"] = df["FASE"].apply(_normalize_fase)

    # Normalize PEDRA values
    pedra_map = {
        "QUARTZO": "Quartzo",
        "ÁGATA": "Ágata",
        "AGATA": "Ágata",
        "AMETISTA": "Ametista",
        "TOPÁZIO": "Topázio",
        "TOPAZIO": "Topázio",
    }
    invalid_pedra = {"NAN", "#NULO!", "DESCONHECIDO", "NONE", ""}
    if "PEDRA" in df.columns:
        def _normalize_pedra(x: object) -> object:
            if pd.isna(x):
                return np.nan
            s = str(x).strip().upper()
            if s in invalid_pedra:
                return np.nan
            return pedra_map.get(s, str(x).strip())

        df["PEDRA"] = df["PEDRA"].apply(_normalize_pedra)

        # Re-classify any remaining NaN PEDRA using INDE
        if "INDE" in df.columns:
            mask = df["PEDRA"].isna() & df["INDE"].notna()
            df.loc[mask, "PEDRA"] = df.loc[mask, "INDE"].apply(classify_pedra)

    # Normalize PONTO_VIRADA to binary
    if "PONTO_VIRADA" in df.columns:
        df["PONTO_VIRADA"] = (
            df["PONTO_VIRADA"].astype(str).str.upper()
            .isin(["SIM", "TRUE", "1", "S", "YES"])
        ).astype(int)

    # Clip numeric scores to [0, 10]
    score_cols = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
    for col in [c for c in score_cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 10)

    # Drop rows where all key indicators are null
    key_indicators = [c for c in ["INDE", "IAA", "IDA"] if c in df.columns]
    df = df.dropna(subset=key_indicators, how="all")

    logger.info(f"Unified dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Years present: {sorted(df['ANO'].unique())}")
    logger.info(f"PEDRA distribution:\n{df['PEDRA'].value_counts().to_string()}")

    return df


# ─────────────────────────────────────────────
# Main ETL
# ─────────────────────────────────────────────

def run_etl() -> pd.DataFrame:
    """
    Runs the complete ETL pipeline.

    Returns:
        pd.DataFrame: Unified, clean dataset saved to data/processed/
    """
    ensure_dir(PROCESSED)

    df_hist = process_fiap_historical()
    df_2024 = process_pede_2024()
    df_unified = unify_datasets(df_hist, df_2024)

    # Save outputs
    out_unified = PROCESSED / "pede_unified.csv"
    df_unified.to_csv(out_unified, index=False, encoding="utf-8-sig")
    logger.info(f"Unified dataset saved: {out_unified}")

    # Also save reference split for drift monitoring
    train_ref = df_unified[df_unified["ANO"].isin([2020, 2021, 2022])]
    train_ref_path = PROCESSED / "train_reference.csv"
    train_ref.to_csv(train_ref_path, index=False, encoding="utf-8-sig")
    logger.info(f"Training reference saved: {train_ref_path}")

    current_prod = df_unified[df_unified["ANO"] == 2024]
    current_prod_path = PROCESSED / "current_production.csv"
    current_prod.to_csv(current_prod_path, index=False, encoding="utf-8-sig")
    logger.info(f"Production data saved: {current_prod_path}")

    # Print summary stats
    print("\n" + "=" * 60)
    print("ETL COMPLETE — DATASET SUMMARY")
    print("=" * 60)
    print(df_unified.groupby("ANO").size().rename("Records").to_string())
    print(f"\nTotal records: {len(df_unified):,}")
    print(f"Columns: {list(df_unified.columns)}")
    print(f"\nINDE stats:\n{df_unified['INDE'].describe().round(3).to_string()}")
    if "PEDRA" in df_unified.columns:
        print(f"\nPEDRA distribution:\n{df_unified['PEDRA'].value_counts().to_string()}")
    print("=" * 60)

    return df_unified


if __name__ == "__main__":
    run_etl()
