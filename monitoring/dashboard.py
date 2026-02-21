"""
Streamlit monitoring dashboard for the Passos Mágicos ML API.

Displays:
- Model performance metrics
- Feature distribution: training vs. production
- Drift alerts (PSI threshold)
- Prediction logs over time

Usage:
    streamlit run monitoring/dashboard.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import Config

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Passos Mágicos — ML Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORT_DIR = Path(Config.DRIFT_REPORT_PATH)
MODEL_META_PATH = Path("app/model/model_metadata.json")
DATA_PATH = Path(Config.DATA_PROCESSED_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_model_metadata() -> dict:
    if MODEL_META_PATH.exists():
        with open(MODEL_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_latest_drift_report() -> dict | None:
    reports = sorted(REPORT_DIR.glob("drift_report_*.json"), reverse=True)
    if reports:
        with open(reports[0], encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image("https://passosmagicos.org.br/wp-content/uploads/2020/08/logo.png", width=200)
st.sidebar.title("🎓 Passos Mágicos")
st.sidebar.markdown("**ML Monitoring Dashboard**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navegação",
    ["📊 Visão Geral", "🔍 Drift de Dados", "📈 Métricas do Modelo"],
)

st.sidebar.divider()
st.sidebar.info(f"Threshold de Drift: `{Config.DRIFT_THRESHOLD}`")

# ---------------------------------------------------------------------------
# Page: Visão Geral
# ---------------------------------------------------------------------------

if page == "📊 Visão Geral":
    st.title("🎓 Passos Mágicos — Monitoramento ML")
    st.caption("Pipeline de risco de defasagem escolar")

    meta = load_model_metadata()
    drift = load_latest_drift_report()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_name = meta.get("model_name", "N/A")
        st.metric("Modelo", model_name)

    with col2:
        recall = meta.get("test_metrics", {}).get("recall", None)
        st.metric("Recall (Test)", f"{recall:.2%}" if recall else "N/A")

    with col3:
        auc = meta.get("test_metrics", {}).get("auc_roc", None)
        st.metric("AUC-ROC (Test)", f"{auc:.4f}" if auc else "N/A")

    with col4:
        if drift:
            drifted = len(drift.get("drifted_features", []))
            total = len(drift.get("features", {}))
            color = "normal" if drifted == 0 else "inverse"
            st.metric("Features com Drift", f"{drifted}/{total}", delta_color=color)
        else:
            st.metric("Features com Drift", "Sem relatório")

    st.divider()

    if drift and drift.get("drift_detected"):
        st.error(
            f"⚠️ **Drift detectado em {len(drift['drifted_features'])} feature(s):** "
            + ", ".join(f"`{f}`" for f in drift["drifted_features"])
        )
    elif drift:
        st.success("✅ Nenhum drift significativo detectado.")
    else:
        st.info("ℹ️ Nenhum relatório de drift encontrado. Execute `make drift-report` para gerar.")

    if meta:
        st.subheader("Informações do Modelo")
        st.json(
            {
                "model": meta.get("model_name"),
                "version": meta.get("model_version"),
                "trained_at": meta.get("trained_at"),
                "training_time_s": meta.get("training_time_seconds"),
                "threshold": meta.get("threshold"),
            }
        )

# ---------------------------------------------------------------------------
# Page: Drift de Dados
# ---------------------------------------------------------------------------

elif page == "🔍 Drift de Dados":
    st.title("🔍 Análise de Drift de Dados")

    drift = load_latest_drift_report()

    if not drift:
        st.warning("Nenhum relatório de drift disponível. Execute `python monitoring/drift_report.py`.")
        st.stop()

    st.metric("Data do Relatório", drift.get("generated_at", "N/A"))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Amostras de Referência", f"{drift.get('reference_samples', 0):,}")
    with col2:
        st.metric("Amostras de Produção", f"{drift.get('current_samples', 0):,}")

    st.divider()
    st.subheader("PSI por Feature")

    features = drift.get("features", {})
    if features:
        df_drift = pd.DataFrame(
            [
                {
                    "Feature": k,
                    "PSI": v["psi"],
                    "KS p-value": v["ks_p_value"],
                    "Drift Detectado": "⚠️ Sim" if v["drift_detected"] else "✅ Não",
                    "Interpretação": v["psi_interpretation"],
                }
                for k, v in features.items()
            ]
        ).sort_values("PSI", ascending=False)

        st.dataframe(df_drift, use_container_width=True)

        st.bar_chart(df_drift.set_index("Feature")["PSI"])

        st.caption(
            "**PSI:** < 0.1 = sem mudança | 0.1–0.25 = mudança moderada | > 0.25 = mudança significativa"
        )

# ---------------------------------------------------------------------------
# Page: Métricas do Modelo
# ---------------------------------------------------------------------------

elif page == "📈 Métricas do Modelo":
    st.title("📈 Métricas do Modelo")

    meta = load_model_metadata()

    if not meta:
        st.warning(
            "Metadados do modelo não encontrados. Execute `make train` para treinar o modelo."
        )
        st.stop()

    st.subheader("Métricas no Conjunto de Teste")
    test_metrics = meta.get("test_metrics", {})

    if test_metrics:
        cols = st.columns(len(test_metrics))
        for col, (name, value) in zip(cols, test_metrics.items()):
            if isinstance(value, (int, float)):
                col.metric(name.upper().replace("_", "-"), f"{value:.4f}")

    st.divider()
    st.subheader("Resultado da Validação Cruzada (CV)")
    cv_results = meta.get("cv_results", {})
    if cv_results:
        df_cv = pd.DataFrame(
            [{"Modelo": k, "CV Recall": v} for k, v in cv_results.items()]
        ).sort_values("CV Recall", ascending=False)
        st.dataframe(df_cv, use_container_width=True)
        st.bar_chart(df_cv.set_index("Modelo"))

    st.divider()
    st.subheader("Hiperparâmetros do Melhor Modelo")
    hp = meta.get("hyperparameters", {})
    if hp:
        st.json(hp)
    else:
        st.info("Hiperparâmetros padrão utilizados (sem otimização).")

    st.divider()
    st.subheader("Justificativa da Métrica Principal")
    st.markdown(
        """
        **Recall** foi escolhido como métrica principal porque:

        > No contexto de predição de risco de defasagem escolar, o custo de um
        > **Falso Negativo** (deixar de identificar um aluno em risco) é muito maior do que
        > o custo de um **Falso Positivo** (acionar intervenção desnecessária).

        - Alto Recall → minimiza alunos em risco não identificados
        - AUC-ROC → avalia a capacidade discriminativa geral do modelo
        - F1-Score → equilíbrio entre Precision e Recall como métrica secundária
        """
    )
