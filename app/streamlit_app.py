from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.agreement.bias import bias_table, ensemble_scores
from src.agreement.kappa import cohen_kappa
from src.calibration.corrector import correction_table
from src.visualisation import bias_heatmap, calibration_curve, kappa_heatmap

st.set_page_config(page_title="LLM Judge Benchmark", layout="wide")
st.title("LLM Judge Benchmark — Interactive Explorer")

EVAL_DIR = Path("data/eval")
METRICS = ["context_relevance", "groundedness", "answer_relevance"]


@st.cache_data
def load_scores() -> dict[str, dict[str, np.ndarray]]:
    scores: dict[str, dict[str, np.ndarray]] = {}

    for path in sorted(EVAL_DIR.glob("scores_*.json")):
        records = json.loads(path.read_text())
        if not records:
            continue
        df = pd.DataFrame(records)
        if df["score"].isna().all():
            continue
        model = df["model"].iloc[0]
        model_scores: dict[str, list[float]] = {m: [] for m in METRICS}
        for rec in records:
            if rec["metric"] in model_scores:
                model_scores[rec["metric"]].append(float(rec["score"]))
        scores[model] = {m: np.array(v) for m, v in model_scores.items() if v}

    if len(scores) >= 2:
        scores["ensemble"] = ensemble_scores(scores)

    return scores


scores = load_scores()
available_models = [m for m in scores if m != "ensemble"]

if not scores:
    st.warning("No score data found. Run `make score-all` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Instance Browser", "Kappa Heatmap", "Bias Heatmap", "Calibration"]
)

with tab1:
    st.subheader("Per-instance score comparison")
    if available_models:
        metric = st.selectbox("Metric", METRICS)
        rows = []
        for m in available_models + (["ensemble"] if "ensemble" in scores else []):
            arr = scores[m].get(metric, np.array([]))
            for i, v in enumerate(arr):
                rows.append({"instance": i, "annotator": m, "score": v})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.pivot(index="instance", columns="annotator", values="score"))

with tab2:
    st.subheader("Cohen's Kappa between judges")
    metric = st.selectbox("Metric", METRICS, key="kappa_metric")
    annotators = available_models + (["ensemble"] if "ensemble" in scores else [])
    if len(annotators) >= 2:
        matrix: dict[str, dict[str, float]] = {}
        for a in annotators:
            matrix[a] = {}
            for b in annotators:
                arr_a = scores[a].get(metric, np.array([]))
                arr_b = scores[b].get(metric, np.array([]))
                if len(arr_a) == len(arr_b) and len(arr_a) > 0:
                    matrix[a][b] = cohen_kappa(arr_a, arr_b)
                else:
                    matrix[a][b] = float("nan")
        df_kappa = pd.DataFrame(matrix).T
        st.pyplot(kappa_heatmap(df_kappa, title=f"Kappa — {metric}"))

with tab3:
    st.subheader("Systematic bias relative to ensemble consensus")
    if "ensemble" in scores and available_models:
        btable = bias_table(scores)
        rows_b = [{"model": m, "metric": mt, "bias": v} for (m, mt), v in btable.items()]
        df_bias = pd.DataFrame(rows_b).pivot(index="model", columns="metric", values="bias")
        st.pyplot(bias_heatmap(df_bias))

with tab4:
    st.subheader("Calibration: raw vs corrected scores (target = ensemble)")
    if "ensemble" in scores and available_models:
        model_sel = st.selectbox("Model", available_models, key="cal_model")
        metric_sel = st.selectbox("Metric", METRICS, key="cal_metric")
        raw = scores[model_sel].get(metric_sel, np.array([]))
        reference = scores["ensemble"].get(metric_sel, np.array([]))
        if len(raw) == len(reference) and len(raw) > 0:
            ctable = correction_table(scores)
            corrector = ctable.get((model_sel, metric_sel))
            if corrector:
                corrected = corrector.transform(raw)
                st.pyplot(calibration_curve(raw, corrected, reference, label=model_sel))
