"""
streamlit_app.py — Interactive deployment of the Student Placement Predictor.

Deployment option required by the brief (Streamlit). Run with:

    streamlit run app/streamlit_app.py

Tabs:
    1. Predict   – single student form + gauge + SHAP contributors + salary
    2. Batch     – paste student profiles (CSV/TSV) and score them in bulk
    3. Model Info – CV comparison, test metrics, learning curve + EDA plots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config          # noqa: E402  — sys.path must precede src imports
from src.predict import Predictor  # noqa: E402
from src.salary import load_salary_predictor  # noqa: E402

st.set_page_config(
    page_title="PlaceIQ — Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model artifacts once ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained model…")
def load_artifacts():
    predictor = Predictor() if config.BUNDLE_PATH.exists() else None
    salary = load_salary_predictor()
    meta = None
    if config.META_PATH.exists():
        meta = json.loads(config.META_PATH.read_text())
    return predictor, salary, meta


try:
    predictor, salary_predictor, meta = load_artifacts()
    if predictor is None:
        st.error("Train a model first:  python scripts/train_all.py")
        st.stop()
except Exception as exc:  # pragma: no cover
    st.error(f"Could not load model: {exc}")
    st.info("Run `python scripts/train_all.py`, then re-open this app.")
    st.stop()

# ── Shared helpers ────────────────────────────────────────────────────────────

CATEGORY_VALUES = {
    "gender": ["M", "F"],
    "ssc_b": ["Central", "Others"],
    "hsc_b": ["Central", "Others"],
    "hsc_s": ["Commerce", "Science", "Arts"],
    "degree_t": ["Sci&Tech", "Comm&Mgmt", "Others"],
    "workex": ["Yes", "No"],
    "specialisation": ["Mkt&Fin", "Mkt&HR"],
}


def build_form_inputs(defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    st.markdown("#### Academic & demographic details")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.radio("Gender", CATEGORY_VALUES["gender"],
                          index=CATEGORY_VALUES["gender"].index(defaults.get("gender", "M")),
                          horizontal=True)
        ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, float(defaults.get("ssc_p", 70.0)), 0.5)
        hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, float(defaults.get("hsc_p", 70.0)), 0.5)
    with c2:
        ssc_b = st.selectbox("SSC Board", CATEGORY_VALUES["ssc_b"],
                             index=CATEGORY_VALUES["ssc_b"].index(defaults.get("ssc_b", "Central")))
        hsc_b = st.selectbox("HSC Board", CATEGORY_VALUES["hsc_b"],
                             index=CATEGORY_VALUES["hsc_b"].index(defaults.get("hsc_b", "Central")))
        hsc_s = st.selectbox("HSC Stream", CATEGORY_VALUES["hsc_s"],
                             index=CATEGORY_VALUES["hsc_s"].index(defaults.get("hsc_s", "Science")))
    with c3:
        degree_p = st.number_input("Degree Percentage", 0.0, 100.0, float(defaults.get("degree_p", 65.0)), 0.5)
        etest_p = st.number_input("Employability Test %", 0.0, 100.0, float(defaults.get("etest_p", 65.0)), 0.5)
        mba_p = st.number_input("MBA Percentage", 0.0, 100.0, float(defaults.get("mba_p", 65.0)), 0.5)

    st.markdown("#### Course & experience")
    c4, c5 = st.columns(2)
    with c4:
        degree_t = st.selectbox("Degree Type", CATEGORY_VALUES["degree_t"],
                                index=CATEGORY_VALUES["degree_t"].index(defaults.get("degree_t", "Sci&Tech")))
        specialisation = st.selectbox("MBA Specialisation", CATEGORY_VALUES["specialisation"],
                                      index=CATEGORY_VALUES["specialisation"].index(defaults.get("specialisation", "Mkt&Fin")))
    with c5:
        workex = st.radio("Work Experience", CATEGORY_VALUES["workex"],
                          index=CATEGORY_VALUES["workex"].index(defaults.get("workex", "No")),
                          horizontal=True)

    with st.expander("Optional skill features (improve accuracy)"):
        c6, c7, c8 = st.columns(3)
        with c6:
            cgpa = st.number_input("CGPA (0-10)", 0.0, 10.0,
                                   float(defaults.get("cgpa", 7.5) or 7.5), 0.1)
            internships = st.number_input("Internships (#)", 0, 10,
                                          int(defaults.get("internships", 1) or 1), 1)
        with c7:
            projects = st.number_input("Projects (#)", 0, 20,
                                       int(defaults.get("projects", 2) or 2), 1)
            workshops = st.number_input("Workshops/Certs (#)", 0, 20,
                                        int(defaults.get("workshops", 1) or 1), 1)
        with c8:
            aptitude_score = st.number_input("Aptitude Score (0-100)", 0.0, 100.0,
                                             float(defaults.get("aptitude_score", 65.0) or 65.0), 1.0)
            soft_skills = st.number_input("Soft Skills Rating (0-10)", 0.0, 10.0,
                                          float(defaults.get("soft_skills", 6.0) or 6.0), 0.5)

    return {
        "gender": gender, "ssc_p": ssc_p, "ssc_b": ssc_b, "hsc_p": hsc_p,
        "hsc_b": hsc_b, "hsc_s": hsc_s, "degree_p": degree_p, "degree_t": degree_t,
        "workex": workex, "etest_p": etest_p, "specialisation": specialisation,
        "mba_p": mba_p, "cgpa": cgpa, "internships": internships, "projects": projects,
        "workshops": workshops, "aptitude_score": aptitude_score, "soft_skills": soft_skills,
    }


def render_gauge(probability: float, label: str) -> None:
    color = "#2e7d32" if label == "Placed" else "#c62828"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": label, "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#ffccbc"},
                {"range": [40, 60], "color": "#ffe0b2"},
                {"range": [60, 100], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.9,
                "value": predictor.bundle["threshold"] * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_contributors(contributors: list) -> None:
    st.markdown("#### What influenced this prediction?  (SHAP)")
    if not contributors:
        st.caption("No explanation available for this run.")
        return
    rows = pd.DataFrame(contributors)
    rows["direction"] = rows["shap"].apply(
        lambda s: "Toward Placed" if s > 0 else "Toward Not Placed"
    )
    fig = px.bar(
        rows.sort_values("shap"),
        x="shap", y="feature", color="direction",
        orientation="h",
        color_discrete_map={"Toward Placed": "#2e7d32", "Toward Not Placed": "#c62828"},
        labels={"shap": "SHAP contribution", "feature": ""},
        title="Feature contributions to the prediction",
    )
    fig.update_layout(height=max(320, 42 * len(rows)), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 PlaceIQ")
    st.caption("Student Placement Predictor")
    st.divider()
    st.markdown(
        f"**Champion model:** `{predictor.model_name}`  \n"
        f"**ROC-AUC (CV):** {meta.get('cv_roc_auc_mean', 0):.3f}  \n"
        f"**Test accuracy:** {meta.get('test_metrics', {}).get('accuracy', 0):.2f}  \n"
        f"**Threshold:** {predictor.bundle.get('threshold', 0.5):.2f}"
    )
    st.divider()
    tab = st.radio("Navigate", ["Predict", "Batch Scoring", "Model Info & EDA"],
                   label_visibility="collapsed")

# ── Tab: Single prediction ────────────────────────────────────────────────────
if tab == "Predict":
    st.header("🎯 Single Student Prediction")
    col_form, col_result = st.columns([3, 2])

    with col_form:
        inputs = build_form_inputs()

        if st.button("Predict Placement", type="primary", use_container_width=True):
            try:
                result = predictor.predict(inputs)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = inputs
            st.rerun()

    with col_result:
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Fill the form on the left and click **Predict Placement**.")
        else:
            st.markdown("### Result")
            render_gauge(result["probability"], result["prediction"])
            if result["prediction"] == "Placed" and salary_predictor is not None:
                try:
                    sal = salary_predictor.predict({k: v for k, v in st.session_state.get("last_inputs", {}).items()})
                    st.markdown(
                        f"##### 💰 Estimated salary: **₹{sal['salary_estimate']:,}/yr**  \n"
                        f"Band: **{sal['salary_band'].upper()}**  ·  range ₹{sal['salary_range'][0]:,} – ₹{sal['salary_range'][1]:,}"
                    )
                except Exception as exc:
                    st.caption(f"Salary estimate unavailable: {exc}")
            elif result["prediction"] == "Placed":
                st.info("Salary model not trained.")
            render_contributors(result.get("contributors", []))

# ── Tab: Batch scoring ────────────────────────────────────────────────────────
elif tab == "Batch Scoring":
    st.header("📋 Batch Scoring")
    st.caption("Paste rows in CSV/TSV format with the same column names (gender, ssc_p, … mba_p). "
               "Optional skill columns may be omitted.")
    sample = st.code(
        "gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p\n"
        "M,78.0,Central,82.0,Central,Science,72.0,Sci&Tech,Yes,85.0,Mkt&Fin,68.0\n"
        "F,55.0,Others,58.0,Others,Arts,60.0,Comm&Mgmt,No,45.0,Mkt&HR,52.0",
        language="text",
    )
    text_area = st.text_area("Paste your data (CSV/TSV)", height=180)
    if st.button("Score batch", type="primary"):
        if not text_area.strip():
            st.warning("Paste some data first.")
            st.stop()
        df = None
        for sep in (",", "\t", ";"):
            try:
                df = pd.read_csv(pd.io.common.StringIO(text_area), sep=sep)
                break
            except Exception:
                continue
        if df is None:
            st.error("Could not parse input. Use CSV/TSV format.")
            st.stop()
        required = [c for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
                    if c in df.columns and c not in config.KUMBHAR_NUMERIC]
        missing = [c for c in config.CATEGORICAL_FEATURES + config.BENROSHAN_NUMERIC
                   if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        rows = []
        for _, row in df.iterrows():
            raw = row.to_dict()
            proba, pred = predictor.probability(raw)
            rec = {"prediction": pred, "probability": round(proba, 3)}
            if pred == "Placed" and salary_predictor is not None:
                try:
                    sal = salary_predictor.predict(raw)
                    rec["salary_estimate"] = sal["salary_estimate"]
                    rec["salary_band"] = sal["salary_band"]
                except Exception:
                    pass
            rows.append(rec)

        out = df.copy()
        out["prediction"] = [r["prediction"] for r in rows]
        out["probability"] = [r["probability"] for r in rows]
        if salary_predictor is not None:
            out["salary_estimate"] = [r.get("salary_estimate", "") for r in rows]
            out["salary_band"] = [r.get("salary_band", "") for r in rows]
        st.success(f"Scored {len(out)} students — "
                   f"{(out['prediction'] == 'Placed').sum()} placed, "
                   f"{(out['prediction'] != 'Placed').sum()} not placed.")
        st.dataframe(out, use_container_width=True)

# ── Tab: Model info + EDA ─────────────────────────────────────────────────────
else:
    st.header("📊 Model Comparison & EDA")
    tab_a, tab_b, tab_c = st.tabs(["Model comparison", "Evaluation metrics", "EDA plots"])

    with tab_a:
        st.subheader("Cross-validated comparison (5-fold × 3 repeats)")
        comp_path = config.CV_COMPARISON_PATH
        if comp_path.exists():
            comp = pd.read_csv(comp_path)
            st.dataframe(comp.round(4), use_container_width=True)
        else:
            st.caption("No comparison saved. Run training first.")

    with tab_b:
        st.subheader("Test-set performance")
        tm = meta.get("test_metrics", {})
        if tm:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{tm.get('accuracy', 0):.2%}")
            c2.metric("Precision", f"{tm.get('precision', 0):.2%}")
            c3.metric("Recall", f"{tm.get('recall', 0):.2%}")
            c4.metric("F1-score", f"{tm.get('f1', 0):.2%}")
            c5, c6, c7 = st.columns(3)
            c5.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")
            c6.metric("PR-AUC", f"{tm.get('pr_auc', 0):.3f}")
            c7.metric("Brier score", f"{tm.get('brier', 0):.3f}")
            st.caption(f"Decision threshold: {meta.get('threshold', 0.5):.3f} "
                       f"(optimised for {meta.get('threshold_metric', 'f1')})  ·  "
                       f"calibration: {meta.get('calibration_method', 'none')}")
        else:
            st.caption("No metrics recorded yet.")

        lc = meta.get("learning_curve")
        if lc:
            st.subheader("Learning curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["train"],
                                     name="Training score", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=lc["train_sizes"], y=lc["cv"],
                                     name="Cross-validation score", mode="lines+markers"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Accuracy (CV)", xaxis_title="Training examples")
            st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        st.subheader("EDA visualisations (Phase 1)")
        eda_dir = config.PLOTS_DIR / "eda"
        captions = {
            "placement_distribution.png": "1 · Target class balance",
            "marks_distribution.png": "2 · Distribution of academic marks",
            "placement_by_categorical.png": "3 · Placement rate by categorical features",
            "correlation_heatmap.png": "4 · Correlation heatmap",
            "marks_boxplots.png": "5 · Marks by placement status",
            "academic_avg_vs_status.png": "6 · Academic average vs placement",
            "gender_specialisation.png": "7 · Gender x specialisation placement rates",
        }
        files = [p for p in sorted(eda_dir.glob("*.png"))
                 if p.name in captions] if eda_dir.exists() else []
        if not files:
            st.info("Run `python scripts/eda.py` to generate the EDA plots.")
        else:
            for idx in range(0, len(files), 2):
                row = st.columns(2)
                for col, f in zip(row, files[idx:idx + 2]):
                    col.image(str(f), caption=captions.get(f.name, f.stem),
                              use_container_width=True)

st.divider()
st.caption("PlaceIQ · CIE ML project · run "
           f"`{predictor.run_id}` · trained {predictor.bundle.get('trained_at', '?')}"
           if predictor else "")
