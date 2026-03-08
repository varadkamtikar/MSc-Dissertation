import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="EEG Reward Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.metric-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #374151;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #9CA3AF;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #F9FAFB;
}
.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.section-subtitle {
    color: #9CA3AF;
    margin-bottom: 1rem;
    font-size: 18px;
}
.insight-box {
    background-color: #F3F4F6;
    padding: 14px;
    border-radius: 12px;
    border-left: 5px solid #2563EB;
    color: #111827;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

subject_json = load_json(DATA_DIR / "metrics/subject_metrics.json")
subject_csv = load_csv(DATA_DIR / "metrics/subject_metrics.csv")
model_csv = load_csv(DATA_DIR / "model_comparison.csv")
label_csv = load_csv(DATA_DIR / "label_distribution.csv")

# -----------------------------
# Helpers
# -----------------------------
def build_flat_model_df(subject_data):
    rows = []
    for subject_id, subject_info in subject_data.items():
        for method in ["plain", "smote"]:
            if method in subject_info and "models" in subject_info[method]:
                for model_name, vals in subject_info[method]["models"].items():
                    rows.append({
                        "subject": subject_id,
                        "method": method,
                        "model": model_name,
                        "acc_mean": vals.get("acc_mean"),
                        "acc_std": vals.get("acc_std"),
                        "f1_mean": vals.get("f1_mean"),
                        "f1_std": vals.get("f1_std"),
                        "n_splits": vals.get("n_splits")
                    })
    return pd.DataFrame(rows)

def build_best_df(subject_data):
    rows = []
    for subject_id, subject_info in subject_data.items():
        for method in ["plain", "smote"]:
            if method in subject_info and "best" in subject_info[method]:
                best = subject_info[method]["best"]
                rows.append({
                    "subject": subject_id,
                    "method": method,
                    "best_model": best.get("model"),
                    "acc_mean": best.get("acc_mean"),
                    "acc_std": best.get("acc_std"),
                    "f1_mean": best.get("f1_mean"),
                    "f1_std": best.get("f1_std")
                })
    return pd.DataFrame(rows)

def build_fold_df(subject_data):
    rows = []
    for subject_id, subject_info in subject_data.items():
        for method in ["plain", "smote"]:
            if method in subject_info and "fold_details" in subject_info[method]:
                for row in subject_info[method]["fold_details"]:
                    rows.append({
                        "subject": subject_id,
                        "method": method,
                        "model": row.get("model"),
                        "fold": row.get("fold"),
                        "acc": row.get("acc"),
                        "macro_f1": row.get("macro_f1"),
                        "confusion_matrix": str(row.get("confusion_matrix"))
                    })
    return pd.DataFrame(rows)

flat_df = build_flat_model_df(subject_json)
best_df = build_best_df(subject_json)
fold_df = build_fold_df(subject_json)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 EEG Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset & Pipeline",
        "Subject Analytics",
        "Model Comparison",
        "Evaluation & Leakage",
        "Conclusions"
    ]
)

st.sidebar.markdown("---")

subject_options = sorted(flat_df["subject"].unique().tolist()) if not flat_df.empty else []
model_options = sorted(flat_df["model"].unique().tolist()) if not flat_df.empty else []

selected_subject = st.sidebar.selectbox("Select Subject", subject_options) if subject_options else None
selected_model = st.sidebar.selectbox("Select Model", model_options) if model_options else None
selected_method = st.sidebar.selectbox("Method", ["plain", "smote"])

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.markdown('<div class="section-title">AI-Based Detection of Reward-Related Neural Patterns from EEG Signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Interactive dissertation dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Key insight:</b> This research prioritises <b>leakage-free evaluation</b>, <b>cross-subject realism</b>, and <b>reproducible EEG classification</b> rather than inflated performance.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Subjects</div>
            <div class="metric-value">32</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">EEG Channels</div>
            <div class="metric-value">32</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_macro = flat_df["f1_mean"].max() if not flat_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Macro-F1</div>
            <div class="metric-value">{best_macro:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        best_acc = flat_df["acc_mean"].max() if not flat_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Accuracy</div>
            <div class="metric-value">{best_acc:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Research Question")
    st.write("Can a strictly controlled, leakage-free EEG pipeline produce reliable classification performance under realistic multi-subject conditions?")

    st.markdown("### Objectives")
    st.write("""
    - Build a fully automated end-to-end EEG processing pipeline  
    - Compare subject-dependent and subject-independent evaluation  
    - Prevent data leakage during preprocessing and training  
    - Quantify realistic performance and subject variability  
    """)

# -----------------------------
# Dataset & Pipeline
# -----------------------------
elif page == "Dataset & Pipeline":
    st.markdown('<div class="section-title">Dataset & Pipeline</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Dataset", "Pipeline", "Feature Strategy"])

    with tab1:
        st.markdown("### DEAP Dataset Summary")
        st.write("""
        - 32 participants  
        - 32 EEG channels  
        - 40 trials per participant  
        - Emotion elicitation through video stimuli  
        - Labels derived from DEAP self-assessment ratings  
        """)

    with tab2:
        pipeline_path = ASSETS_DIR / "pipeline.png"
        if pipeline_path.exists():
            st.image(str(pipeline_path), caption="End-to-End EEG Classification Pipeline", use_container_width=True)
        else:
            st.warning("pipeline.png not found in assets folder.")

        st.markdown("### Pipeline stages")
        st.write("""
        1. Raw EEG loading  
        2. Filtering and re-referencing  
        3. ICA-based artefact removal  
        4. Event detection and epoching  
        5. Feature extraction  
        6. Subject-dependent and subject-independent evaluation  
        7. Performance reporting  
        """)

    with tab3:
        st.markdown("### Feature extraction design")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Subject-Dependent")
            st.write("""
            - TSFEL-based spectral features  
            - Channel-wise feature extraction  
            - Richer feature space  
            - Better within-subject fit  
            """)

        with col2:
            st.markdown("#### Subject-Independent")
            st.write("""
            - Bandpower features  
            - Hemispheric asymmetry measures  
            - Lower dimensionality  
            - Better generalisation across subjects  
            """)

# -----------------------------
# Subject Analytics
# -----------------------------
elif page == "Subject Analytics":
    st.markdown('<div class="section-title">Subject Analytics</div>', unsafe_allow_html=True)

    if flat_df.empty or selected_subject is None:
        st.warning("No subject data found.")
    else:
        subject_df = flat_df[
            (flat_df["subject"] == selected_subject) &
            (flat_df["method"] == selected_method)
        ]

        best_subject_df = best_df[
            (best_df["subject"] == selected_subject) &
            (best_df["method"] == selected_method)
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Subject", selected_subject)

        with col2:
            if not best_subject_df.empty:
                st.metric("Best Model", best_subject_df.iloc[0]["best_model"])

        with col3:
            if not best_subject_df.empty:
                st.metric("Best Macro-F1", f"{best_subject_df.iloc[0]['f1_mean']:.3f}")

        st.markdown("### Subject model performance")
        fig = px.bar(
            subject_df,
            x="model",
            y="f1_mean",
            color="model",
            text="f1_mean",
            title=f"Macro-F1 by Model for {selected_subject} ({selected_method})"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detailed table")
        st.dataframe(subject_df, use_container_width=True)

        fold_subject_df = fold_df[
            (fold_df["subject"] == selected_subject) &
            (fold_df["method"] == selected_method)
        ]

        with st.expander("View fold-level performance"):
            st.dataframe(fold_subject_df, use_container_width=True)

# -----------------------------
# Model Comparison
# -----------------------------
elif page == "Model Comparison":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    if flat_df.empty:
        st.warning("No model comparison data found.")
    else:
        filtered = flat_df[flat_df["method"] == selected_method]

        tab1, tab2, tab3 = st.tabs(["Across Subjects", "Selected Model", "Best Models"])

        with tab1:
            agg = filtered.groupby("model", as_index=False)[["acc_mean", "f1_mean"]].mean()

            fig1 = px.bar(
                agg,
                x="model",
                y="f1_mean",
                color="model",
                text="f1_mean",
                title=f"Average Macro-F1 Across Subjects ({selected_method})"
            )
            fig1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.bar(
                agg,
                x="model",
                y="acc_mean",
                color="model",
                text="acc_mean",
                title=f"Average Accuracy Across Subjects ({selected_method})"
            )
            fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            model_df = filtered[filtered["model"] == selected_model]

            fig3 = px.line(
                model_df,
                x="subject",
                y="f1_mean",
                markers=True,
                title=f"{selected_model} Macro-F1 Across Subjects ({selected_method})"
            )
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.box(
                model_df,
                y="f1_mean",
                title=f"{selected_model} Macro-F1 Distribution ({selected_method})"
            )
            st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            best_method_df = best_df[best_df["method"] == selected_method]
            count_best = best_method_df["best_model"].value_counts().reset_index()
            count_best.columns = ["model", "count"]

            fig5 = px.pie(
                count_best,
                values="count",
                names="model",
                title=f"Best Model Frequency Across Subjects ({selected_method})"
            )
            st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# Evaluation & Leakage
# -----------------------------
elif page == "Evaluation & Leakage":
    st.markdown('<div class="section-title">Evaluation & Leakage Control</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Critical point:</b> Lower accuracies in this dissertation are expected because all fitting operations were isolated within training folds, and subject overlap was prevented during subject-independent evaluation.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Leakage-Free Logic", "Why Accuracy Is Lower"])

    with tab1:
        eval_path = ASSETS_DIR / "evaluation_flow.png"
        if eval_path.exists():
            st.image(str(eval_path), caption="Leakage-Free Evaluation Flow", use_container_width=True)
        else:
            st.warning("evaluation_flow.png not found in assets folder.")

        st.write("""
        **Subject-dependent evaluation**
        - Stratified cross-validation
        - Preserves class balance
        - Same subject appears in train and test

        **Subject-independent evaluation**
        - GroupKFold by subject
        - No subject overlap between train and test
        - More realistic measure of generalisation

        **Leakage prevention**
        - scaling only on training fold
        - SMOTE only on training fold
        - test data never seen during fitting
        """)

    with tab2:
        st.write("""
        The lower performance is not a flaw in the work. It is the result of:
        - strict subject-independent testing
        - high inter-subject EEG variability
        - leakage-free preprocessing and model fitting
        - use of Macro-F1 instead of misleading accuracy alone
        """)

        compare = flat_df.groupby("method", as_index=False)[["acc_mean", "f1_mean"]].mean()
        fig = px.bar(
            compare.melt(id_vars="method", var_name="metric", value_name="value"),
            x="method",
            y="value",
            color="metric",
            barmode="group",
            text="value",
            title="Plain vs SMOTE Average Performance"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Conclusions
# -----------------------------
elif page == "Conclusions":
    st.markdown('<div class="section-title">Conclusions</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>Final takeaway:</b> When subject overlap and leakage are removed, EEG classification becomes substantially harder. This dissertation therefore prioritises realistic evaluation over inflated reporting.
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    ### Key contributions
    - Fully automated EEG processing pipeline  
    - Subject-dependent and subject-independent evaluation  
    - Strict leakage prevention  
    - Transparent reporting of realistic performance  

    ### Key learning
    Evaluation design has a greater impact on reported performance than model complexity.

    ### Future work
    - connectivity-based features  
    - transfer learning and subject adaptation  
    - cross-dataset validation  
    """)

    if not flat_df.empty:
        summary = flat_df.groupby("model", as_index=False)[["acc_mean", "f1_mean"]].mean()
        st.download_button(
            label="Download model summary as CSV",
            data=summary.to_csv(index=False),
            file_name="model_summary.csv",
            mime="text/csv"
        )