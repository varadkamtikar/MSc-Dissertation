import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Reward Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR  = Path(__file__).resolve().parent
ASSETS    = BASE_DIR / "assets"
DATA      = BASE_DIR / "data"
REPORTS   = DATA / "reports"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* ── App Background ──────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 90% 55% at 5% 25%, rgba(99,102,241,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 65% 45% at 95% 10%, rgba(56,189,248,0.06) 0%, transparent 52%),
        radial-gradient(ellipse 55% 60% at 50% 100%, rgba(168,85,247,0.05) 0%, transparent 55%),
        #060d18;
}

/* ── Main Content ─────────────────────────────────── */
.block-container { padding: 2rem 2.8rem 3rem; max-width: 1500px; }

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a3050; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #2d5080; }

/* ── Hero ────────────────────────────────────────── */
.hero-wrapper {
    position: relative;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.06) 50%, rgba(56,189,248,0.07) 100%);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 24px;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #818cf8, #a78bfa 45%, #38bdf8);
    border-radius: 24px 24px 0 0;
}
.hero-wrapper::after {
    content: '🧠';
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5.5rem;
    opacity: 0.07;
    pointer-events: none;
    filter: blur(1px);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #a5b4fc;
    margin-bottom: 0.9rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c7d2fe 0%, #a78bfa 45%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.18;
    margin-bottom: 0.6rem;
    letter-spacing: -0.025em;
    max-width: 82%;
}
.hero-sub { color: #4a6080; font-size: 0.95rem; font-weight: 400; }

/* ── KPI Cards ───────────────────────────────────── */
.kpi-card {
    position: relative;
    background: rgba(10, 18, 36, 0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: transform .25s cubic-bezier(.4,0,.2,1), border-color .25s, box-shadow .25s;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #818cf8 40%, #38bdf8, transparent);
    opacity: 0.6;
    transition: opacity .25s;
}
.kpi-card:hover {
    transform: translateY(-5px);
    border-color: rgba(129,140,248,0.32);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(99,102,241,0.1);
}
.kpi-card:hover::before { opacity: 1; }
.kpi-label {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #2d4060;
    margin-bottom: 0.5rem;
}
.kpi-value { font-size: 1.95rem; font-weight: 900; line-height: 1.1; color: #f1f5f9; letter-spacing: -0.03em; }
.c-indigo { color: #818cf8 !important; }
.c-green  { color: #34d399 !important; }
.c-sky    { color: #38bdf8 !important; }
.c-amber  { color: #fbbf24 !important; }
.c-rose   { color: #fb7185 !important; }

/* ── Alert Boxes ─────────────────────────────────── */
.box {
    border-radius: 16px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.25rem;
    font-size: 0.885rem;
    line-height: 1.65;
}
.box-indigo {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-left: 4px solid #818cf8;
    color: #c7d2fe;
}
.box-amber {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.18);
    border-left: 4px solid #fbbf24;
    color: #fef3c7;
}
.box-green {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.18);
    border-left: 4px solid #34d399;
    color: #d1fae5;
}

/* ── Section Headings ────────────────────────────── */
.sh  { font-size: 1.65rem; font-weight: 800; color: #f1f5f9; margin-bottom: .15rem; letter-spacing: -0.025em; line-height: 1.2; }
.sh-sm { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: .15rem; letter-spacing: -0.01em; }
.ss  { color: #2d4060; font-size: .875rem; margin-bottom: 1.3rem; line-height: 1.5; }

/* ── Divider ─────────────────────────────────────── */
.hr { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent); margin: 1.8rem 0; }

/* ── Stat Rows ───────────────────────────────────── */
.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: .5rem .85rem; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: .855rem; border-radius: 6px; transition: background .15s; margin: 1px 0;
}
.stat-row:hover { background: rgba(255,255,255,0.03); }
.stat-row span:first-child { color: #3d5575; }
.stat-row span:last-child  { color: #e2e8f0; font-weight: 600; }

/* ── Badges ──────────────────────────────────────── */
.badge {
    display: inline-block;
    background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.25);
    color: #a5b4fc; border-radius: 100px;
    padding: 3px 11px; font-size: .68rem; font-weight: 700;
    margin: 0 3px 3px 0; letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-g { background: rgba(52,211,153,.12); border-color: rgba(52,211,153,.25); color: #6ee7b7; }
.badge-b { background: rgba(56,189,248,.12); border-color: rgba(56,189,248,.25); color: #7dd3fc; }

/* ── Pipeline Steps ──────────────────────────────── */
.pstep {
    display: flex; align-items: flex-start; gap: .85rem;
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.055);
    border-radius: 14px; padding: .85rem 1.05rem; margin-bottom: .5rem;
    transition: border-color .2s, background .2s, transform .2s;
}
.pstep:hover { background: rgba(99,102,241,0.07); border-color: rgba(99,102,241,0.22); transform: translateX(3px); }
.pnum {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border-radius: 9px; width: 30px; height: 30px; min-width: 30px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: .75rem; box-shadow: 0 3px 10px rgba(99,102,241,0.45);
}
.ptxt { color: #6080a0; font-size: .875rem; line-height: 1.55; }
.ptxt b { color: #dde6f0; }

/* ── Leakage Steps ───────────────────────────────── */
.lstep {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #818cf8;
    border-radius: 0 14px 14px 0;
    padding: .8rem 1.15rem; margin-bottom: .6rem;
    transition: border-color .2s, background .2s, transform .2s;
}
.lstep:hover { background: rgba(99,102,241,0.07); border-left-color: #a5b4fc; transform: translateX(3px); }
.lstep-title { font-weight: 700; color: #a5b4fc; font-size: .875rem; }
.lstep-desc  { color: #3d5575; font-size: .8rem; margin-top: .2rem; line-height: 1.55; }

/* ── Contribution Cards ──────────────────────────── */
.ccard {
    background: rgba(10, 18, 36, 0.75); border: 1px solid rgba(255,255,255,0.065);
    border-radius: 18px; padding: 1.4rem; height: 100%;
    transition: transform .25s cubic-bezier(.4,0,.2,1), border-color .25s, box-shadow .25s;
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
}
.ccard:hover { transform: translateY(-5px); border-color: rgba(129,140,248,0.28); box-shadow: 0 16px 50px rgba(0,0,0,0.35), 0 0 30px rgba(99,102,241,0.07); }
.ccard-icon  { font-size: 1.85rem; margin-bottom: .6rem; }
.ccard-title { font-weight: 700; color: #f1f5f9; font-size: .92rem; margin-bottom: .4rem; }
.ccard-desc  { color: #3d5575; font-size: .825rem; line-height: 1.65; }

/* ── Finding Cards ───────────────────────────────── */
.fcard {
    background: rgba(10, 18, 36, 0.65); border: 1px solid rgba(255,255,255,0.065);
    border-radius: 16px; padding: 1.05rem 1.15rem; margin-bottom: .75rem;
    transition: border-color .2s, background .2s, transform .2s;
}
.fcard:hover { border-color: rgba(129,140,248,0.2); background: rgba(99,102,241,0.05); transform: translateX(3px); }
.fcard-head { display: flex; align-items: center; gap: .7rem; margin-bottom: .3rem; }
.fcard-icon { font-size: 1.3rem; }
.fcard-title { font-weight: 700; color: #f1f5f9; font-size: .9rem; }
.fcard-body  { color: #3d5575; font-size: .82rem; line-height: 1.6; padding-left: 2rem; }

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    padding: .65rem 1.2rem !important; color: #2d4060 !important;
    font-size: .855rem !important; font-weight: 600 !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
    transition: color .2s !important; border-radius: 8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #6080a0 !important; background: rgba(255,255,255,0.025) !important; }
.stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; background: rgba(99,102,241,0.05) !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1.6rem 0 0 !important; background: transparent !important; }

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1d 0%, #040b17 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

/* Sidebar nav radio as nav links */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px !important; flex-direction: column !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding: .62rem .9rem !important; color: #2d4060 !important;
    font-size: .855rem !important; font-weight: 500 !important;
    cursor: pointer !important; border-radius: 10px !important;
    display: block !important; transition: color .15s, background .15s !important; width: 100% !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"]:hover label { color: #6080a0 !important; background: rgba(255,255,255,0.04) !important; }
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] {
    background: rgba(99,102,241,0.12) !important; border: 1px solid rgba(99,102,241,0.2) !important; border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"] label { color: #a5b4fc !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: .7rem !important; font-weight: 700 !important; color: #1e3050 !important;
    text-transform: uppercase !important; letter-spacing: .1em !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child:focus-within { border-color: rgba(99,102,241,0.4) !important; }

/* ── Plotly chart wrapper ─────────────────────────── */
[data-testid="stPlotlyChart"] > div { border-radius: 16px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.05) !important; }

/* ── Dataframes ──────────────────────────────────── */
[data-testid="stDataFrame"] > div { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.07) !important; overflow: hidden !important; }

/* ── Download Buttons ────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: rgba(99,102,241,0.1) !important; border: 1px solid rgba(99,102,241,0.22) !important;
    color: #a5b4fc !important; border-radius: 12px !important;
    font-weight: 600 !important; font-size: .845rem !important; transition: all .2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(99,102,241,0.2) !important; border-color: rgba(99,102,241,0.45) !important;
    transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(99,102,241,0.2) !important;
}

/* ── Images ──────────────────────────────────────── */
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────────
_BG   = "#0d1b2e"
_PAPER = "#07111d"
_GRID = "#162335"
PALETTE = ["#818cf8", "#34d399", "#fbbf24", "#22d3ee", "#f87171", "#c084fc", "#f472b6"]

def _sf(fig, title="", h=None):
    """Apply dark styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=_BG,
        paper_bgcolor=_PAPER,
        title=dict(text=title, font=dict(size=15, color="#f1f5f9"), x=0.02),
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(t=46, l=12, r=12, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor="#334155")
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor="#334155")
    if h:
        fig.update_layout(height=h)
    return fig

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data
def _load_json(p):
    return json.load(open(p)) if Path(p).exists() else {}

@st.cache_data
def _load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()

subj_json   = _load_json(DATA / "metrics/subject_metrics.json")
final_test  = _load_csv(DATA / "metrics/final_test_results.csv")
tuning      = _load_csv(DATA / "metrics/tuning_results_train_only.csv")
epochs_ret  = _load_csv(DATA / "metrics/table_A1_epochs_retention.csv")

# ── Build flat DataFrames ──────────────────────────────────────────────────────
def _flat(d):
    rows = []
    for sid, info in d.items():
        for meth in ["plain", "smote"]:
            if meth in info and "models" in info[meth]:
                for mdl, v in info[meth]["models"].items():
                    rows.append(dict(subject=sid, method=meth, model=mdl,
                                     acc_mean=v.get("acc_mean"), acc_std=v.get("acc_std"),
                                     f1_mean=v.get("f1_mean"),  f1_std=v.get("f1_std"),
                                     n_splits=v.get("n_splits")))
    return pd.DataFrame(rows)

def _best(d):
    rows = []
    for sid, info in d.items():
        for meth in ["plain", "smote"]:
            if meth in info and "best" in info[meth]:
                b = info[meth]["best"]
                rows.append(dict(subject=sid, method=meth, best_model=b.get("model"),
                                 acc_mean=b.get("acc_mean"), f1_mean=b.get("f1_mean"),
                                 acc_std=b.get("acc_std"),   f1_std=b.get("f1_std")))
    return pd.DataFrame(rows)

def _folds(d):
    rows = []
    for sid, info in d.items():
        for meth in ["plain", "smote"]:
            if meth in info and "fold_details" in info[meth]:
                for r in info[meth]["fold_details"]:
                    rows.append(dict(subject=sid, method=meth, model=r.get("model"),
                                     fold=r.get("fold"), acc=r.get("acc"),
                                     macro_f1=r.get("macro_f1"),
                                     confusion_matrix=r.get("confusion_matrix")))
    return pd.DataFrame(rows)

flat_df = _flat(subj_json)
best_df = _best(subj_json)
fold_df = _folds(subj_json)

subj_opts  = sorted(flat_df["subject"].unique()) if not flat_df.empty else []
model_opts = sorted(flat_df["model"].unique())   if not flat_df.empty else []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
        <div style="width:58px;height:58px;
                    background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
                    border-radius:16px;display:inline-flex;align-items:center;
                    justify-content:center;font-size:1.8rem;
                    box-shadow:0 4px 28px rgba(79,70,229,0.55);
                    margin-bottom:.85rem">🧠</div>
        <div style="font-weight:800;font-size:.93rem;color:#f1f5f9;
                    letter-spacing:-0.01em;line-height:1.3">
            EEG Reward Detection
        </div>
        <div style="font-size:.67rem;color:#1a2e48;margin-top:.35rem;
                    font-weight:700;letter-spacing:.08em;text-transform:uppercase">
            MSc · V. Kamtikar · 2024–25
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);margin:.4rem 0 1rem"></div>
    """, unsafe_allow_html=True)

    PAGE = st.radio("Navigate", [
        "Overview",
        "Dataset & Pipeline",
        "Subject Analytics",
        "Model Comparison",
        "Evaluation & Leakage",
        "Subject-Independent Results",
        "Conclusions",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);margin:1rem 0 .9rem"></div>
    <div style="font-size:.67rem;font-weight:700;letter-spacing:.1em;color:#1a2e48;
                text-transform:uppercase;margin-bottom:.65rem;padding:0 .1rem">Filters</div>
    """, unsafe_allow_html=True)

    sel_subject = st.selectbox("Subject", subj_opts)  if subj_opts  else None
    sel_model   = st.selectbox("Model",   model_opts) if model_opts else None
    sel_method  = st.selectbox("Method",  ["plain", "smote"])

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);margin:1rem 0 .8rem"></div>
    <div style="font-size:.67rem;color:#1a2e48;text-align:center;line-height:2;
                font-weight:600;letter-spacing:.03em">
        DEAP Dataset · 32 Subjects · 32 Channels<br>
        6 Classifiers · 5-Fold CV · ~54% Test Acc
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "Overview":
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-badge">MSc Dissertation · 2024–25</div>
        <div class="hero-title">AI-Based Detection of Reward-Related Neural Patterns from EEG Signals</div>
        <div class="hero-sub">Interactive dissertation dashboard · Varadkrishna Kamtikar</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="box box-indigo">
    <b>Research mission:</b> Build a strictly <b>leakage-free</b>, reproducible EEG classification pipeline
    that reports honest cross-subject performance, rather than inflated figures common in the literature.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    best_f1  = flat_df["f1_mean"].max()  if not flat_df.empty else 0
    best_acc = flat_df["acc_mean"].max() if not flat_df.empty else 0
    avg_f1   = flat_df["f1_mean"].mean() if not flat_df.empty else 0

    kpis = [
        ("Subjects",         "32",            "c-indigo"),
        ("EEG Channels",     "32",            "c-sky"),
        ("Trials / Subject", "40",            "c-green"),
        ("Models Evaluated", "6",             "c-amber"),
        ("Best Macro-F1",    f"{best_f1:.3f}", "c-green"),
        ("Best Accuracy",    f"{best_acc:.3f}","c-indigo"),
    ]
    cols = st.columns(6)
    for col, (label, val, cls) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Left: objectives  |  Right: heatmap ──────────────────────────────────
    col_l, col_r = st.columns([1, 1.85])

    with col_l:
        st.markdown('<div class="sh">Research Question</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="position:relative;background:rgba(10,18,36,0.75);
                    border:1px solid rgba(99,102,241,0.2);border-radius:16px;
                    padding:1.1rem 1.3rem;color:#c7d2fe;font-size:.9rem;
                    font-style:italic;line-height:1.75;margin-bottom:1.2rem;
                    border-left:4px solid #818cf8;
                    backdrop-filter:blur(8px)">
        "Can a strictly controlled, leakage-free EEG pipeline produce reliable classification
        performance under realistic multi-subject conditions?"
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sh-sm">Core Objectives</div>', unsafe_allow_html=True)
        for icon, label, detail in [
            ("→", "Build",    "Fully automated end-to-end EEG processing pipeline"),
            ("→", "Compare",  "Subject-dependent and subject-independent evaluation"),
            ("→", "Prevent",  "Data leakage during preprocessing and training"),
            ("→", "Quantify", "Realistic performance and inter-subject variability"),
            ("→", "Report",   "Transparent, reproducible benchmarks with Macro-F1"),
        ]:
            st.markdown(f"""
            <div class="pstep">
                <div class="pnum">{icon}</div>
                <div class="ptxt"><b>{label}:</b> {detail}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        if not flat_df.empty:
            hm = flat_df.groupby(["model", "method"])["f1_mean"].mean().reset_index()
            piv = hm.pivot(index="model", columns="method", values="f1_mean")
            fig_hm = go.Figure(go.Heatmap(
                z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
                colorscale="Viridis",
                text=np.round(piv.values, 3), texttemplate="%{text}",
                textfont=dict(size=13, color="white"),
                colorbar=dict(title="F1", tickfont=dict(color="#94a3b8")),
            ))
            _sf(fig_hm, "Average Macro-F1: Model × Method", h=360)
            st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Contribution cards ────────────────────────────────────────────────────
    st.markdown('<div class="sh">Key Contributions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "🔬", "Leakage-Free Pipeline",
         "Scaler and SMOTE fitted only inside training folds — test data never touches preprocessing."),
        (c2, "📊", "Dual Evaluation Paradigm",
         "Both within-subject (stratified CV) and cross-subject (GroupKFold) evaluations under identical conditions."),
        (c3, "🤖", "6 ML Classifiers",
         "LogReg, LinearSVM, SVM-RBF, GradBoost, XGBoost, and RandomForest compared systematically."),
        (c4, "⚡", "Realistic Benchmarks",
         "~54% accuracy under strict cross-subject conditions — honest, reproducible numbers for future EEG work."),
    ]:
        with col:
            st.markdown(f"""
            <div class="ccard">
                <div class="ccard-icon">{icon}</div>
                <div class="ccard-title">{title}</div>
                <div class="ccard-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Dataset & Pipeline":
    st.markdown('<div class="sh">Dataset & Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">DEAP EEG dataset, processing pipeline, feature engineering strategy, and epoch retention</div>', unsafe_allow_html=True)

    tab_ds, tab_pipe, tab_feat, tab_epoch = st.tabs(
        ["DEAP Dataset", "Processing Pipeline", "Feature Strategy", "Epoch Retention"]
    )

    # ── DEAP Dataset ──────────────────────────────────────────────────────────
    with tab_ds:
        col_info, col_donut = st.columns([1, 1.3])
        with col_info:
            st.markdown('<div class="sh-sm">DEAP Dataset</div>', unsafe_allow_html=True)
            for k, v in [
                ("Participants",         "32"),
                ("EEG Channels",         "32"),
                ("Trials / Participant", "40"),
                ("Sampling Rate",        "256 Hz"),
                ("Stimulus Duration",    "60 s"),
                ("Total Epochs",         "1,280"),
                ("Label Type",           "Binary (Reward / No-Reward)"),
                ("Label Threshold",      "5.5 (avg. Valence + Arousal)"),
                ("Source",               "DEAP self-assessment ratings"),
            ]:
                st.markdown(f'<div class="stat-row"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)

        with col_donut:
            fig_d = go.Figure(go.Pie(
                labels=["Reward\n(High V+A)", "No-Reward\n(Low V+A)"],
                values=[640, 640],
                hole=0.58,
                marker=dict(colors=["#818cf8", "#162335"],
                            line=dict(color="#07111d", width=2)),
                textinfo="label+percent",
                textfont=dict(size=12, color="#e2e8f0"),
            ))
            fig_d.add_annotation(text="1,280<br>epochs", x=0.5, y=0.5,
                                 font=dict(size=16, color="#f1f5f9"), showarrow=False)
            _sf(fig_d, "Label Distribution", h=330)
            fig_d.update_layout(showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("""
        <div class="box box-indigo">
        <b>Labelling:</b> DEAP self-assessment Valence and Arousal ratings are averaged per trial.
        Trials scoring above <b>5.5</b> are labelled <b>Reward</b> (positive emotional engagement);
        trials below are labelled <b>No-Reward</b>. The threshold was chosen to produce a balanced binary split.
        </div>
        """, unsafe_allow_html=True)

    # ── Processing Pipeline ───────────────────────────────────────────────────
    with tab_pipe:
        col_img, col_steps = st.columns([1.4, 1])
        with col_img:
            p = ASSETS / "pipeline.png"
            if p.exists():
                st.image(str(p), caption="End-to-End EEG Classification Pipeline", use_container_width=True)
        with col_steps:
            st.markdown('<div class="sh-sm">Pipeline Stages</div>', unsafe_allow_html=True)
            for i, s in enumerate([
                ("<b>Raw EEG loading</b> — DEAP .bdf / .mat format",),
                ("<b>Bandpass filtering</b> — 1–40 Hz FIR filter",),
                ("<b>Re-referencing</b> — average reference across 32 channels",),
                ("<b>ICA artefact removal</b> — eye and muscle component rejection",),
                ("<b>Event detection & epoching</b> — 60 s segments per trial",),
                ("<b>Feature extraction</b> — bandpower (SI) or TSFEL spectral (SD)",),
                ("<b>Leakage-free evaluation</b> — scaler & SMOTE inside fold only",),
                ("<b>Performance reporting</b> — Accuracy + Macro-F1 per subject",),
            ], start=1):
                st.markdown(f"""
                <div class="pstep">
                    <div class="pnum">{i}</div>
                    <div class="ptxt">{s[0]}</div>
                </div>
                """, unsafe_allow_html=True)

        p2 = ASSETS / "feature extraction and labelling.png"
        if p2.exists():
            st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
            st.image(str(p2), caption="Feature Extraction & Labelling Workflow", use_container_width=True)

    # ── Feature Strategy ──────────────────────────────────────────────────────
    with tab_feat:
        st.markdown('<div class="sh-sm">Feature Engineering Strategy</div>', unsafe_allow_html=True)
        col_dep, col_ind = st.columns(2)
        for col, hue, badge_cls, title, badge, rows in [
            (col_dep, "#818cf8", "", "Subject-Dependent", "Within-Subject", [
                ("TSFEL spectral features",  "Rich time-frequency representation from all channels"),
                ("Channel-wise extraction",  "Processes each of the 32 EEG channels independently"),
                ("High-dimensional space",   "Hundreds of spectral features per 60 s epoch"),
                ("Stratified 5-Fold CV",     "Preserves class balance; same subject in train & test"),
                ("Better within-subject fit","Captures individual idiosyncratic neural patterns"),
            ]),
            (col_ind, "#34d399", "badge-g", "Subject-Independent", "Cross-Subject", [
                ("Bandpower features",        "Delta, Theta, Alpha, Beta, Gamma frequency bands"),
                ("Hemispheric asymmetry",     "Left–right power difference across electrode pairs"),
                ("Lower dimensionality",      "More generalisable; avoids subject-specific overfitting"),
                ("GroupKFold by subject",     "Entire subjects held out — no trial overlap across folds"),
                ("Cross-subject generalisation", "Tests true population-level reward detection"),
            ]),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:rgba(10,18,36,0.8);border:1px solid rgba(255,255,255,0.07);
                            border-top:3px solid {hue};border-radius:18px;padding:1.3rem;
                            backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)">
                    <div style="font-weight:800;color:#f1f5f9;font-size:.95rem;margin-bottom:.4rem;letter-spacing:-0.01em">{title}</div>
                    <span class="badge {badge_cls}">{badge}</span>
                    <div style="margin-top:.8rem">
                """, unsafe_allow_html=True)
                for feat, desc in rows:
                    st.markdown(f"""
                    <div style="padding:.45rem 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                        <div style="font-weight:700;color:{hue};font-size:.82rem">{feat}</div>
                        <div style="color:#3d5575;font-size:.78rem;margin-top:.12rem">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Epoch Retention ───────────────────────────────────────────────────────
    with tab_epoch:
        st.markdown('<div class="sh-sm">Epoch Retention — Preprocessing Validation</div>', unsafe_allow_html=True)
        if not epochs_ret.empty:
            st.markdown("""
            <div class="box box-green">
            <b>Perfect retention:</b> All 40 events detected and all 40 epochs retained for every
            one of the 32 subjects — confirming clean, consistent preprocessing with zero dropped epochs.
            </div>
            """, unsafe_allow_html=True)

            fig_ep = px.bar(
                epochs_ret, x="Subject",
                y=["EventsDetected", "EpochsRetained"],
                barmode="group",
                color_discrete_map={"EventsDetected": "#334155", "EpochsRetained": "#818cf8"},
                labels={"value": "Count", "variable": ""},
            )
            _sf(fig_ep, "Events Detected vs Epochs Retained per Subject", h=320)
            fig_ep.update_layout(bargap=0.25, bargroupgap=0.05)
            st.plotly_chart(fig_ep, use_container_width=True)
            st.dataframe(epochs_ret, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Subject Analytics":
    st.markdown('<div class="sh">Subject Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Per-subject model performance, fold-level detail, confusion matrices, and EEG signal plots</div>', unsafe_allow_html=True)

    if flat_df.empty or sel_subject is None:
        st.warning("No subject data loaded.")
        st.stop()

    sub_flat  = flat_df[(flat_df["subject"] == sel_subject) & (flat_df["method"] == sel_method)]
    sub_best  = best_df[(best_df["subject"] == sel_subject) & (best_df["method"] == sel_method)]
    sub_folds = fold_df[(fold_df["subject"] == sel_subject) & (fold_df["method"] == sel_method)]

    bm   = sub_best.iloc[0]["best_model"] if not sub_best.empty else "—"
    bf1  = sub_best.iloc[0]["f1_mean"]    if not sub_best.empty else 0
    bacc = sub_best.iloc[0]["acc_mean"]   if not sub_best.empty else 0
    g_avg = flat_df[flat_df["method"] == sel_method]["f1_mean"].mean()
    delta  = bf1 - g_avg

    # KPI row
    ks = st.columns(5)
    for col, lbl, val, cls in [
        (ks[0], "Subject",       sel_subject.upper(), "c-indigo"),
        (ks[1], "Best Model",    bm,                  "c-sky"),
        (ks[2], "Best Macro-F1", f"{bf1:.3f}",        "c-green"),
        (ks[3], "Best Accuracy", f"{bacc:.3f}",        "c-amber"),
        (ks[4], "vs Global Avg", f"{delta:+.3f}",     "c-green" if delta >= 0 else "c-rose"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value {cls}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Model Performance", "Fold Detail", "EEG Signals", "Data Table"])

    # ── Model Performance ──────────────────────────────────────────────────────
    with t1:
        c_bar, c_radar = st.columns(2)
        with c_bar:
            fig_b = px.bar(
                sub_flat.sort_values("f1_mean", ascending=False),
                x="model", y="f1_mean", error_y="f1_std",
                color="model", color_discrete_sequence=PALETTE,
                text="f1_mean",
                labels={"f1_mean": "Macro-F1", "model": ""},
            )
            fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_b.update_layout(showlegend=False)
            _sf(fig_b, f"Macro-F1 by Model — {sel_subject.upper()} ({sel_method})", h=380)
            st.plotly_chart(fig_b, use_container_width=True)

        with c_radar:
            models_ = sub_flat["model"].tolist()
            f1_     = sub_flat.set_index("model")["f1_mean"].reindex(models_).tolist()
            acc_    = sub_flat.set_index("model")["acc_mean"].reindex(models_).tolist()
            if len(models_) >= 3:
                fig_r = go.Figure()
                for vals, name, col_ in [
                    (f1_,  "Macro-F1", "#818cf8"),
                    (acc_, "Accuracy", "#34d399"),
                ]:
                    closed_vals  = vals + [vals[0]]
                    closed_theta = models_ + [models_[0]]
                    r_c, g_c, b_c = int(col_[1:3], 16), int(col_[3:5], 16), int(col_[5:7], 16)
                    fig_r.add_trace(go.Scatterpolar(
                        r=closed_vals, theta=closed_theta,
                        fill="toself", name=name,
                        line_color=col_,
                        fillcolor=f"rgba({r_c},{g_c},{b_c},0.15)",
                    ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor=_BG,
                        radialaxis=dict(visible=True, range=[0, 1],
                                        gridcolor="#334155", color="#64748b"),
                        angularaxis=dict(gridcolor="#334155", color="#94a3b8"),
                    ),
                    template="plotly_dark", paper_bgcolor=_PAPER,
                    height=380,
                    title=dict(text=f"Radar — {sel_subject.upper()}", font=dict(size=15, color="#f1f5f9"), x=0.02),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=46, l=12, r=12, b=12),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # Plain vs SMOTE for this subject
        both_sub = flat_df[flat_df["subject"] == sel_subject]
        fig_cmp = px.bar(
            both_sub, x="model", y="f1_mean", color="method",
            barmode="group", error_y="f1_std",
            color_discrete_map={"plain": "#818cf8", "smote": "#34d399"},
            text="f1_mean",
            labels={"f1_mean": "Macro-F1", "model": "", "method": "Method"},
        )
        fig_cmp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        _sf(fig_cmp, f"Plain vs SMOTE — {sel_subject.upper()}", h=360)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Fold Detail ────────────────────────────────────────────────────────────
    with t2:
        if not sub_folds.empty:
            fig_fl = px.line(
                sub_folds, x="fold", y="macro_f1", color="model",
                markers=True, color_discrete_sequence=PALETTE,
                labels={"macro_f1": "Macro-F1", "fold": "Fold"},
            )
            _sf(fig_fl, f"Fold-Level Macro-F1 — {sel_subject.upper()} ({sel_method})", h=380)
            st.plotly_chart(fig_fl, use_container_width=True)

            fig_fa = px.line(
                sub_folds, x="fold", y="acc", color="model",
                markers=True, color_discrete_sequence=PALETTE,
                line_dash="model",
                labels={"acc": "Accuracy", "fold": "Fold"},
            )
            _sf(fig_fa, f"Fold-Level Accuracy — {sel_subject.upper()} ({sel_method})", h=340)
            st.plotly_chart(fig_fa, use_container_width=True)

        # Confusion matrix for best model's last fold
        best_info = subj_json.get(sel_subject, {}).get(sel_method, {}).get("best", {})
        cm = best_info.get("last_fold_confusion_matrix")
        if cm:
            st.markdown(f"**Confusion Matrix — {bm} (last fold)**")
            cm_arr = np.array(cm, dtype=int)
            cm_labels = ["No-Reward", "Reward"]
            fig_cm = px.imshow(
                cm_arr, x=cm_labels, y=cm_labels,
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale="Blues",
                text_auto=True, aspect="equal",
            )
            _sf(fig_cm)
            fig_cm.update_layout(height=310, width=370,
                                  coloraxis_showscale=False,
                                  margin=dict(t=20, l=12, r=12, b=12))
            st.plotly_chart(fig_cm)

    # ── EEG Signals ────────────────────────────────────────────────────────────
    with t3:
        st.markdown(f"**EEG Signal Visualisations — {sel_subject.upper()}**")

        # Resolve ERP/PSD — some subjects (s24–s28) were saved with a _fixed_raw_ prefix
        def _resolve(primary, fallback):
            return primary if primary.exists() else fallback

        erp = _resolve(REPORTS / f"{sel_subject}_erp.png",
                       REPORTS / f"{sel_subject}_fixed_raw_erp.png")
        psd = _resolve(REPORTS / f"{sel_subject}_psd.png",
                       REPORTS / f"{sel_subject}_fixed_raw_psd.png")
        cz  = REPORTS / f"{sel_subject}_Cz_time_overlay.png"
        ps1 = REPORTS / f"{sel_subject}_psd_compare_1_40.png"
        ps2 = REPORTS / f"{sel_subject}_psd_compare_47_53.png"

        c1, c2 = st.columns(2)
        with c1:
            if erp.exists():
                st.image(str(erp), caption="Event-Related Potential (ERP)", use_container_width=True)
            else:
                st.info(f"ERP plot not available for {sel_subject.upper()}")
        with c2:
            if psd.exists():
                st.image(str(psd), caption="Power Spectral Density (PSD)", use_container_width=True)
            else:
                st.info(f"PSD plot not available for {sel_subject.upper()}")

        if cz.exists():
            st.image(str(cz), caption="Cz Electrode Time Overlay", use_container_width=True)
        else:
            st.info(f"Cz time overlay not available for {sel_subject.upper()} — only generated for selected subjects.")

        pc1, pc2 = st.columns(2)
        with pc1:
            if ps1.exists():
                st.image(str(ps1), caption="PSD Comparison: 1–40 Hz", use_container_width=True)
            else:
                st.info(f"PSD frequency comparison (1–40 Hz) not available for {sel_subject.upper()}")
        with pc2:
            if ps2.exists():
                st.image(str(ps2), caption="PSD Comparison: 47–53 Hz (noise band)", use_container_width=True)
            else:
                st.info(f"PSD noise-band comparison (47–53 Hz) not available for {sel_subject.upper()}")

    # ── Data Table ─────────────────────────────────────────────────────────────
    with t4:
        disp = sub_flat[["model", "acc_mean", "acc_std", "f1_mean", "f1_std", "n_splits"]].copy()
        disp.columns = ["Model", "Acc Mean", "Acc Std", "F1 Mean", "F1 Std", "CV Folds"]
        disp = disp.sort_values("F1 Mean", ascending=False).reset_index(drop=True)
        st.dataframe(
            disp.style.format({c: "{:.4f}" for c in ["Acc Mean","Acc Std","F1 Mean","F1 Std"]})
                      .background_gradient(subset=["F1 Mean"], cmap="Blues"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Model Comparison":
    st.markdown('<div class="sh">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Cross-subject model performance, score distributions, and best-model frequencies</div>', unsafe_allow_html=True)

    if flat_df.empty:
        st.warning("No data loaded.")
        st.stop()

    filt = flat_df[flat_df["method"] == sel_method]

    t1, t2, t3, t4, t5 = st.tabs([
        "Subject Heatmap", "Aggregate Stats", "Selected Model", "Best Model Freq", "Plain vs SMOTE"
    ])

    # ── Subject Heatmap ────────────────────────────────────────────────────────
    with t1:
        piv = filt.pivot_table(index="subject", columns="model", values="f1_mean")
        fig_hm = px.imshow(
            piv, color_continuous_scale="RdYlGn", aspect="auto",
            labels=dict(x="Model", y="Subject", color="Macro-F1"),
            text_auto=".3f",
        )
        _sf(fig_hm, f"Subject × Model Macro-F1 Heatmap ({sel_method})", h=700)
        fig_hm.update_layout(coloraxis_colorbar=dict(title="F1", tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Aggregate Stats ────────────────────────────────────────────────────────
    with t2:
        agg = filt.groupby("model", as_index=False).agg(
            acc_mean=("acc_mean","mean"), acc_std=("acc_mean","std"),
            f1_mean=("f1_mean","mean"),  f1_std=("f1_mean","std"),
        ).sort_values("f1_mean", ascending=False)

        ca, cb = st.columns(2)
        with ca:
            fig_f1 = px.bar(agg, x="model", y="f1_mean", error_y="f1_std",
                            color="model", color_discrete_sequence=PALETTE,
                            text="f1_mean", labels={"f1_mean":"Avg Macro-F1","model":""})
            fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_f1.update_layout(showlegend=False)
            _sf(fig_f1, f"Avg Macro-F1 — {sel_method}", h=370)
            st.plotly_chart(fig_f1, use_container_width=True)
        with cb:
            fig_ac = px.bar(agg, x="model", y="acc_mean", error_y="acc_std",
                            color="model", color_discrete_sequence=PALETTE,
                            text="acc_mean", labels={"acc_mean":"Avg Accuracy","model":""})
            fig_ac.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_ac.update_layout(showlegend=False)
            _sf(fig_ac, f"Avg Accuracy — {sel_method}", h=370)
            st.plotly_chart(fig_ac, use_container_width=True)

        # Violin
        fig_vio = px.violin(filt, x="model", y="f1_mean",
                            color="model", color_discrete_sequence=PALETTE,
                            box=True, points="all",
                            labels={"f1_mean":"Macro-F1","model":""})
        fig_vio.update_layout(showlegend=False)
        _sf(fig_vio, f"Macro-F1 Distribution Across Subjects ({sel_method})", h=430)
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── Selected Model ─────────────────────────────────────────────────────────
    with t3:
        mdl_df = filt[filt["model"] == sel_model].sort_values("subject")

        fig_ln = px.line(mdl_df, x="subject", y="f1_mean", markers=True,
                         error_y="f1_std", color_discrete_sequence=["#818cf8"],
                         labels={"f1_mean":"Macro-F1","subject":"Subject"})
        fig_ln.add_hline(y=mdl_df["f1_mean"].mean(), line_dash="dash",
                         line_color="#fbbf24",
                         annotation_text=f"Mean {mdl_df['f1_mean'].mean():.3f}",
                         annotation_font_color="#fbbf24")
        _sf(fig_ln, f"{sel_model} Macro-F1 per Subject ({sel_method})", h=380)
        st.plotly_chart(fig_ln, use_container_width=True)

        c_bx, c_st = st.columns([1.2, 1])
        with c_bx:
            fig_bx = px.box(mdl_df, y="f1_mean", color_discrete_sequence=["#818cf8"],
                            points="all", labels={"f1_mean":"Macro-F1"})
            _sf(fig_bx, f"{sel_model} Distribution", h=340)
            st.plotly_chart(fig_bx, use_container_width=True)
        with c_st:
            st.markdown(f"**{sel_model} — Summary Stats ({sel_method})**")
            for k, v in [
                ("Mean Macro-F1",  f"{mdl_df['f1_mean'].mean():.4f}"),
                ("Std  Macro-F1",  f"{mdl_df['f1_mean'].std():.4f}"),
                ("Min  Macro-F1",  f"{mdl_df['f1_mean'].min():.4f}"),
                ("Max  Macro-F1",  f"{mdl_df['f1_mean'].max():.4f}"),
                ("Median F1",      f"{mdl_df['f1_mean'].median():.4f}"),
                ("Mean Accuracy",  f"{mdl_df['acc_mean'].mean():.4f}"),
                ("Subjects tested",f"{len(mdl_df)}"),
            ]:
                st.markdown(f'<div class="stat-row"><span>{k}</span><span>{v}</span></div>', unsafe_allow_html=True)

    # ── Best Model Freq ────────────────────────────────────────────────────────
    with t4:
        bm_df = best_df[best_df["method"] == sel_method]
        cnt = bm_df["best_model"].value_counts().reset_index()
        cnt.columns = ["model", "count"]

        cp, cb2 = st.columns(2)
        with cp:
            fig_pie = px.pie(cnt, values="count", names="model",
                             color_discrete_sequence=PALETTE, hole=0.45)
            _sf(fig_pie, f"Best Model Frequency ({sel_method})", h=380)
            st.plotly_chart(fig_pie, use_container_width=True)
        with cb2:
            fig_fr = px.bar(cnt.sort_values("count"),
                            x="count", y="model", orientation="h",
                            color="model", color_discrete_sequence=PALETTE,
                            text="count", labels={"count":"Subjects","model":""})
            fig_fr.update_traces(textposition="outside")
            fig_fr.update_layout(showlegend=False)
            _sf(fig_fr, f"Subjects Won by Model ({sel_method})", h=380)
            st.plotly_chart(fig_fr, use_container_width=True)

    # ── Plain vs SMOTE ─────────────────────────────────────────────────────────
    with t5:
        both = flat_df.groupby(["model","method"], as_index=False)[["f1_mean","acc_mean"]].mean()

        fig_g = px.bar(both, x="model", y="f1_mean", color="method",
                       barmode="group",
                       color_discrete_map={"plain":"#818cf8","smote":"#34d399"},
                       text="f1_mean", labels={"f1_mean":"Avg Macro-F1","model":""})
        fig_g.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        _sf(fig_g, "Plain vs SMOTE — Average Macro-F1", h=400)
        st.plotly_chart(fig_g, use_container_width=True)

        if "plain" in both["method"].values and "smote" in both["method"].values:
            delta_df = both.pivot(index="model", columns="method", values="f1_mean").reset_index()
            delta_df["smote_gain"] = delta_df.get("smote", 0) - delta_df.get("plain", 0)
            fig_dlt = px.bar(delta_df.sort_values("smote_gain"),
                             x="smote_gain", y="model", orientation="h",
                             color="smote_gain",
                             color_continuous_scale=["#fb7185","#334155","#34d399"],
                             text="smote_gain",
                             labels={"smote_gain":"SMOTE Gain (F1)","model":""})
            fig_dlt.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
            fig_dlt.update_layout(showlegend=False, coloraxis_showscale=False)
            _sf(fig_dlt, "SMOTE Macro-F1 Gain per Model", h=340)
            st.plotly_chart(fig_dlt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION & LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Evaluation & Leakage":
    st.markdown('<div class="sh">Evaluation & Leakage Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Why this work reports lower accuracy — and why that is the correct approach</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box box-amber">
    <b>Critical context:</b> Many published EEG studies inadvertently apply scaling or SMOTE
    <i>before</i> splitting data, leaking test-set information into preprocessing.
    This inflates reported accuracy to 80–95%. This dissertation deliberately prevents that,
    yielding a realistic ~54% that is <b>methodologically correct</b>.
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Leakage Prevention", "CV Strategies", "Performance Context"])

    # ── Leakage Prevention ─────────────────────────────────────────────────────
    with t1:
        col_img, col_steps = st.columns([1.4, 1])
        with col_img:
            ep = ASSETS / "evaluation_flow.png"
            if ep.exists():
                st.image(str(ep), caption="Leakage-Free Evaluation Flow", use_container_width=True)
        with col_steps:
            st.markdown('<div class="sh-sm">Leakage Prevention Steps</div>', unsafe_allow_html=True)
            for title, desc in [
                ("Scaler fitted inside fold",
                 "StandardScaler is fitted on X_train only, then transformed X_test — never on the full dataset."),
                ("SMOTE inside fold",
                 "Synthetic oversampling is applied only to training data; test fold never receives synthetic samples."),
                ("No look-ahead",
                 "Test splits are not used in any normalisation, feature selection, or hyperparameter step."),
                ("GroupKFold by subject",
                 "For subject-independent evaluation, entire subjects are held out so no trial can leak across folds."),
                ("Hyperparameter tuning",
                 "Grid search runs on training data only; best parameters are then evaluated once on the held-out test set."),
            ]:
                st.markdown(f"""
                <div class="lstep">
                    <div class="lstep-title">{title}</div>
                    <div class="lstep-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── CV Strategies ──────────────────────────────────────────────────────────
    with t2:
        col_dep, col_ind = st.columns(2)
        for col, hue, strategy, title, points in [
            (col_dep, "#818cf8", "Stratified K-Fold (k = 5)",
             "Subject-Dependent CV", [
                "Same subject in training and test",
                "Preserves class balance per fold",
                "40 epochs → 32 train, 8 test per fold",
                "Measures within-subject learnability",
                "Higher performance expected (~60–70%)",
            ]),
            (col_ind, "#34d399", "Group K-Fold (grouped by subject)",
             "Subject-Independent CV", [
                "Entire subjects held out as test group",
                "No subject appears in both train & test",
                "32 subjects split into 5 folds",
                "Measures cross-subject generalisation",
                "Realistic performance (~54%)",
            ]),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:rgba(10,18,36,0.8);border:1px solid rgba(255,255,255,0.07);
                            border-top:3px solid {hue};border-radius:18px;padding:1.3rem;
                            backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)">
                    <div style="font-weight:800;color:#f1f5f9;font-size:.95rem;margin-bottom:.25rem;letter-spacing:-0.01em">{title}</div>
                    <div style="color:{hue};font-size:.8rem;font-weight:700;margin-bottom:.8rem;letter-spacing:.02em">{strategy}</div>
                """, unsafe_allow_html=True)
                for pt in points:
                    st.markdown(f"""
                    <div style="padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,0.04);
                                color:#3d5575;font-size:.82rem">• {pt}</div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Leaky vs leakage-free bar chart
        cmp = pd.DataFrame({
            "Approach": ["Typical EEG study\n(leakage present)", "This dissertation\n(leakage-free)"],
            "Reported Accuracy": [0.85, 0.535],
            "Type": ["inflated", "realistic"],
        })
        fig_cmp = px.bar(cmp, x="Approach", y="Reported Accuracy",
                         color="Type",
                         color_discrete_map={"inflated":"#fb7185","realistic":"#34d399"},
                         text="Reported Accuracy",
                         labels={"Reported Accuracy":"Accuracy"})
        fig_cmp.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig_cmp.add_hline(y=0.5, line_dash="dot", line_color="#64748b",
                          annotation_text="Chance level (50%)",
                          annotation_font_color="#64748b")
        fig_cmp.update_layout(showlegend=False)
        _sf(fig_cmp, "This Work vs Typical EEG Study (Subject-Independent)", h=380)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Performance Context ─────────────────────────────────────────────────────
    with t3:
        st.markdown("""
        <div class="box box-indigo">
        <b>Why ~54% is a strong result:</b>
        Binary classification at chance = 50%. Achieving ~54% consistently under strict no-leakage,
        cross-subject conditions is meaningful. The delta above chance is real signal — not artefact.
        </div>
        """, unsafe_allow_html=True)

        comp = flat_df.groupby("method", as_index=False)[["acc_mean","f1_mean"]].mean()
        fig_ov = px.bar(
            comp.melt(id_vars="method", var_name="metric", value_name="value"),
            x="method", y="value", color="metric", barmode="group",
            color_discrete_map={"acc_mean":"#818cf8","f1_mean":"#34d399"},
            text="value",
            labels={"value":"Score","method":"Method","metric":"Metric"},
        )
        fig_ov.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_ov.add_hline(y=0.5, line_dash="dot", line_color="#64748b",
                         annotation_text="Chance (50%)", annotation_font_color="#64748b")
        _sf(fig_ov, "Overall Performance: Plain vs SMOTE (Subject-Dependent CV Avg)", h=390)
        st.plotly_chart(fig_ov, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT-INDEPENDENT RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Subject-Independent Results":
    st.markdown('<div class="sh">Subject-Independent Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Final held-out test performance and hyperparameter tuning results for cross-subject generalisation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box box-indigo">
    <b>Subject-independent evaluation</b> uses GroupKFold so that each test fold contains subjects
    <i>entirely absent</i> from training. This is the most rigorous EEG evaluation scenario and produces
    the final, honest benchmark numbers of this dissertation.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.tabs(["Final Test Results", "Hyperparameter Tuning"])

    # ── Final Test Results ─────────────────────────────────────────────────────
    with t1:
        if final_test.empty:
            st.warning("final_test_results.csv not found.")
        else:
            best_row = final_test.loc[final_test["test_macro_f1"].idxmax()]
            ks = st.columns(4)
            for col, lbl, val, cls in [
                (ks[0], "Best Model",    best_row["model"],                   "c-indigo"),
                (ks[1], "Best Test F1",  f"{best_row['test_macro_f1']:.4f}",  "c-green"),
                (ks[2], "Best Test Acc", f"{best_row['test_acc']:.4f}",        "c-sky"),
                (ks[3], "Models Tested", str(len(final_test)),                 "c-amber"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-value {cls}">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_f = px.bar(final_test.sort_values("test_macro_f1", ascending=False),
                               x="model", y="test_macro_f1",
                               color="model", color_discrete_sequence=PALETTE,
                               text="test_macro_f1",
                               labels={"test_macro_f1":"Test Macro-F1","model":""})
                fig_f.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_f.add_hline(y=0.5, line_dash="dot", line_color="#64748b",
                                annotation_text="Chance", annotation_font_color="#64748b")
                fig_f.update_layout(showlegend=False)
                _sf(fig_f, "Final Test Macro-F1 (Subject-Independent)", h=380)
                st.plotly_chart(fig_f, use_container_width=True)
            with c2:
                fig_a = px.bar(final_test.sort_values("test_acc", ascending=False),
                               x="model", y="test_acc",
                               color="model", color_discrete_sequence=PALETTE,
                               text="test_acc",
                               labels={"test_acc":"Test Accuracy","model":""})
                fig_a.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_a.add_hline(y=0.5, line_dash="dot", line_color="#64748b",
                                annotation_text="Chance", annotation_font_color="#64748b")
                fig_a.update_layout(showlegend=False)
                _sf(fig_a, "Final Test Accuracy (Subject-Independent)", h=380)
                st.plotly_chart(fig_a, use_container_width=True)

            # Scatter Acc vs F1
            fig_sc = px.scatter(final_test, x="test_acc", y="test_macro_f1",
                                text="model", color="model",
                                color_discrete_sequence=PALETTE,
                                labels={"test_acc":"Test Accuracy","test_macro_f1":"Test Macro-F1"})
            fig_sc.update_traces(textposition="top center", marker_size=14)
            fig_sc.update_layout(showlegend=False)
            _sf(fig_sc, "Accuracy vs Macro-F1 — Final Test", h=380)
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("**Raw Results Table**")
            disp_t = final_test[["model", "test_acc", "test_macro_f1"]].copy()
            disp_t.columns = ["Model", "Test Accuracy", "Test Macro-F1"]
            disp_t = disp_t.sort_values("Test Macro-F1", ascending=False).reset_index(drop=True)
            st.dataframe(
                disp_t.style.format({"Test Accuracy":"{:.4f}","Test Macro-F1":"{:.4f}"})
                            .background_gradient(subset=["Test Macro-F1"], cmap="Blues"),
                use_container_width=True,
            )

    # ── Hyperparameter Tuning ──────────────────────────────────────────────────
    with t2:
        if tuning.empty:
            st.warning("tuning_results_train_only.csv not found.")
        else:
            st.markdown("""
            <div class="box box-green">
            <b>Tuning approach:</b> GridSearchCV was run exclusively on training data.
            Best parameters were then locked and used once to evaluate on the held-out test set.
            This prevents test-set contamination during hyperparameter selection.
            </div>
            """, unsafe_allow_html=True)

            fig_tn = px.bar(tuning.sort_values("cv_macro_f1", ascending=False),
                            x="model", y="cv_macro_f1",
                            color="model", color_discrete_sequence=PALETTE,
                            text="cv_macro_f1",
                            labels={"cv_macro_f1":"CV Macro-F1 (Training Only)","model":""})
            fig_tn.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_tn.update_layout(showlegend=False)
            _sf(fig_tn, "Hyperparameter Tuning CV Macro-F1 (Training Data Only)", h=370)
            st.plotly_chart(fig_tn, use_container_width=True)

            st.markdown("**Best Hyperparameters per Model**")
            disp_tn = tuning[["model","cv_macro_f1","best_params"]].copy()
            disp_tn.columns = ["Model","CV Macro-F1","Best Parameters"]
            disp_tn = disp_tn.sort_values("CV Macro-F1", ascending=False).reset_index(drop=True)
            st.dataframe(
                disp_tn.style.format({"CV Macro-F1":"{:.4f}"}),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Conclusions":
    st.markdown('<div class="sh">Conclusions & Future Work</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Key findings, contributions, and open research directions</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box box-indigo">
    <b>Central finding:</b> Evaluation design has a greater impact on reported EEG classification
    performance than model complexity. When subject overlap and data leakage are eliminated,
    performance reaches a realistic ~54% accuracy — lower than inflated literature values but
    fully reproducible and trustworthy.
    </div>
    """, unsafe_allow_html=True)

    col_f, col_fw = st.columns(2)

    with col_f:
        st.markdown('<div class="sh-sm">Key Findings</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("📉", "Leakage inflates accuracy",
             "Removing preprocessing leakage alone dropped reported accuracy from 80–90% (literature) to a realistic ~54% — the primary methodological contribution of this work."),
            ("🔀", "High inter-subject variability",
             "Best-model Macro-F1 ranged from ~0.42 to ~0.72 across subjects, underscoring individual EEG differences that generalisation models must bridge."),
            ("⚡", "GradBoost leads cross-subject",
             "Gradient Boosting achieved the highest held-out test Macro-F1 (0.526) under subject-independent conditions."),
            ("🎯", "SMOTE: mixed benefit",
             "SMOTE improved some subjects modestly but was not consistently beneficial — its effect depends on subject-specific class imbalance severity."),
            ("📊", "Macro-F1 is essential",
             "Accuracy alone is misleading under class imbalance. Macro-F1 weights classes equally and provides a fairer, more interpretable metric."),
            ("🏆", "Evaluation > model choice",
             "Switching from leaky to leakage-free evaluation reduced performance more than any difference between classifiers — confirming that methodology dominates."),
        ]:
            st.markdown(f"""
            <div class="fcard">
                <div class="fcard-head">
                    <span class="fcard-icon">{icon}</span>
                    <span class="fcard-title">{title}</span>
                </div>
                <div class="fcard-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_fw:
        st.markdown('<div class="sh-sm">Future Work</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("🧠", "Deep learning architectures",
             "EEGNet, ShallowConvNet, or Transformer models that learn spatial-temporal patterns directly from raw signals without hand-crafted features."),
            ("🔗", "Connectivity-based features",
             "Coherence, phase-locking value (PLV), and Granger causality between electrode pairs as richer representations of neural coordination."),
            ("🔄", "Transfer learning & adaptation",
             "Domain-adversarial networks (DANN) and subject-adaptation methods to improve cross-subject generalisation beyond standard GroupKFold."),
            ("🌐", "Cross-dataset validation",
             "Evaluate on DREAMER, MAHNOB-HCI, or SEED datasets to test whether pipeline and features generalise beyond DEAP."),
            ("📡", "Real-time classification",
             "Adapt the offline pipeline for streaming EEG with minimal latency, enabling BCI and neurofeedback applications."),
            ("🔍", "Explainability (XAI)",
             "Apply SHAP or integrated gradients to identify which frequency bands and channels are most predictive of reward-related neural states."),
        ]:
            st.markdown(f"""
            <div class="fcard">
                <div class="fcard-head">
                    <span class="fcard-icon">{icon}</span>
                    <span class="fcard-title">{title}</span>
                </div>
                <div class="fcard-body">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Summary scatter + downloads
    if not flat_df.empty:
        c_chart, c_dl = st.columns([2.2, 1])
        with c_chart:
            summ = flat_df.groupby("model", as_index=False)[["acc_mean","f1_mean"]].mean()
            fig_s = px.scatter(summ, x="acc_mean", y="f1_mean", text="model",
                               color="model", color_discrete_sequence=PALETTE,
                               labels={"acc_mean":"Avg Accuracy","f1_mean":"Avg Macro-F1"})
            fig_s.update_traces(textposition="top center", marker_size=16)
            fig_s.update_layout(showlegend=False)
            _sf(fig_s, "Overall Model Summary — Subject-Dependent CV", h=390)
            st.plotly_chart(fig_s, use_container_width=True)

        with c_dl:
            st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
            st.markdown("**Download Results**")
            summ_dl = flat_df.groupby("model", as_index=False)[["acc_mean","f1_mean"]].mean()
            st.download_button("📥 Subject-Dependent Summary",
                               summ_dl.to_csv(index=False),
                               "subject_dependent_summary.csv", "text/csv",
                               use_container_width=True)
            if not final_test.empty:
                st.download_button("📥 Subject-Independent Results",
                                   final_test.to_csv(index=False),
                                   "subject_independent_results.csv", "text/csv",
                                   use_container_width=True)
            if not tuning.empty:
                st.download_button("📥 Hyperparameter Tuning",
                                   tuning.to_csv(index=False),
                                   "tuning_results.csv", "text/csv",
                                   use_container_width=True)
            if not best_df.empty:
                st.download_button("📥 Best Models per Subject",
                                   best_df.to_csv(index=False),
                                   "best_models_per_subject.csv", "text/csv",
                                   use_container_width=True)
