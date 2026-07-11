"""Credit Card Fraud Detection — Streamlit Dashboard.

Run with:
    streamlit run app.py
Make sure credit_card_fraud_eda.py has been executed at least once so that
fraud_detection_model.pkl, results_metadata.json, and plots/ all exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "fraud_detection_model.pkl"
SCALER_PATH = BASE / "scaler.pkl"
META_PATH = BASE / "results_metadata.json"
PLOTS = BASE / "plots"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] .stRadio label { font-size: 1rem; padding: 6px 0; }
        [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
            font-size: 0.78rem; color: #94a3b8 !important;
        }

        /* ── Main ── */
        .main { background-color: #0f172a; }
        h1, h2, h3, h4 { color: #f1f5f9 !important; }
        p, li, .stMarkdown { color: #cbd5e1; }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px 22px;
        }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.82rem; }
        [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.7rem; font-weight: 700; }

        /* ── Dataframe ── */
        .stDataFrame { border-radius: 10px; overflow: hidden; }

        /* ── Section header banner ── */
        .section-header {
            background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
            border-radius: 10px;
            padding: 14px 24px;
            margin-bottom: 24px;
        }
        .section-header h2 { color: #ffffff !important; margin: 0; }

        /* ── Highlight box ── */
        .highlight-box {
            background: #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 12px 0;
        }

        /* ── Best model badge ── */
        .best-badge {
            background: linear-gradient(90deg, #16a34a, #22c55e);
            color: #ffffff !important;
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        /* ── Info/warning boxes ── */
        .info-box {
            background: #172554;
            border: 1px solid #1d4ed8;
            border-radius: 10px;
            padding: 16px 20px;
        }
        .warn-box {
            background: #451a03;
            border: 1px solid #c2410c;
            border-radius: 10px;
            padding: 16px 20px;
        }
        div[data-testid="stExpander"] {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load model (and scaler if it exists) from disk."""
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


@st.cache_data
def load_metadata() -> dict:
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)


def plotly_theme() -> dict:
    return dict(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font_color="#e2e8f0",
        colorway=["#38bdf8", "#f97316", "#a78bfa", "#34d399", "#f43f5e"],
    )


def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><h2>{icon} {title}</h2></div>',
        unsafe_allow_html=True,
    )


def show_plot(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot not found: {path.name}. Run `python credit_card_fraud_eda.py` first.")


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dataset Analysis",
            "⚖️ SMOTE",
            "🤖 Model Performance",
            "📈 Confusion Matrices",
            "📊 Feature Importance",
            "🔍 Prediction",
            "📋 Project Summary",
        ],
    )
    st.markdown("---")
    meta = load_metadata()
    if meta:
        st.markdown("**Best Model**")
        st.markdown(f"`{meta.get('best_model', 'N/A')}`")
        st.markdown(f"**F1 Score:** `{meta.get('best_f1', 0):.4f}`")

model, scaler = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 0 20px 0;">
            <h1 style="font-size:3rem; background: linear-gradient(90deg,#38bdf8,#818cf8);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🛡️ Credit Card Fraud Detection
            </h1>
            <p style="color:#94a3b8; font-size:1.1rem;">
                End-to-end ML pipeline · EDA · SMOTE · Multi-model evaluation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown("### 📌 Project Overview")
    st.markdown(
        """
        <div class="highlight-box">
        Credit card fraud costs financial institutions billions of dollars annually.
        This project builds a robust fraud detection system on the highly imbalanced
        <b>Kaggle Credit Card Fraud dataset</b> (284 807 transactions, only 0.17 % fraudulent).
        We handle the imbalance with <b>SMOTE</b> and compare three classifiers —
        Logistic Regression, Decision Tree, and Random Forest — selecting the best
        model by F1-score, which balances precision and recall appropriately for fraud use-cases.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Workflow ──────────────────────────────────────────────────────────────
    st.markdown("### 🔄 Project Workflow")
    steps = [
        ("1️⃣", "Data Loading", "Read the raw CSV into a pandas DataFrame"),
        ("2️⃣", "EDA", "Inspect shape, dtypes, statistics, missing values, duplicates"),
        ("3️⃣", "Preprocessing", "Drop 1 081 duplicate rows; validate data quality"),
        ("4️⃣", "Train/Test Split", "80 / 20 stratified split to preserve class ratio"),
        ("5️⃣", "SMOTE", "Oversample the minority (fraud) class in training data"),
        ("6️⃣", "Model Training", "Logistic Regression · Decision Tree · Random Forest"),
        ("7️⃣", "Evaluation", "Accuracy · Precision · Recall · F1 · ROC-AUC + confusion matrices"),
        ("8️⃣", "Best Model", "Select winner by F1-score; persist with joblib"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="background:#1e293b; border:1px solid #334155;
                            border-radius:10px; padding:14px; margin:6px 0; min-height:110px;">
                    <div style="font-size:1.6rem;">{icon}</div>
                    <div style="color:#38bdf8; font-weight:700; margin:6px 0 4px 0;">{title}</div>
                    <div style="color:#94a3b8; font-size:0.82rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Technologies ──────────────────────────────────────────────────────────
    st.markdown("### 🛠️ Technologies Used")
    techs = {
        "Python 3.12": "Core language",
        "pandas": "Data manipulation",
        "NumPy": "Numerical operations",
        "scikit-learn": "ML models & metrics",
        "imbalanced-learn": "SMOTE oversampling",
        "joblib": "Model serialisation",
        "Matplotlib / Seaborn": "Static visualisations",
        "Plotly": "Interactive charts",
        "Streamlit": "Dashboard framework",
    }
    t_cols = st.columns(3)
    for i, (tech, role) in enumerate(techs.items()):
        with t_cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px;
                            padding:10px 14px; margin:4px 0; display:flex; gap:10px; align-items:center;">
                    <span style="color:#38bdf8; font-weight:700;">{tech}</span>
                    <span style="color:#64748b; font-size:0.82rem;">— {role}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analysis":
    section_header("📊", "Dataset Analysis")

    if not meta:
        st.error("results_metadata.json not found. Run `python credit_card_fraud_eda.py` first.")
        st.stop()

    raw = meta["raw_shape"]
    cleaned = meta["cleaned_shape"]

    # ── Key metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Raw Rows", f"{raw[0]:,}")
    m2.metric("Cleaned Rows", f"{cleaned[0]:,}")
    m3.metric("Features", str(meta["num_features"]))
    m4.metric("Missing Values", str(meta["missing_values"]))
    m5.metric("Duplicates Removed", f"{meta['duplicate_rows']:,}")

    st.markdown("---")

    # ── Class counts ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔢 Class Distribution")
        fraud = meta["fraud_count"]
        genuine = meta["genuine_count"]
        fig = go.Figure(
            go.Bar(
                x=["Genuine (0)", "Fraud (1)"],
                y=[genuine, fraud],
                marker_color=["#38bdf8", "#f97316"],
                text=[f"{genuine:,}", f"{fraud:,}"],
                textposition="outside",
            )
        )
        fig.update_layout(
            **plotly_theme(),
            title="Transaction Counts by Class",
            xaxis_title="Class",
            yaxis_title="Count",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        g_pct = genuine / (genuine + fraud) * 100
        f_pct = fraud / (genuine + fraud) * 100
        fig2 = go.Figure(
            go.Pie(
                labels=["Genuine", "Fraud"],
                values=[genuine, fraud],
                hole=0.55,
                marker_colors=["#38bdf8", "#f97316"],
                textinfo="label+percent",
            )
        )
        fig2.update_layout(
            **plotly_theme(),
            title="Class Proportion",
            height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Saved EDA plot ────────────────────────────────────────────────────────
    st.markdown("#### 🖼️ EDA Plot — Class Distribution (Generated by Training Script)")
    show_plot(PLOTS / "class_distribution.png")

    # ── Data quality expander ─────────────────────────────────────────────────
    with st.expander("📋 Data Quality Summary"):
        quality = {
            "Total Raw Rows": f"{raw[0]:,}",
            "Total Cleaned Rows": f"{cleaned[0]:,}",
            "Number of Features": meta["num_features"],
            "Missing Values (all cols)": meta["missing_values"],
            "Duplicate Rows Found": f"{meta['duplicate_rows']:,}",
            "Fraud Transactions": f"{fraud:,}  ({f_pct:.4f}%)",
            "Genuine Transactions": f"{genuine:,}  ({g_pct:.4f}%)",
        }
        df_q = pd.DataFrame(quality.items(), columns=["Attribute", "Value"])
        st.dataframe(df_q, use_container_width=True, hide_index=True)

    with st.expander("📝 Feature Names"):
        feat_df = pd.DataFrame(
            {"#": range(1, len(meta["feature_names"]) + 1), "Feature": meta["feature_names"]}
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SMOTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ SMOTE":
    section_header("⚖️", "SMOTE — Synthetic Minority Oversampling Technique")

    st.markdown(
        """
        <div class="highlight-box">
        <b>What is SMOTE?</b><br>
        The raw training set contains roughly <b>226 602 genuine</b> transactions
        and only <b>378 fraudulent</b> ones — a ratio of ~600 : 1.
        Training a model on such skewed data causes it to almost always predict "genuine"
        and still achieve >99 % accuracy, while completely missing actual fraud.<br><br>
        SMOTE fixes this by <b>synthesising new minority-class samples</b> rather than
        simply duplicating existing ones.  For each fraud sample it picks <i>k</i> nearest
        neighbours in feature space and creates new points along the line segments between
        them.  After SMOTE both classes have <b>226 602 samples</b>, giving the model a
        balanced view of what fraud looks like.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if meta:
        st.markdown("#### 📊 Class Counts Before and After SMOTE")
        fraud_before = meta["fraud_count"]
        genuine_before = meta["genuine_count"]
        # After SMOTE both classes equal the genuine count in training (80 % split)
        genuine_train = int(genuine_before * 0.8)
        fraud_after = genuine_train

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Before SMOTE",
                x=["Genuine (0)", "Fraud (1)"],
                y=[genuine_train, int(fraud_before * 0.8)],
                marker_color="#38bdf8",
            )
        )
        fig.add_trace(
            go.Bar(
                name="After SMOTE",
                x=["Genuine (0)", "Fraud (1)"],
                y=[genuine_train, fraud_after],
                marker_color="#f97316",
            )
        )
        fig.update_layout(
            **plotly_theme(),
            barmode="group",
            title="Training Set Class Distribution Before vs After SMOTE",
            xaxis_title="Class",
            yaxis_title="Sample Count",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🖼️ Saved SMOTE Balancing Plot")
    show_plot(PLOTS / "smote_balancing.png")

    st.markdown("#### 📖 Why SMOTE over Simple Oversampling?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="info-box">
            <b>✅ SMOTE Advantages</b><br><br>
            • Creates <i>new</i> synthetic data points, not copies<br>
            • Reduces risk of overfitting to minority class<br>
            • Produces smoother decision boundaries<br>
            • Only applied on training data (no data leakage)
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="warn-box">
            <b>⚠️ Simple Random Oversampling Pitfalls</b><br><br>
            • Duplicates existing minority samples exactly<br>
            • Model memorises those exact points (overfits)<br>
            • Decision boundary remains identical<br>
            • Generalisation on unseen fraud is weaker
            </div>
            """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    section_header("🤖", "Model Performance")

    if not meta or "model_results" not in meta:
        st.error("No model results found. Run `python credit_card_fraud_eda.py` first.")
        st.stop()

    results = meta["model_results"]
    df_r = pd.DataFrame(results)
    best_model_name = meta["best_model"]

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Model Comparison Table")

    def style_table(df: pd.DataFrame) -> pd.Styler:
        best_idx = df[df["Model"] == best_model_name].index
        numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        styled = df.style.format({c: "{:.4f}" for c in numeric_cols})
        styled = styled.apply(
            lambda row: [
                "background-color: #14532d; color: #86efac; font-weight: bold;"
                if row.name in best_idx
                else ""
                for _ in row
            ],
            axis=1,
        )
        styled = styled.background_gradient(subset=numeric_cols, cmap="Blues")
        return styled

    st.dataframe(style_table(df_r), use_container_width=True, hide_index=True)

    # ── Best model callout ────────────────────────────────────────────────────
    best_row = df_r[df_r["Model"] == best_model_name].iloc[0]
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#14532d,#166534);
                    border-radius:12px; padding:20px 24px; margin:16px 0;">
            <span style="font-size:1.1rem; color:#86efac; font-weight:700;">
                🏆 Best Model: {best_model_name}
            </span><br>
            <span style="color:#bbf7d0;">
                Selected because it achieved the highest <b>F1-Score ({best_row['F1-Score']:.4f})</b>,
                reflecting the best balance between Precision ({best_row['Precision']:.4f}) and
                Recall ({best_row['Recall']:.4f}) — critical for fraud detection where both
                false negatives (missed fraud) and false positives (wrongly blocked cards) are costly.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Interactive radar chart ───────────────────────────────────────────────
    st.markdown("#### 📡 Radar Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    colors = ["#38bdf8", "#f97316", "#a78bfa"]
    for i, row in df_r.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=metrics + [metrics[0]],
                fill="toself",
                name=row["Model"],
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)].replace(")", ", 0.15)").replace("rgb", "rgba")
                if colors[i % len(colors)].startswith("rgb")
                else colors[i % len(colors)],
                opacity=0.7,
            )
        )
    fig.update_layout(
        **plotly_theme(),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color="#64748b"),
            bgcolor="#0f172a",
        ),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Bar comparison ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Metric Comparison by Model")
    metric_sel = st.selectbox("Select metric", ["F1-Score", "Precision", "Recall", "Accuracy", "ROC-AUC"])
    fig2 = px.bar(
        df_r,
        x="Model",
        y=metric_sel,
        color="Model",
        text=metric_sel,
        color_discrete_sequence=["#38bdf8", "#f97316", "#a78bfa"],
    )
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig2.update_layout(**plotly_theme(), showlegend=False, height=360)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Confusion Matrices":
    section_header("📈", "Confusion Matrices")

    st.markdown(
        """
        <div class="highlight-box">
        A confusion matrix shows how many predictions the model got right vs wrong for each class.
        For fraud detection: <b>True Positives (TP)</b> = caught frauds ✅ &nbsp;|&nbsp;
        <b>False Negatives (FN)</b> = missed frauds ❌ (most costly) &nbsp;|&nbsp;
        <b>False Positives (FP)</b> = genuine transactions flagged as fraud ⚠️
        </div>
        """,
        unsafe_allow_html=True,
    )

    matrix_files = {
        "Logistic Regression": PLOTS / "confusion_matrix_logistic_regression.png",
        "Decision Tree": PLOTS / "confusion_matrix_decision_tree.png",
        "Random Forest": PLOTS / "confusion_matrix_random_forest.png",
    }
    best = meta.get("best_model", "")

    c1, c2, c3 = st.columns(3)
    for col, (model_name, path) in zip([c1, c2, c3], matrix_files.items()):
        with col:
            badge = '<span class="best-badge">✦ BEST</span>' if model_name == best else ""
            st.markdown(
                f"<h4 style='color:#e2e8f0;'>{model_name} {badge}</h4>",
                unsafe_allow_html=True,
            )
            show_plot(path)

    # ── Interactive plotly version using stored results ────────────────────────
    if meta and "model_results" in meta:
        st.markdown("---")
        st.markdown("#### 🔢 Confusion Matrix Breakdown (from stored metrics)")
        df_r2 = pd.DataFrame(meta["model_results"])
        total_neg = meta["genuine_count"]
        total_pos = meta["fraud_count"]
        test_neg = int(total_neg * 0.2)
        test_pos = int(total_pos * 0.2)

        for _, row in df_r2.iterrows():
            tp = int(round(row["Recall"] * test_pos))
            fn = test_pos - tp
            fp = int(round(tp / row["Precision"])) - tp if row["Precision"] > 0 else 0
            tn = test_neg - fp
            with st.expander(f"📊 {row['Model']} — TP={tp}  FN={fn}  FP={fp}  TN={tn:,}"):
                fig = px.imshow(
                    [[tn, fp], [fn, tp]],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Predicted 0", "Predicted 1"],
                    y=["Actual 0", "Actual 1"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                )
                fig.update_layout(**plotly_theme(), height=300)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Feature Importance":
    section_header("📊", "Feature Importance — Random Forest")

    st.markdown(
        """
        <div class="highlight-box">
        Random Forest computes <b>mean decrease in impurity</b> (Gini importance) across all
        trees for each feature.  Higher values indicate the feature has a stronger influence
        on the model's fraud/genuine decision.  PCA-anonymised features V1–V28 along with
        <b>Amount</b> and <b>Time</b> make up all 30 predictors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🖼️ Top 10 Feature Importances (saved plot)")
    show_plot(PLOTS / "feature_importance.png")

    # ── Live interactive chart if model is loaded ──────────────────────────────
    if model is not None and hasattr(model, "feature_importances_"):
        features = meta.get("feature_names", [])
        importances = model.feature_importances_
        if len(features) == len(importances):
            fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(10)

            st.markdown("#### 📊 Interactive Chart (live from loaded model)")
            fig = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                text="Importance",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(
                **plotly_theme(),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    elif model is None:
        st.info("Model not loaded — showing saved plot only.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediction":
    section_header("🔍", "Real-Time Fraud Prediction")

    if model is None:
        st.error("Model not found. Run `python credit_card_fraud_eda.py` first.")
        st.stop()

    feature_names = meta.get("feature_names", [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])

    def run_prediction(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (predictions, probabilities)."""
        if scaler is not None:
            arr = scaler.transform(arr)
        preds = model.predict(arr)
        proba = model.predict_proba(arr)[:, 1] if hasattr(model, "predict_proba") else None
        return preds, proba

    tab1, tab2 = st.tabs(["📂 Upload CSV", "⌨️ Manual Entry"])

    # ── Tab 1: CSV upload ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("Upload a CSV with the same feature columns as the training data (no *Class* column needed).")
        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if "Class" in df_up.columns:
                    df_up = df_up.drop(columns=["Class"])

                # Align columns
                missing_cols = [c for c in feature_names if c not in df_up.columns]
                extra_cols = [c for c in df_up.columns if c not in feature_names]

                if missing_cols:
                    st.warning(f"Missing columns (will be filled with 0): {missing_cols}")
                    for c in missing_cols:
                        df_up[c] = 0.0
                if extra_cols:
                    st.info(f"Extra columns ignored: {extra_cols}")
                    df_up = df_up[feature_names]
                else:
                    df_up = df_up[feature_names]

                preds, proba = run_prediction(df_up.values)
                df_up["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Genuine" for p in preds]
                if proba is not None:
                    df_up["Fraud Probability"] = (proba * 100).round(2)

                st.success(f"Processed {len(df_up):,} rows.")
                fraud_found = (preds == 1).sum()
                g1, g2, g3 = st.columns(3)
                g1.metric("Total Rows", f"{len(df_up):,}")
                g2.metric("🚨 Flagged as Fraud", str(fraud_found))
                g3.metric("✅ Genuine", str(len(df_up) - fraud_found))

                st.dataframe(df_up, use_container_width=True)

                csv_out = df_up.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # ── Tab 2: Manual entry ────────────────────────────────────────────────────
    with tab2:
        st.markdown("Enter feature values for a single transaction to get an instant prediction.")
        st.markdown(
            "<div class='highlight-box'>💡 Leave fields at 0 if unknown. "
            "PCA features V1–V28 are normalised; Amount is in USD.</div>",
            unsafe_allow_html=True,
        )

        with st.form("manual_prediction"):
            input_vals = {}
            cols_per_row = 5
            feat_chunks = [
                feature_names[i : i + cols_per_row]
                for i in range(0, len(feature_names), cols_per_row)
            ]
            for chunk in feat_chunks:
                row_cols = st.columns(len(chunk))
                for col, feat in zip(row_cols, chunk):
                    with col:
                        input_vals[feat] = st.number_input(feat, value=0.0, format="%.6f", key=f"inp_{feat}")

            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

        if submitted:
            arr = np.array([[input_vals[f] for f in feature_names]])
            preds, proba = run_prediction(arr)
            is_fraud = int(preds[0]) == 1
            fraud_prob = float(proba[0]) * 100 if proba is not None else None

            if is_fraud:
                st.markdown(
                    """
                    <div style="background:#450a0a; border:2px solid #dc2626; border-radius:12px;
                                padding:24px; text-align:center;">
                        <div style="font-size:3rem;">🚨</div>
                        <div style="color:#fca5a5; font-size:1.5rem; font-weight:700;">
                            FRAUDULENT TRANSACTION
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background:#052e16; border:2px solid #16a34a; border-radius:12px;
                                padding:24px; text-align:center;">
                        <div style="font-size:3rem;">✅</div>
                        <div style="color:#86efac; font-size:1.5rem; font-weight:700;">
                            GENUINE TRANSACTION
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if fraud_prob is not None:
                st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=fraud_prob,
                        title={"text": "Fraud Probability (%)", "font": {"color": "#e2e8f0"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                            "bar": {"color": "#dc2626" if is_fraud else "#16a34a"},
                            "steps": [
                                {"range": [0, 30], "color": "#052e16"},
                                {"range": [30, 70], "color": "#713f12"},
                                {"range": [70, 100], "color": "#450a0a"},
                            ],
                            "threshold": {
                                "line": {"color": "#fbbf24", "width": 3},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                        number={"font": {"color": "#e2e8f0"}, "suffix": "%"},
                    )
                )
                gauge.update_layout(**plotly_theme(), height=300)
                st.plotly_chart(gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Project Summary
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Project Summary":
    section_header("📋", "Project Summary")

    if not meta:
        st.error("results_metadata.json not found. Run `python credit_card_fraud_eda.py` first.")
        st.stop()

    # ── Hero metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Size", f"{meta['raw_shape'][0]:,} rows")
    m2.metric("Features Used", str(meta["num_features"]))
    m3.metric("Best Model", meta["best_model"])
    m4.metric("Best F1 Score", f"{meta['best_f1']:.4f}")

    st.markdown("---")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### ✅ Pipeline Checklist")
        steps_done = [
            "Dataset Loaded",
            "Exploratory Data Analysis (EDA)",
            "Missing Value Check",
            "Duplicate Removal",
            "Train / Test Split (80/20 stratified)",
            "SMOTE Applied (minority oversampling)",
            "Logistic Regression Trained",
            "Decision Tree Trained",
            "Random Forest Trained",
            "Models Evaluated (Accuracy, Precision, Recall, F1, ROC-AUC)",
            "Confusion Matrices Generated",
            "Feature Importance Plotted",
            "Best Model Selected by F1-Score",
            "Model Saved with joblib",
            "Results Metadata Persisted",
        ]
        for step in steps_done:
            st.markdown(
                f"<div style='padding:5px 0; color:#86efac;'>✓ {step}</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("#### 📊 Model Results at a Glance")
        df_r3 = pd.DataFrame(meta["model_results"])
        best = meta["best_model"]
        numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        fig = go.Figure()
        colors = ["#38bdf8", "#f97316", "#a78bfa"]
        for i, row in df_r3.iterrows():
            fig.add_trace(
                go.Bar(
                    name=row["Model"],
                    x=numeric_cols,
                    y=[row[c] for c in numeric_cols],
                    marker_color=colors[i % len(colors)],
                )
            )
        fig.update_layout(
            **plotly_theme(),
            barmode="group",
            title="All Models — All Metrics",
            yaxis_title="Score",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Dataset Stats")
        stats = {
            "Raw Rows": f"{meta['raw_shape'][0]:,}",
            "Cleaned Rows": f"{meta['cleaned_shape'][0]:,}",
            "Duplicates Removed": f"{meta['duplicate_rows']:,}",
            "Missing Values": str(meta["missing_values"]),
            "Fraud Transactions": f"{meta['fraud_count']:,}",
            "Genuine Transactions": f"{meta['genuine_count']:,}",
            "Imbalance Ratio": f"1 : {meta['genuine_count'] // max(meta['fraud_count'], 1)}",
        }
        for k, v in stats.items():
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                            padding:6px 12px; margin:3px 0; background:#1e293b;
                            border-radius:6px;">
                    <span style="color:#94a3b8;">{k}</span>
                    <span style="color:#38bdf8; font-weight:600;">{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#475569; padding:16px 0; font-size:0.85rem;">
            🛡️ Credit Card Fraud Detection Dashboard &nbsp;·&nbsp;
            Built with Streamlit &amp; Plotly &nbsp;·&nbsp;
            Model: scikit-learn Random Forest
        </div>
        """,
        unsafe_allow_html=True,
    )