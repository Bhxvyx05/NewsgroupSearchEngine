"""
Streamlit Page 3: Quantitative Evaluation Dashboard.
Displays Precision@10, Recall, MAP, nDCG@10 metrics, interactive comparison charts, and PR-Curves.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import EVAL_RESULTS_CSV, FIGURES_DIR, EVALUATION_DIR
from visualization.comparison import create_interactive_plotly_comparison

PER_QUERY_RESULTS_CSV = EVALUATION_DIR / "per_query_results.csv"

st.set_page_config(page_title="Evaluation - 20 Newsgroups IR", page_icon="📊", layout="wide")

st.title("📊 Quantitative Evaluation Dashboard")
st.markdown("Rigorous benchmarking of TF-IDF, BM25, Sentence Transformer, and Hybrid RRF across 200 evaluation queries.")

if not EVAL_RESULTS_CSV.exists():
    st.warning("⚠️ Evaluation benchmarks have not yet been executed. Please run `python evaluation/evaluate.py` to generate results.")
else:
    df_results = pd.read_csv(EVAL_RESULTS_CSV)

    # Highlight top performing model
    best_row = df_results.sort_values("ndcg_at_10", ascending=False).iloc[0]

    st.subheader("🏆 Overall Benchmark Metrics (Best Model: " + best_row["method"].upper() + ")")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@10", f"{best_row['precision_at_10']:.4f}", help="Top-10 document precision")
    c2.metric("Recall", f"{best_row['recall']:.4f}", help="Cutoff recall across corpus")
    c3.metric("MAP", f"{best_row['map']:.4f}", help="Mean Average Precision")
    c4.metric("nDCG@10", f"{best_row['ndcg_at_10']:.4f}", help="Normalized Discounted Cumulative Gain at 10")

    st.divider()

    st.subheader("📋 Model Comparison Table")
    st.dataframe(
        df_results.style.highlight_max(axis=0, color="#d4edda", subset=["precision_at_10", "recall", "map", "ndcg_at_10"]),
        use_container_width=True
    )

    st.divider()

    col_chart, col_pr = st.columns([1, 1])

    with col_chart:
        st.subheader("📊 Metric Comparison Bar Chart")
        plotly_fig = create_interactive_plotly_comparison(df_results)
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True)

    with col_pr:
        st.subheader("📈 Precision-Recall Curves")
        pr_img = FIGURES_DIR / "precision_recall.png"
        if pr_img.exists():
            st.image(str(pr_img), use_container_width=True)
        else:
            st.info("Run `python visualization/precision_recall.py` to generate PR curve image.")

    st.divider()

    st.subheader("🔍 Per-Query Detailed Metrics")
    if PER_QUERY_RESULTS_CSV.exists():
        df_per_query = pd.read_csv(PER_QUERY_RESULTS_CSV)
        
        categories = ["All"] + sorted(df_per_query["category"].unique().tolist())
        selected_cat = st.selectbox("Filter Per-Query Results by Category:", categories)

        if selected_cat != "All":
            df_filtered = df_per_query[df_per_query["category"] == selected_cat]
        else:
            df_filtered = df_per_query

        st.dataframe(df_filtered, use_container_width=True, height=350)
