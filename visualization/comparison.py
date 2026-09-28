"""
Model Performance & Corpus Comparison Visualization Generator.

Generates:
1. Metric comparison grouped bar charts (P@10, Recall, MAP, nDCG@10).
2. Dataset category distribution charts.
Saved to report/figures/ for documentation and Streamlit display.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import EVAL_RESULTS_CSV, METADATA_CSV, FIGURES_DIR


def generate_category_distribution_plot(save_path: Path = None):
    """Generate category document distribution plot."""
    if save_path is None:
        save_path = FIGURES_DIR / "category_distribution.png"

    if not METADATA_CSV.exists():
        return None

    df_meta = pd.read_csv(METADATA_CSV)
    cat_counts = df_meta["category_name"].value_counts()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(cat_counts))
    ax.barh(y_pos, cat_counts.values, color="#2b5c8f", edgecolor="#1e3d59")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_counts.index, fontsize=10)
    ax.invert_yaxis()  # top-down display
    ax.set_xlabel("Number of Documents", fontsize=12)
    ax.set_title("20 Newsgroups Document Distribution per Category", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"[+] Saved category distribution plot to: {save_path}")
    return fig


def generate_metric_comparison_plot(save_path: Path = None):
    """Generate static grouped bar chart comparing P@10, Recall, MAP, nDCG@10 across models."""
    if save_path is None:
        save_path = FIGURES_DIR / "metric_comparison.png"

    if not EVAL_RESULTS_CSV.exists():
        return None

    df_results = pd.read_csv(EVAL_RESULTS_CSV)

    metrics = ["precision_at_10", "recall", "map", "ndcg_at_10"]
    methods = df_results["method"].tolist()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.2

    colors = {
        "tfidf": "#1f77b4",
        "bm25": "#ff7f0e",
        "semantic": "#2ca02c",
        "hybrid": "#d62728"
    }

    labels = {
        "tfidf": "TF-IDF",
        "bm25": "BM25",
        "semantic": "Sentence Transformer",
        "hybrid": "Hybrid (RRF)"
    }

    for i, method in enumerate(methods):
        m_row = df_results[df_results["method"] == method].iloc[0]
        values = [m_row[m] for m in metrics]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=labels.get(method, method), color=colors.get(method, "#333"))
        # Add value labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=30)

    ax.set_title("Comparative Evaluation of Retrieval Models on 20 Newsgroups", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision@10", "Recall@100", "MAP", "nDCG@10"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"[+] Saved metric comparison plot to: {save_path}")
    return fig


def create_interactive_plotly_comparison(results_df: pd.DataFrame):
    """Generate interactive Plotly bar chart for Streamlit UI."""
    if results_df.empty:
        return None

    df_melted = results_df.melt(
        id_vars=["method"],
        value_vars=["precision_at_10", "recall", "map", "ndcg_at_10"],
        var_name="Metric",
        value_name="Score"
    )

    metric_name_map = {
        "precision_at_10": "Precision@10",
        "recall": "Recall",
        "map": "MAP",
        "ndcg_at_10": "nDCG@10"
    }
    df_melted["Metric"] = df_melted["Metric"].map(metric_name_map)

    method_name_map = {
        "tfidf": "TF-IDF",
        "bm25": "BM25",
        "semantic": "Sentence Transformer",
        "hybrid": "Hybrid RRF"
    }
    df_melted["method"] = df_melted["method"].map(method_name_map)

    fig = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="method",
        barmode="group",
        title="Comparative Evaluation across IR Metrics",
        text_auto=".4f",
        color_discrete_map={
            "TF-IDF": "#1f77b4",
            "BM25": "#ff7f0e",
            "Sentence Transformer": "#2ca02c",
            "Hybrid RRF": "#d62728"
        }
    )
    fig.update_layout(yaxis_range=[0, 1.0], legend_title_text="Retrieval Model")
    return fig


if __name__ == "__main__":
    generate_category_distribution_plot()
    generate_metric_comparison_plot()
