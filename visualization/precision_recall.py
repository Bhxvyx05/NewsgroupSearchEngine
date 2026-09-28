"""
Precision-Recall Curve Visualization Generator.

Calculates aggregated Precision and Recall curves across varying retrieval cutoffs (K=1..100)
for TF-IDF, BM25, Sentence Transformer, and Hybrid RRF models.
Saves output figure to report/figures/precision_recall.png.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import QUERIES_CSV, QRELS_CSV, FIGURES_DIR
from retrieval.retrieval import search
from evaluation.metrics import precision_at_k, recall_at_k


def generate_precision_recall_curves(save_path: Path = None):
    """
    Compute Precision-Recall trade-off points across cutoffs K=[1, 5, 10, 20, 30, 50, 75, 100]
    for all 4 retrieval models and plot PR curves.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "precision_recall.png"

    queries_df = pd.read_csv(QUERIES_CSV)
    qrels_df = pd.read_csv(QRELS_CSV)

    relevant_map: Dict[str, set] = {}
    for q_id, group in qrels_df[qrels_df["relevance"] == 1].groupby("query_id"):
        relevant_map[q_id] = set(group["doc_id"])

    methods = ["tfidf", "bm25", "semantic", "hybrid"]
    cutoffs = [1, 5, 10, 20, 30, 50, 75, 100]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "tfidf": "#1f77b4",
        "bm25": "#ff7f0e",
        "semantic": "#2ca02c",
        "hybrid": "#d62728"
    }

    labels = {
        "tfidf": "TF-IDF (Cosine Sim)",
        "bm25": "BM25 (Okapi)",
        "semantic": "Sentence Transformer (all-MiniLM-L6-v2)",
        "hybrid": "Hybrid (Reciprocal Rank Fusion k=60)"
    }

    for method in methods:
        mean_precisions = []
        mean_recalls = []

        # Retrieve top 100 for each query
        for k in cutoffs:
            k_p = []
            k_r = []
            for _, row in queries_df.iterrows():
                q_id = row["query_id"]
                q_text = row["query"]
                rel_docs = relevant_map.get(q_id, set())

                results = search(q_text, method=method, top_k=k)
                retrieved_ids = [res["doc_id"] for res in results]

                p = precision_at_k(retrieved_ids, rel_docs, k=k)
                r = recall_at_k(retrieved_ids, rel_docs, k=k)
                k_p.append(p)
                k_r.append(r)

            mean_precisions.append(np.mean(k_p))
            mean_recalls.append(np.mean(k_r))

        ax.plot(
            mean_recalls, 
            mean_precisions, 
            marker="o", 
            linewidth=2, 
            color=colors[method], 
            label=labels[method]
        )

    ax.set_title("Precision-Recall Curves on 20 Newsgroups Corpus", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0.0, max(mean_recalls) * 1.15 if mean_recalls else 0.2])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"[+] Saved Precision-Recall curve plot to: {save_path}")
    return fig


if __name__ == "__main__":
    generate_precision_recall_curves()
