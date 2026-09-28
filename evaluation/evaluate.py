"""
Full Quantitative Evaluation Benchmark Runner.

Executes comparative benchmark across TF-IDF, BM25, Sentence Transformer, and Hybrid RRF
over all 200 queries in data/queries.csv against ground-truth qrels in data/qrels.csv.
Exports evaluation/results.csv, evaluation/per_query_results.csv, and evaluation/performance.csv.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import QUERIES_CSV, QRELS_CSV, EVALUATION_DIR, EVAL_RESULTS_CSV
from retrieval.retrieval import search
from evaluation.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k

PER_QUERY_RESULTS_CSV = EVALUATION_DIR / "per_query_results.csv"
PERFORMANCE_CSV = EVALUATION_DIR / "performance.csv"


def run_evaluation_suite():
    print("=" * 60)
    print("RUNNING COMPLETE QUANTITATIVE EVALUATION BENCHMARK")
    print("=" * 60)

    if not (QUERIES_CSV.exists() and QRELS_CSV.exists()):
        raise FileNotFoundError("queries.csv or qrels.csv missing. Run scripts/build_qrels.py first.")

    queries_df = pd.read_csv(QUERIES_CSV)
    qrels_df = pd.read_csv(QRELS_CSV)

    # Group relevant doc IDs by query_id for fast set lookup
    print("[+] Loading relevance judgments (qrels)...")
    relevant_map: Dict[str, set] = {}
    for q_id, group in qrels_df[qrels_df["relevance"] == 1].groupby("query_id"):
        relevant_map[q_id] = set(group["doc_id"])

    methods = ["tfidf", "bm25", "semantic", "hybrid"]
    summary_records = []
    per_query_records = []
    performance_records = []

    print(f"[+] Benchmarking {len(queries_df)} queries across {len(methods)} retrieval models...\n")

    for method in methods:
        p10_list = []
        rec_list = []
        ap_list = []
        ndcg10_list = []
        latencies = []

        print(f"--- Evaluating Method: {method.upper()} ---")

        for idx, row in queries_df.iterrows():
            q_id = row["query_id"]
            q_text = row["query"]
            rel_docs = relevant_map.get(q_id, set())

            start_time = time.perf_counter()
            # Retrieve top 100 documents for recall and MAP cutoff
            results = search(q_text, method=method, top_k=100)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_ids = [res["doc_id"] for res in results]

            p10 = precision_at_k(retrieved_ids, rel_docs, k=10)
            rec = recall_at_k(retrieved_ids, rel_docs, k=100)
            ap = average_precision(retrieved_ids, rel_docs, cutoff_k=100)
            ndcg10 = ndcg_at_k(retrieved_ids, rel_docs, k=10)

            p10_list.append(p10)
            rec_list.append(rec)
            ap_list.append(ap)
            ndcg10_list.append(ndcg10)

            per_query_records.append({
                "query_id": q_id,
                "category": row["category"],
                "method": method,
                "precision_at_10": round(p10, 4),
                "recall": round(rec, 4),
                "average_precision": round(ap, 4),
                "ndcg_at_10": round(ndcg10, 4),
                "latency_ms": round(elapsed_ms, 2)
            })

        mean_p10 = np.mean(p10_list)
        mean_rec = np.mean(rec_list)
        mean_map = np.mean(ap_list)
        mean_ndcg10 = np.mean(ndcg10_list)
        avg_latency = np.mean(latencies)

        summary_records.append({
            "method": method,
            "precision_at_10": round(mean_p10, 4),
            "recall": round(mean_rec, 4),
            "map": round(mean_map, 4),
            "ndcg_at_10": round(mean_ndcg10, 4)
        })

        performance_records.append({
            "method": method,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2)
        })

        print(f"    P@10   : {mean_p10:.4f}")
        print(f"    Recall : {mean_rec:.4f}")
        print(f"    MAP    : {mean_map:.4f}")
        print(f"    nDCG@10: {mean_ndcg10:.4f}")
        print(f"    Avg Latency: {avg_latency:.2f} ms\n")

    # Export DataFrames
    df_summary = pd.DataFrame(summary_records)
    df_per_query = pd.DataFrame(per_query_records)
    df_perf = pd.DataFrame(performance_records)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(EVAL_RESULTS_CSV, index=False)
    df_per_query.to_csv(PER_QUERY_RESULTS_CSV, index=False)
    df_perf.to_csv(PERFORMANCE_CSV, index=False)

    print("=" * 60)
    print("SUMMARY COMPARATIVE EVALUATION RESULTS")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print(f"\n[+] Exported summary results to : {EVAL_RESULTS_CSV}")
    print(f"[+] Exported per-query results to: {PER_QUERY_RESULTS_CSV}")
    print(f"[+] Exported performance stats to: {PERFORMANCE_CSV}")

    return df_summary


if __name__ == "__main__":
    run_evaluation_suite()
