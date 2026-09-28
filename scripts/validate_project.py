"""
End-to-End System Integration & Regression Validation Script.

Validates that:
1. Data files (documents, metadata, queries, qrels) exist and are non-empty.
2. Preprocessing pipeline functions correctly.
3. TF-IDF, BM25, Semantic, and Hybrid RRF retrievers return valid non-empty results.
4. Unified search interface API works for all 4 methods and enforces consistent output schema.
5. Evaluation results CSV and performance metrics exist.
6. Report visualization figures exist.
7. Streamlit app modules import cleanly without error.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
from core.config import (
    DOCUMENTS_CSV, METADATA_CSV, QUERIES_CSV, QRELS_CSV,
    EVAL_RESULTS_CSV, FIGURES_DIR
)
from preprocessing.preprocess import preprocess_text
from retrieval.retrieval import search


def validate_entire_project():
    print("=" * 70)
    print("RUNNING END-TO-END PROJECT INTEGRATION & REGRESSION VALIDATION")
    print("=" * 70)

    # 1. Check Data Artifacts
    print("[Step 1/7] Verifying Data Artifacts...")
    for path, name in [(DOCUMENTS_CSV, "documents.csv"), (METADATA_CSV, "metadata.csv"), 
                       (QUERIES_CSV, "queries.csv"), (QRELS_CSV, "qrels.csv")]:
        assert path.exists(), f"Missing file: {name}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"File {name} is empty!"
        print(f"   [+] {name:<15}: {len(df):,} records")

    # 2. Check Preprocessing
    print("\n[Step 2/7] Verifying Preprocessing Pipeline...")
    sample_text = "From: test@domain.com\nSubject: Test\n\nNASA space shuttle launch in 1999."
    cleaned = preprocess_text(sample_text)
    assert "nasa" in cleaned and "shuttl" in cleaned, "Preprocessing failed text cleaning!"
    print(f"   [+] Preprocessing output verified: '{cleaned}'")

    # 3. Check Unified Retrieval API across all 4 methods
    print("\n[Step 3/7] Verifying Unified Retrieval API (`search(query, method, top_k)`)...")
    methods = ["tfidf", "bm25", "semantic", "hybrid"]
    test_query = "NASA space shuttle launch"

    required_keys = {"doc_id", "rank", "score", "category", "snippet", "text"}

    for method in methods:
        results = search(test_query, method=method, top_k=5)
        assert len(results) > 0, f"Method {method} returned zero search results for query '{test_query}'"
        first = results[0]
        assert required_keys.issubset(set(first.keys())), f"Method {method} missing standard schema keys!"
        assert first["doc_id"].startswith("doc_"), f"Method {method} doc_id invalid format: {first['doc_id']}"
        print(f"   [+] Method '{method:<8}': Top-1 -> {first['doc_id']} (Score: {first['score']:.4f}, Category: {first['category']})")

    # 4. Check Evaluation Output
    print("\n[Step 4/7] Verifying Evaluation Results...")
    assert EVAL_RESULTS_CSV.exists(), "Evaluation results.csv missing!"
    df_eval = pd.read_csv(EVAL_RESULTS_CSV)
    assert set(df_eval["method"]) == set(methods), "Evaluation results missing methods!"
    print("   [+] Evaluation Summary:")
    for _, row in df_eval.iterrows():
        print(f"       - {row['method']:<10}: P@10={row['precision_at_10']:.4f} | Recall={row['recall']:.4f} | MAP={row['map']:.4f} | nDCG@10={row['ndcg_at_10']:.4f}")

    # 5. Check Visualization Figures
    print("\n[Step 5/7] Verifying Generated Visualization Figures...")
    for fig_name in ["precision_recall.png", "metric_comparison.png", "category_distribution.png"]:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing visualization figure: {fig_name}"
        print(f"   [+] Figure found: {fig_name}")

    # 6. Check Streamlit Page Modules
    print("\n[Step 6/7] Verifying Streamlit App Imports...")
    import importlib
    import app
    importlib.import_module("pages.0_Home")
    importlib.import_module("pages.1_Search")
    importlib.import_module("pages.2_Compare")
    importlib.import_module("pages.3_Evaluation")
    importlib.import_module("pages.4_About")
    print("   [+] All Streamlit app launchers and page modules imported successfully without errors!")

    # 7. Final Sanity Confirmation
    print("\n" + "=" * 70)
    print("[+] ALL 7 INTEGRATION & REGRESSION CHECKS PASSED CLEANLY!")
    print("======================================================================")


if __name__ == "__main__":
    validate_entire_project()
