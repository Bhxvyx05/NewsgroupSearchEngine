"""
Reciprocal Rank Fusion (RRF) Validation Script.

Performs mathematical unit tests on RRF scoring formula (k=60)
and tests full hybrid retrieval using the unified search API.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.rrf import rrf_fusion
from retrieval.retrieval import search


def run_rrf_math_unit_test():
    """Mathematical verification of RRF formula using synthetic rank lists."""
    print("=" * 60)
    print("RUNNING RRF MATHEMATICAL UNIT TEST")
    print("=" * 60)

    # BM25: D1(r=1), D2(r=2), D3(r=3)
    list_bm25 = [
        {"doc_id": "D1", "rank": 1, "score": 10.0, "category": "catA", "snippet": "s1", "text": "t1"},
        {"doc_id": "D2", "rank": 2, "score": 8.0, "category": "catA", "snippet": "s2", "text": "t2"},
        {"doc_id": "D3", "rank": 3, "score": 5.0, "category": "catA", "snippet": "s3", "text": "t3"},
    ]

    # Semantic: D2(r=1), D3(r=2), D4(r=3)
    list_sem = [
        {"doc_id": "D2", "rank": 1, "score": 0.9, "category": "catA", "snippet": "s2", "text": "t2"},
        {"doc_id": "D3", "rank": 2, "score": 0.8, "category": "catA", "snippet": "s3", "text": "t3"},
        {"doc_id": "D4", "rank": 3, "score": 0.7, "category": "catB", "snippet": "s4", "text": "t4"},
    ]

    # TF-IDF: D1(r=1), D3(r=2), D4(r=3)
    list_tfidf = [
        {"doc_id": "D1", "rank": 1, "score": 0.85, "category": "catA", "snippet": "s1", "text": "t1"},
        {"doc_id": "D3", "rank": 2, "score": 0.75, "category": "catA", "snippet": "s3", "text": "t3"},
        {"doc_id": "D4", "rank": 3, "score": 0.65, "category": "catB", "snippet": "s4", "text": "t4"},
    ]

    # Expected RRF Scores (k=60):
    # D1: 1/(60+1) + 1/(60+1) = 1/61 + 1/61 = 0.0163934 + 0.0163934 = 0.0327868
    # D2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325224
    # D3: 1/(60+3) + 1/(60+2) + 1/(60+2) = 1/63 + 1/62 + 1/62 = 0.0158730 + 0.0161290 + 0.0161290 = 0.0481310
    # D4: 1/(60+3) + 1/(60+3) = 1/63 + 1/63 = 0.0158730 + 0.0158730 = 0.0317460

    fused = rrf_fusion([list_bm25, list_sem, list_tfidf], k=60, top_k=10)
    top_doc = fused[0]["doc_id"]
    assert top_doc == "D3", f"Expected D3 to win RRF fusion, but got {top_doc}"
    print(f"[+] Mathematical Unit Test Passed! Top fused doc: {top_doc} (Score: {fused[0]['score']})")


def validate_rrf_corpus_search():
    """Empirical verification of search(query, 'hybrid') API."""
    print("\n" + "=" * 60)
    print("RUNNING CORPUS HYBRID SEARCH VALIDATION")
    print("=" * 60)

    test_queries = [
        "NASA space shuttle launch",
        "computer graphics rendering software",
        "motorcycle engine oil change",
        "baseball world series game win",
        "encryption RSA secret key security"
    ]

    for q in test_queries:
        results = search(q, method="hybrid", top_k=5)
        print(f"\nHybrid Query: '{q}' -> Returned {len(results)} results")
        if results:
            top = results[0]
            print(f"   Top-1: {top['doc_id']} | Score: {top['score']} | Category: {top['category']}")

    print("\n[+] RRF Hybrid Retrieval Validation Completed Successfully.")


if __name__ == "__main__":
    run_rrf_math_unit_test()
    validate_rrf_corpus_search()
