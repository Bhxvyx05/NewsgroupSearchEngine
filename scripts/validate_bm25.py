"""
BM25 Retrieval Subsystem Validation Script.
Fits BM25 index, saves artifacts, and tests sample queries.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.bm25 import BM25Retriever


def validate_bm25():
    print("=" * 60)
    print("VALIDATING BM25 RETRIEVAL SUBSYSTEM")
    print("=" * 60)

    retriever = BM25Retriever()
    retriever.fit_and_save()

    # Verify reload capability
    reloaded_retriever = BM25Retriever()
    assert reloaded_retriever.load_index(), "Failed to load BM25 artifacts from disk."
    print("[+] Verified artifact reloading from disk.")

    # Test Search Queries
    test_queries = [
        "NASA space shuttle launch",
        "computer graphics rendering software",
        "motorcycle engine oil change",
        "baseball world series game win",
        "encryption RSA secret key security"
    ]

    for q in test_queries:
        results = reloaded_retriever.search(q, top_k=5)
        print(f"\nQuery: '{q}' -> Returned {len(results)} results")
        if results:
            top_res = results[0]
            print(f"   Top-1: {top_res['doc_id']} | Score: {top_res['score']} | Category: {top_res['category']}")

    print("\n[+] BM25 Validation Completed Successfully.")


if __name__ == "__main__":
    validate_bm25()
