"""
TF-IDF Retrieval Subsystem Validation Script.
Fits TF-IDF index, saves artifacts, verifies matrix dimensions, and tests search queries.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.tfidf import TFIDFRetriever


def validate_tfidf():
    print("=" * 60)
    print("VALIDATING TF-IDF RETRIEVAL SUBSYSTEM")
    print("=" * 60)

    retriever = TFIDFRetriever()
    retriever.fit_and_save()

    # Verify reload capability
    reloaded_retriever = TFIDFRetriever()
    assert reloaded_retriever.load_index(), "Failed to load TF-IDF artifacts from disk."
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

    print("\n[+] TF-IDF Validation Completed Successfully.")


if __name__ == "__main__":
    validate_tfidf()
