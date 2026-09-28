"""
Semantic Retrieval Subsystem Validation Script.
Encodes corpus using Sentence Transformer (all-MiniLM-L6-v2), saves numpy array, and tests queries.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.semantic import SemanticRetriever


def validate_semantic():
    print("=" * 60)
    print("VALIDATING SENTENCE TRANSFORMER SEMANTIC RETRIEVAL")
    print("=" * 60)

    retriever = SemanticRetriever()
    retriever.fit_and_save()

    # Verify reload capability
    reloaded_retriever = SemanticRetriever()
    assert reloaded_retriever.load_index(), "Failed to load embeddings from disk."
    print("[+] Verified embedding artifact reloading from disk.")

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

    print("\n[+] Semantic Retrieval Validation Completed Successfully.")


if __name__ == "__main__":
    validate_semantic()
