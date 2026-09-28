"""
Index Construction Command Line Utility.

Precomputes and persists reusable index structures for all retrieval algorithms:
- TF-IDF Sparse Matrix & Vectorizer
- BM25 Okapi Index
- Sentence Transformer Dense Vector Embeddings (.npy)
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.tfidf import TFIDFRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever


def build_indexes(method: str = "all"):
    method = method.lower()
    print("=" * 60)
    print(f"BUILDING INDEX ARTIFACTS FOR METHOD: {method.upper()}")
    print("=" * 60)

    if method in ["tfidf", "all"]:
        print("\n--- Building TF-IDF Index ---")
        tfidf = TFIDFRetriever()
        tfidf.fit_and_save()

    if method in ["bm25", "all"]:
        print("\n--- Building BM25 Index ---")
        bm25 = BM25Retriever()
        bm25.fit_and_save()

    if method in ["semantic", "all"]:
        print("\n--- Building Sentence Transformer Semantic Embeddings ---")
        semantic = SemanticRetriever()
        semantic.fit_and_save()

    print("\n[+] Index building completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retrieval indexes for 20 Newsgroups corpus.")
    parser.add_argument(
        "--method", 
        type=str, 
        default="all", 
        choices=["tfidf", "bm25", "semantic", "all"],
        help="Retrieval method index to build ('tfidf', 'bm25', 'semantic', 'all')"
    )
    args = parser.parse_args()
    build_indexes(args.method)
