"""
Unified Common Retrieval API Contract.

Exposes a clean single entry point search(query, method, top_k) for:
- 'tfidf'
- 'bm25'
- 'semantic'
- 'hybrid'
"""

from typing import List, Dict, Any, Literal
from retrieval.tfidf import TFIDFRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.rrf import rrf_fusion
from core.config import RRF_K

# Global singleton retriever instances for efficient memory usage
_TFIDF_INSTANCE: TFIDFRetriever = None
_BM25_INSTANCE: BM25Retriever = None
_SEMANTIC_INSTANCE: SemanticRetriever = None


def _get_tfidf() -> TFIDFRetriever:
    global _TFIDF_INSTANCE
    if _TFIDF_INSTANCE is None:
        _TFIDF_INSTANCE = TFIDFRetriever()
        _TFIDF_INSTANCE.load_index()
    return _TFIDF_INSTANCE


def _get_bm25() -> BM25Retriever:
    global _BM25_INSTANCE
    if _BM25_INSTANCE is None:
        _BM25_INSTANCE = BM25Retriever()
        _BM25_INSTANCE.load_index()
    return _BM25_INSTANCE


def _get_semantic() -> SemanticRetriever:
    global _SEMANTIC_INSTANCE
    if _SEMANTIC_INSTANCE is None:
        _SEMANTIC_INSTANCE = SemanticRetriever()
        _SEMANTIC_INSTANCE.load_index()
    return _SEMANTIC_INSTANCE


def search(
    query: str, 
    method: Literal["tfidf", "bm25", "semantic", "hybrid"] = "hybrid", 
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Common Retrieval Interface Contract.

    Args:
        query: User search query string.
        method: One of 'tfidf', 'bm25', 'semantic', or 'hybrid'.
        top_k: Number of top documents to return.

    Returns:
        List of standardized document dictionaries:
        [
            {
                "doc_id": "doc_000001",
                "rank": 1,
                "score": float,
                "category": "sci.space",
                "snippet": "...",
                "text": "..."
            }, ...
        ]
    """
    valid_methods = ["tfidf", "bm25", "semantic", "hybrid"]
    method_lower = method.lower()
    if method_lower not in valid_methods:
        raise ValueError(f"Invalid retrieval method '{method}'. Must be one of {valid_methods}.")

    if not query or not query.strip():
        return []

    if method_lower == "tfidf":
        return _get_tfidf().search(query, top_k=top_k)

    elif method_lower == "bm25":
        return _get_bm25().search(query, top_k=top_k)

    elif method_lower == "semantic":
        return _get_semantic().search(query, top_k=top_k)

    elif method_lower == "hybrid":
        # Retrieve candidate top-50 from each base model to perform RRF rank fusion
        candidate_depth = max(top_k * 3, 50)
        tfidf_res = _get_tfidf().search(query, top_k=candidate_depth)
        bm25_res = _get_bm25().search(query, top_k=candidate_depth)
        semantic_res = _get_semantic().search(query, top_k=candidate_depth)

        ranked_lists = [tfidf_res, bm25_res, semantic_res]
        return rrf_fusion(ranked_lists, k=RRF_K, top_k=top_k)

    return []
