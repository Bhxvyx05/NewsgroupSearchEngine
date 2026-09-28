"""
Reciprocal Rank Fusion (RRF) Hybrid Retrieval Module.

Combines distinct ranked lists from TF-IDF, BM25, and Sentence Transformer models
into a unified hybrid ranking using rank-position reciprocal weighting (k=60).
"""

from typing import List, Dict, Any


def rrf_fusion(
    ranked_lists: List[List[Dict[str, Any]]], 
    k: int = 60, 
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion (RRF) on multiple ranked result lists.

    Formula:
        RRF_Score(doc_id) = sum( 1.0 / (k + rank_m(doc_id)) ) for all rankers m.

    Args:
        ranked_lists: List of ranked result lists from individual retrieval models.
        k: RRF smoothing constant (default = 60).
        top_k: Number of fused top-ranked results to return.

    Returns:
        List of fused result dicts sorted by descending RRF score.
    """
    if not ranked_lists:
        return []

    rrf_scores: Dict[str, float] = {}
    doc_metadata: Dict[str, Dict[str, Any]] = {}

    for r_list in ranked_lists:
        for item in r_list:
            doc_id = item["doc_id"]
            rank = item["rank"]  # 1-based rank position

            # Accumulate RRF score contribution
            score_contrib = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contrib

            # Store metadata reference if not already recorded
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = {
                    "category": item.get("category", ""),
                    "snippet": item.get("snippet", ""),
                    "text": item.get("text", "")
                }

    if not rrf_scores:
        return []

    # Sort documents by descending RRF score, with deterministic tie-breaker by doc_id
    sorted_doc_ids = sorted(
        rrf_scores.keys(), 
        key=lambda d_id: (-rrf_scores[d_id], d_id)
    )

    fused_results = []
    for rank_idx, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
        meta = doc_metadata[doc_id]
        fused_results.append({
            "doc_id": doc_id,
            "rank": rank_idx,
            "score": round(rrf_scores[doc_id], 6),
            "category": meta["category"],
            "snippet": meta["snippet"],
            "text": meta["text"]
        })

    return fused_results
