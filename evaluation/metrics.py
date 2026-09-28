"""
Information Retrieval Evaluation Metrics Engine.

Implements exact standard formulas for:
- Precision@K
- Recall@K
- Average Precision (AP) -> MAP
- Normalized Discounted Cumulative Gain at K (nDCG@K)
"""

import numpy as np
from typing import List, Set


def precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10) -> float:
    """
    Calculate Precision@K.
    Fraction of retrieved documents in top-K that are relevant.
    """
    if not retrieved_doc_ids or k <= 0:
        return 0.0
    cutoff = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in cutoff if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 100) -> float:
    """
    Calculate Recall@K.
    Fraction of total relevant documents retrieved within top-K.
    """
    if not relevant_doc_ids:
        return 0.0
    cutoff = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in cutoff if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def average_precision(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], cutoff_k: int = 100) -> float:
    """
    Calculate Average Precision (AP) for a single query.
    """
    if not relevant_doc_ids or not retrieved_doc_ids:
        return 0.0

    score = 0.0
    hits = 0
    cutoff = retrieved_doc_ids[:cutoff_k]

    for i, doc_id in enumerate(cutoff, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            score += hits / i

    return score / len(relevant_doc_ids)


def ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10) -> float:
    """
    Calculate nDCG@K for binary relevance.
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = ideal DCG@K when all top-min(K, |relevant|) slots are relevant.
    """
    if not retrieved_doc_ids or not relevant_doc_ids or k <= 0:
        return 0.0

    cutoff = retrieved_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(cutoff, start=1):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / np.log2(i + 1)

    # Ideal DCG
    ideal_hits = min(k, len(relevant_doc_ids))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
