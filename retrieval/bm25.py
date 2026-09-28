"""
BM25 Retrieval Subsystem.

Implements Okapi BM25 ranking algorithm over preprocessed corpus tokens using rank_bm25 library,
persists index artifacts, and performs keyword query scoring.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import DOCUMENTS_CSV, ARTIFACTS_DIR
from preprocessing.preprocess import preprocess_text

BM25_DIR = ARTIFACTS_DIR / "bm25"
INDEX_PATH = BM25_DIR / "bm25_index.joblib"
METADATA_PATH = BM25_DIR / "document_index.csv"


class BM25Retriever:
    """
    Okapi BM25 Retrieval Engine.
    Builds BM25 index over preprocessed corpus tokens and scores user queries.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.doc_df: Optional[pd.DataFrame] = None
        self.doc_ids: List[str] = []

    def fit_and_save(self, documents_csv_path: Path = DOCUMENTS_CSV) -> None:
        """
        Tokenize preprocessed documents, fit BM25Okapi index, and persist to disk.
        """
        print("[BM25] Loading document corpus...")
        df = pd.read_csv(documents_csv_path)

        print("[BM25] Tokenizing preprocessed document corpus...")
        raw_texts = df["raw_text"].fillna("").astype(str).tolist()
        tokenized_corpus = [preprocess_text(t).split() for t in raw_texts]

        print("[BM25] Fitting BM25Okapi index on corpus...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.doc_df = df.copy()
        self.doc_ids = df["doc_id"].tolist()

        # Save artifacts
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bm25, INDEX_PATH)
        df[["doc_id", "category_id", "category_name"]].to_csv(METADATA_PATH, index=False)

        print(f"[BM25] Successfully built & saved BM25 index artifacts to {BM25_DIR}")

    def load_index(self) -> bool:
        """Load pre-built BM25Okapi index and document metadata."""
        if not (INDEX_PATH.exists() and METADATA_PATH.exists()):
            return False

        self.bm25 = joblib.load(INDEX_PATH)
        docs_df = pd.read_csv(DOCUMENTS_CSV)
        self.doc_df = docs_df
        self.doc_ids = docs_df["doc_id"].tolist()
        return True

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Tokenize query and score against BM25 index.
        """
        if self.bm25 is None or self.doc_df is None:
            if not self.load_index():
                raise RuntimeError("BM25 index not built. Run fit_and_save() first.")

        if not query or not query.strip():
            return []

        # Preprocess and tokenize query
        query_tokens = preprocess_text(query).split()
        if not query_tokens:
            return []

        # Compute BM25 scores across all corpus documents
        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0:
            return []

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank_idx, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            row = self.doc_df.iloc[idx]
            raw_text = str(row["raw_text"])
            snippet = raw_text[:250].replace("\n", " ").strip() + "..."
            results.append({
                "doc_id": row["doc_id"],
                "rank": rank_idx,
                "score": round(score, 6),
                "category": row["category_name"],
                "snippet": snippet,
                "text": raw_text
            })

        return results
