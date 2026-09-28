"""
TF-IDF Retrieval Subsystem.

Fits a scikit-learn TfidfVectorizer over preprocessed 20 Newsgroups corpus,
persists sparse matrix artifacts, and performs cosine similarity search.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import DOCUMENTS_CSV, ARTIFACTS_DIR
from preprocessing.preprocess import preprocess_text

TFIDF_DIR = ARTIFACTS_DIR / "tfidf"
VECTORIZER_PATH = TFIDF_DIR / "vectorizer.joblib"
MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.npz"
INDEX_PATH = TFIDF_DIR / "document_index.csv"


class TFIDFRetriever:
    """
    Sparse TF-IDF Retrieval Engine.
    Vectorizes corpus text, stores sparse TF-IDF matrix, and ranks queries using Cosine Similarity.
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.doc_df: Optional[pd.DataFrame] = None
        self.doc_ids: List[str] = []

    def fit_and_save(self, documents_csv_path: Path = DOCUMENTS_CSV) -> None:
        """
        Load corpus documents, apply canonical preprocessing, fit TfidfVectorizer,
        and persist sparse artifacts to disk.
        """
        print("[TF-IDF] Loading document corpus...")
        df = pd.read_csv(documents_csv_path)
        
        print("[TF-IDF] Preprocessing document texts...")
        raw_texts = df["raw_text"].fillna("").astype(str).tolist()
        cleaned_texts = [preprocess_text(t) for t in raw_texts]

        print("[TF-IDF] Fitting TfidfVectorizer on corpus...")
        # Fit vectorizer on cleaned preprocessed text
        self.vectorizer = TfidfVectorizer(token_pattern=r'\S+')
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)

        self.doc_df = df.copy()
        self.doc_ids = df["doc_id"].tolist()

        # Save artifacts
        TFIDF_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        sparse.save_npz(MATRIX_PATH, self.tfidf_matrix)
        df[["doc_id", "category_id", "category_name"]].to_csv(INDEX_PATH, index=False)

        print(f"[TF-IDF] Successfully fit & saved index artifacts to {TFIDF_DIR}:")
        print(f"         - Matrix Shape : {self.tfidf_matrix.shape}")
        print(f"         - Vocabulary Size: {len(self.vectorizer.vocabulary_)}")

    def load_index(self) -> bool:
        """Load pre-built TF-IDF vectorizer, sparse matrix, and document metadata."""
        if not (VECTORIZER_PATH.exists() and MATRIX_PATH.exists() and INDEX_PATH.exists()):
            return False

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.tfidf_matrix = sparse.load_npz(MATRIX_PATH)
        meta_df = pd.read_csv(INDEX_PATH)
        
        # Load full documents metadata to include text
        docs_df = pd.read_csv(DOCUMENTS_CSV)
        self.doc_df = docs_df
        self.doc_ids = docs_df["doc_id"].tolist()
        return True

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Preprocess query, encode with fitted vectorizer, and return top-k cosine similarity results.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            if not self.load_index():
                raise RuntimeError("TF-IDF index not built. Run fit_and_save() first.")

        if not query or not query.strip():
            return []

        # Preprocess query text using canonical pipeline
        cleaned_query = preprocess_text(query)
        if not cleaned_query.strip():
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([cleaned_query])
        if query_vec.nnz == 0:
            return []

        # Calculate cosine similarity against all corpus rows
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Top K indices sorted by descending score with deterministic doc_id tie-breaker
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
