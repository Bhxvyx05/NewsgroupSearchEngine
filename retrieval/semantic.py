"""
Sentence Transformer Dense Semantic Retrieval Subsystem.

Encodes documents using sentence-transformers (all-MiniLM-L6-v2),
normalizes vectors for Cosine Similarity search, and persists precomputed dense embeddings to disk.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import DOCUMENTS_CSV, EMBEDDINGS_DIR, EMBEDDING_MODEL
from preprocessing.preprocess import preprocess_text

EMBEDDINGS_NPY_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "document_index.csv"


class SemanticRetriever:
    """
    Dense Semantic Retrieval Engine powered by SentenceTransformers.
    Computes/loads 384-dimensional dense embeddings for the 20 Newsgroups corpus.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.doc_df: Optional[pd.DataFrame] = None
        self.doc_ids: List[str] = []

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"[Semantic] Loading SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit_and_save(self, documents_csv_path: Path = DOCUMENTS_CSV) -> None:
        """
        Encode corpus text into normalized dense vectors and save .npy file.
        """
        print("[Semantic] Loading document corpus...")
        df = pd.read_csv(documents_csv_path)

        # For dense semantic search, preprocess text by stripping headers/quotes but keeping unstemmed natural words
        print("[Semantic] Preparing document texts for dense neural encoding...")
        raw_texts = df["raw_text"].fillna("").astype(str).tolist()
        texts_to_encode = [preprocess_text(t, apply_stemming=False) for t in raw_texts]

        # Handle empty strings gracefully
        texts_to_encode = [t if t.strip() else "empty document" for t in texts_to_encode]

        model = self._get_model()
        print(f"[Semantic] Encoding {len(texts_to_encode)} documents into dense vectors (this may take a few moments)...")
        embeddings = model.encode(
            texts_to_encode,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity via dot product
        )

        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.doc_df = df.copy()
        self.doc_ids = df["doc_id"].tolist()

        # Save artifacts
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_NPY_PATH, self.embeddings)
        df[["doc_id", "category_id", "category_name"]].to_csv(METADATA_PATH, index=False)

        print(f"[Semantic] Successfully saved embeddings to {EMBEDDINGS_NPY_PATH}:")
        print(f"           - Embedding Shape: {self.embeddings.shape}")

    def load_index(self) -> bool:
        """Load precomputed document embeddings and metadata mapping."""
        if not (EMBEDDINGS_NPY_PATH.exists() and METADATA_PATH.exists()):
            return False

        self.embeddings = np.load(EMBEDDINGS_NPY_PATH)
        docs_df = pd.read_csv(DOCUMENTS_CSV)
        self.doc_df = docs_df
        self.doc_ids = docs_df["doc_id"].tolist()
        return True

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Encode query and compute cosine similarity against precomputed document vectors.
        """
        if self.embeddings is None or self.doc_df is None:
            if not self.load_index():
                raise RuntimeError("Dense embeddings index not found. Run fit_and_save() first.")

        if not query or not query.strip():
            return []

        cleaned_query = preprocess_text(query, apply_stemming=False)
        if not cleaned_query.strip():
            cleaned_query = query

        model = self._get_model()
        query_vec = model.encode([cleaned_query], normalize_embeddings=True)[0]

        # Calculate cosine similarity (dot product of L2 normalized vectors)
        scores = np.dot(self.embeddings, query_vec)

        if len(scores) == 0:
            return []

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank_idx, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
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
