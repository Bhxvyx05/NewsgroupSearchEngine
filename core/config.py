"""
Centralized Configuration Module for the 20 Newsgroups Information Retrieval System.
Defines dataset names, directory paths, algorithm constants, and random seeds.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EVALUATION_DIR = BASE_DIR / "evaluation"
FIGURES_DIR = BASE_DIR / "report" / "figures"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
for path in [DATA_DIR, EMBEDDINGS_DIR, EVALUATION_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Data Files
DOCUMENTS_CSV = DATA_DIR / "documents.csv"
METADATA_CSV = DATA_DIR / "metadata.csv"
QUERIES_CSV = DATA_DIR / "queries.csv"
QRELS_CSV = DATA_DIR / "qrels.csv"

# Model Artifact Paths
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.pkl"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
EMBEDDINGS_NPY_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
EVAL_RESULTS_CSV = EVALUATION_DIR / "results.csv"

# System Constants
DATASET_NAME = "20newsgroups"
RANDOM_SEED = 42
DEFAULT_TOP_K = 10
RRF_K = 60
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
