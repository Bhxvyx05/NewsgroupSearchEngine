"""
Dataset Preparation & Deterministic Loading Script.

Fetches the complete official 20 Newsgroups corpus from scikit-learn,
generates deterministic document IDs (doc_000001 ... doc_N),
saves raw document texts and metadata to CSV files,
and performs exploratory analysis on category distribution and text length.
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from core.config import DATASET_NAME, DOCUMENTS_CSV, METADATA_CSV, DATA_DIR


def prepare_dataset():
    """Load scikit-learn 20 newsgroups dataset, save CSVs, and compute stats."""
    print("=" * 60)
    print("PHASE 1: Loading 20 Newsgroups Dataset...")
    print("=" * 60)

    # Deterministic loading configuration
    dataset = fetch_20newsgroups(
        subset="all",
        remove=(),  # Keep raw text (headers, quotes, footers) for dedicated preprocessing
        shuffle=False
    )

    raw_texts = dataset.data
    target_ids = dataset.target
    category_names = dataset.target_names

    num_docs = len(raw_texts)
    print(f"[+] Loaded {num_docs} raw documents across {len(category_names)} categories.")

    # Generate deterministic document IDs: doc_000001, doc_000002, ...
    doc_ids = [f"doc_{i+1:06d}" for i in range(num_docs)]

    # Build document DataFrame
    df_docs = pd.DataFrame({
        "doc_id": doc_ids,
        "category_id": target_ids,
        "category_name": [category_names[cat_id] for cat_id in target_ids],
        "raw_text": raw_texts
    })

    # Build metadata DataFrame
    df_meta = df_docs[["doc_id", "category_id", "category_name"]].copy()

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV outputs
    df_docs.to_csv(DOCUMENTS_CSV, index=False, encoding="utf-8")
    df_meta.to_csv(METADATA_CSV, index=False, encoding="utf-8")
    print(f"[+] Saved documents dataset to: {DOCUMENTS_CSV}")
    print(f"[+] Saved metadata dataset to: {METADATA_CSV}")

    # Integrity Assertions
    assert len(df_docs) == num_docs, "Document count mismatch!"
    assert df_docs["doc_id"].nunique() == num_docs, "Document IDs must be strictly unique!"
    assert df_docs["category_id"].nunique() == len(category_names), "Category count mismatch!"
    assert not df_docs["raw_text"].isna().any(), "Raw text contains NaN values!"
    print("[+] Integrity checks passed successfully (Unique IDs, non-null texts, valid categories).")

    # Exploratory Analysis & Statistics
    text_lengths = df_docs["raw_text"].apply(len)
    
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"1. Total Document Count : {num_docs}")
    print(f"2. Total Category Count : {len(category_names)}")
    print(f"3. Text Length Stats    : Min = {text_lengths.min()} chars | Max = {text_lengths.max()} chars | Avg = {text_lengths.mean():.2f} chars | Median = {text_lengths.median():.2f} chars")

    print("\n4. Category Distribution:")
    category_counts = df_docs["category_name"].value_counts()
    for cat_name, count in category_counts.items():
        print(f"   - {cat_name:<25}: {count} docs")

    print("\n5. Sample Document Preview (doc_000001):")
    print("-" * 50)
    sample_doc = df_docs.iloc[0]
    print(f"ID       : {sample_doc['doc_id']}")
    print(f"Category : {sample_doc['category_name']} (ID: {sample_doc['category_id']})")
    print("Raw Text Snippet (First 300 chars):")
    print(sample_doc['raw_text'][:300] + "...")
    print("-" * 50)

    print("\n[+] Phase 1 dataset preparation completed successfully.")
    return df_docs, df_meta


if __name__ == "__main__":
    prepare_dataset()
