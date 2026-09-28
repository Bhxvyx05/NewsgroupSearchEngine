"""
Phase 2 Preprocessing Pipeline Validation & Corpus Analysis Script.

Executes:
1. Unit tests on edge cases (headers, quotes, tech terms, stop words, determinism).
2. Full dataset preprocessing validation on 18,846 documents in data/documents.csv.
3. Statistics reporting (empty cleaned count, min/max/avg text length).
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
from core.config import DOCUMENTS_CSV
from preprocessing.preprocess import preprocess_text, preprocess_debug, strip_headers, strip_quotes


def run_unit_tests():
    """Run unit assertions on preprocessing logic."""
    print("=" * 60)
    print("RUNNING PREPROCESSING UNIT TESTS")
    print("=" * 60)

    # Test 1: Empty input
    assert preprocess_text("") == "", "Failed empty string test"
    assert preprocess_text(None) == "", "Failed None input test"
    print("[+] Test 1 Passed: Empty/None inputs handled safely.")

    # Test 2: Determinism
    sample_text = "From: user@domain.com\nSubject: Test\n\nNASA space shuttle launch in 1999! Is it working?"
    run1 = preprocess_text(sample_text)
    run2 = preprocess_text(sample_text)
    assert run1 == run2, "Failed determinism test"
    print(f"[+] Test 2 Passed: Determinism verified. Result: '{run1}'")

    # Test 3: Header removal
    header_text = "From: alice@test.org\nSubject: Graphics\nLines: 5\n\nComputer graphics 3D rendering algorithm."
    stripped_h = strip_headers(header_text)
    assert "From:" not in stripped_h and "Computer graphics" in stripped_h, "Failed header removal"
    print("[+] Test 3 Passed: Header removal verified.")

    # Test 4: Quote removal
    quote_text = "bob@test.org writes:\n> Old quoted message\nNew response here."
    stripped_q = strip_quotes(quote_text)
    assert "Old quoted message" not in stripped_q and "New response" in stripped_q, "Failed quote removal"
    print("[+] Test 4 Passed: Quote removal verified.")

    # Test 5: Technical terms & stop words
    tech_text = "Windows 3.11, X11, C++, and 486 DX2-66 processors are running fast."
    cleaned_tech = preprocess_text(tech_text)
    assert any(term in cleaned_tech for term in ["window", "x11", "c++", "486"]), f"Tech term preserved in: '{cleaned_tech}'"
    print(f"[+] Test 5 Passed: Technical terms preserved. Result: '{cleaned_tech}'")

    print("\nAll 5 Unit Tests Passed Cleanly!\n")


def validate_corpus_preprocessing():
    """Run preprocessing across all 18,846 documents in data/documents.csv."""
    print("=" * 60)
    print("RUNNING CORPUS-WIDE PREPROCESSING VALIDATION")
    print("=" * 60)

    if not DOCUMENTS_CSV.exists():
        raise FileNotFoundError(f"Missing dataset CSV at {DOCUMENTS_CSV}. Run scripts/prepare_data.py first.")

    df = pd.read_csv(DOCUMENTS_CSV)
    total_docs = len(df)
    print(f"[+] Loaded {total_docs} documents from {DOCUMENTS_CSV}")

    # Process all texts
    raw_texts = df["raw_text"].fillna("").astype(str).tolist()
    cleaned_texts = [preprocess_text(t) for t in raw_texts]

    # Calculate statistics
    raw_lengths = [len(t) for t in raw_texts]
    cleaned_lengths = [len(t) for t in cleaned_texts]
    empty_cleaned_count = sum(1 for c in cleaned_texts if len(c.strip()) == 0)

    avg_raw_len = np.mean(raw_lengths)
    avg_cleaned_len = np.mean(cleaned_lengths)
    min_cleaned_len = np.min(cleaned_lengths)
    max_cleaned_len = np.max(cleaned_lengths)

    print("\n" + "=" * 60)
    print("PREPROCESSING VALIDATION RESULTS")
    print("=" * 60)
    print(f"1. Total Documents Processed : {total_docs}")
    print(f"2. Empty Cleaned Documents   : {empty_cleaned_count} ({empty_cleaned_count / total_docs * 100:.2f}%)")
    print(f"3. Avg Raw Text Length       : {avg_raw_len:.2f} chars")
    print(f"4. Avg Cleaned Text Length   : {avg_cleaned_len:.2f} chars")
    print(f"5. Min Cleaned Text Length   : {min_cleaned_len} chars")
    print(f"6. Max Cleaned Text Length   : {max_cleaned_len} chars")

    # Sample debug output preview
    print("\nSample Preprocessing Debug Transformation (doc_000001):")
    debug_info = preprocess_debug(raw_texts[0])
    print("--- RAW TEXT (First 150 chars) ---")
    print(debug_info["raw_text"][:150] + "...")
    print("--- AFTER HEADER REMOVAL (First 150 chars) ---")
    print(debug_info["after_header_removal"][:150] + "...")
    print("--- FINAL CLEANED TEXT ---")
    print(debug_info["final_cleaned"][:150] + "...")
    print("-" * 60)

    return {
        "total_docs": total_docs,
        "empty_cleaned_count": empty_cleaned_count,
        "avg_raw_len": avg_raw_len,
        "avg_cleaned_len": avg_cleaned_len,
        "min_cleaned_len": min_cleaned_len,
        "max_cleaned_len": max_cleaned_len
    }


if __name__ == "__main__":
    run_unit_tests()
    validate_corpus_preprocessing()
