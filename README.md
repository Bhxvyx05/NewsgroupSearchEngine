# 20 Newsgroups Information Retrieval System

> **Comparative Evaluation of TF-IDF, BM25, Sentence Transformer and Hybrid RRF Retrieval Models on the 20 Newsgroups Corpus with a Streamlit Search Interface**

---

## 1. Project Overview
This project implements a complete Information Retrieval (IR) system over the official **20 Newsgroups** corpus (`18,846` posts across 20 distinct categories). The system integrates and evaluates four distinct document ranking strategies:
1. **TF-IDF**: Sparse vector space model using Cosine Similarity.
2. **BM25**: Probabilistic keyword retrieval using Okapi BM25 (`rank_bm25`).
3. **Sentence Transformer**: Dense neural semantic search using `all-MiniLM-L6-v2`.
4. **Hybrid RRF**: Reciprocal Rank Fusion ($k=60$) fusing lexical and neural rankings.

---

## 2. Problem Statement
Traditional lexical IR systems (TF-IDF, BM25) rely on exact keyword matches and fail when queries use synonyms or paraphrased terms. Conversely, dense semantic models capture conceptual intent but can miss exact technical terms or acronyms. This system compares all four paradigms and demonstrates how **Reciprocal Rank Fusion (RRF)** combines keyword precision with neural recall.

---

## 3. Core System Architecture

```text
RAW 20 NEWSGROUPS DATASET (18,846 posts)
                  |
                  v
         DATASET PREPROCESSING (Header/Quote Removal, Tokenization, Stemming)
                  |
       +----------+----------+-----------------------+
       |                     |                       |
       v                     v                       v
    TF-IDF                 BM25             Sentence Transformer
(Cosine Similarity)     (Okapi BM25)         (all-MiniLM-L6-v2)
       |                     |                       |
       +----------+----------+-----------------------+
                             |
                             v
                  RECIPROCAL RANK FUSION (RRF k=60)
                             |
                             v
                    HYBRID FINAL RANKING
                             |
             +---------------+---------------+
             |                               |
             v                               v
    STREAMLIT SEARCH UI             EVALUATION ENGINE
                               (P@10, Recall, MAP, nDCG@10)
```

---

## 4. Text Preprocessing Pipeline
To normalize terms, strip non-topical metadata, and reduce index vocabulary noise, a deterministic preprocessing engine is implemented in `preprocessing/preprocess.py`:

1. **Header Removal (`strip_headers`)**: Strips newsgroup post email headers (`From:`, `Subject:`, `Organization:`, `Lines:`, etc.).
2. **Quote Stripping (`strip_quotes`)**: Filters out quoted reply text lines (starting with `>` or `|>`) and attribution headers (`writes:`).
3. **Lowercasing & Normalization**: Converts text to lowercase while explicitly preserving technical terms (e.g. `win3.1`, `x11`, `3d`, `vga`, `c++`, `486`).
4. **Tokenization**: Regex-based token extraction.
5. **English Stop-word Filtering**: Removes NLTK standard English stopwords.
6. **Stemming**: Applies NLTK `SnowballStemmer("english")` for deterministic term canonicalization.

---

## 5. Quantitative Evaluation Results

The evaluation benchmark was executed across **200 curated topic-based queries** mapped to ground-truth category relevance judgments (`data/qrels.csv` with `188,460` relevance pairs).

| Retrieval Method | Precision@10 | Recall@100 | MAP | nDCG@10 | Avg Query Latency |
|---|---|---|---|---|---|
| **TF-IDF** | `0.6400` | `0.0540` | `0.0380` | `0.6531` | `41.72 ms` |
| **BM25** | `0.7295` | `0.0587` | `0.0449` | `0.7365` | `54.44 ms` |
| **Sentence Transformer** | **`0.7515`** | **`0.0683`** | **`0.0549`** | **`0.7556`** | `52.16 ms` |
| **Hybrid RRF** | `0.7425` | `0.0624` | `0.0487` | `0.7514` | `145.72 ms` |

### Key Observations:
- **Sentence Transformer (`all-MiniLM-L6-v2`)** achieved the highest overall Precision@10 (`0.7515`) and nDCG@10 (`0.7556`) due to its ability to bridge vocabulary mismatch in newsgroup queries.
- **BM25** significantly outperformed TF-IDF (`0.7295` vs `0.6400` P@10) due to non-linear term frequency saturation and document length normalization.
- **Hybrid RRF** ($k=60$) effectively balanced keyword precision and dense semantic recall, achieving a strong `0.7425` P@10 and `0.7514` nDCG@10 without requiring raw score calibration.

---

## 6. Directory Structure
```text
20-newsgroups-information-retrieval/
├── app.py                      # Streamlit application main launcher
├── requirements.txt            # Project Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── core/
│   └── config.py               # Centralized configuration & constants
├── data/
│   ├── documents.csv           # Loaded corpus documents (18,846 posts)
│   ├── metadata.csv            # Document metadata mapping
│   ├── queries.csv             # 200 topic-based evaluation queries
│   └── qrels.csv               # Ground-truth relevance judgments (188,460 pairs)
├── preprocessing/
│   └── preprocess.py           # Text cleaning & normalization pipeline
├── retrieval/
│   ├── tfidf.py                # Sparse TF-IDF retriever
│   ├── bm25.py                 # BM25Okapi retriever
│   ├── semantic.py             # Sentence Transformer semantic retriever
│   ├── rrf.py                  # Reciprocal Rank Fusion algorithm
│   └── retrieval.py            # Common search interface API
├── embeddings/
│   └── document_embeddings.npy # Saved 384-dim dense embeddings (18,846 x 384)
├── evaluation/
│   ├── metrics.py              # P@K, Recall, MAP, nDCG@K implementations
│   ├── evaluate.py             # Benchmark test suite runner
│   ├── results.csv             # Measured summary benchmark metrics output
│   ├── per_query_results.csv   # Per-query evaluation breakdown
│   └── performance.csv         # Measured query latencies
├── visualization/
│   ├── precision_recall.py     # PR curve plotter
│   └── comparison.py           # Metric bar chart comparison plotter
├── scripts/
│   ├── prepare_data.py         # Dataset fetcher & metadata builder
│   ├── build_indexes.py        # Offline index builder
│   ├── build_qrels.py          # Ground-truth qrels generator
│   ├── validate_preprocessing.py # Preprocessing validator
│   ├── validate_tfidf.py       # TF-IDF validator
│   ├── validate_bm25.py        # BM25 validator
│   ├── validate_semantic.py    # Semantic validator
│   ├── validate_rrf.py         # RRF validator
│   └── validate_project.py     # End-to-end master regression tester
├── pages/                      # Multi-page Streamlit views
│   ├── 0_Home.py
│   ├── 1_Search.py
│   ├── 2_Compare.py
│   ├── 3_Evaluation.py
│   └── 4_About.py
├── report/figures/             # Generated academic plots & charts
│   ├── category_distribution.png
│   ├── metric_comparison.png
│   └── precision_recall.png
└── notebooks/
    ├── dataset_analysis.ipynb
    └── preprocessing_analysis.ipynb
```

---

## 7. Installation & Reproducibility

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   ```bash
   python scripts/prepare_data.py
   ```

3. **Build Precomputed Indexes & Embeddings**:
   ```bash
   python scripts/build_indexes.py --method all
   ```

4. **Generate Queries & Qrels**:
   ```bash
   python scripts/build_qrels.py
   ```

5. **Run Evaluation Benchmark**:
   ```bash
   python evaluation/evaluate.py
   ```

6. **Generate Report Figures**:
   ```bash
   python visualization/comparison.py
   python visualization/precision_recall.py
   ```

7. **Run Master Regression Test Suite**:
   ```bash
   python scripts/validate_project.py
   ```

8. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

---

## 8. Live Demonstration Workflow
1. Open Streamlit Dashboard (`streamlit run app.py`).
2. Explore **Home** overview and category distribution.
3. Open **Search Engine**, select retrieval algorithm (TF-IDF, BM25, Semantic, or Hybrid), and test queries.
4. Open **Compare Models** side-by-side view to inspect top-5 document rankings across all 4 models simultaneously.
5. Open **Evaluation Dashboard** to view live metric cards, comparison bar chart, and PR-Curves.
