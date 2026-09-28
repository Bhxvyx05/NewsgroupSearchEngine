"""
Streamlit Page 4: Methodology & Theoretical Background.
Provides comprehensive viva-ready explanations of IR concepts, algorithms, and metrics.
"""

import streamlit as st

st.set_page_config(page_title="Methodology - 20 Newsgroups IR", page_icon="📖", layout="wide")

st.title("📖 Theoretical Methodology & IR Concepts")

st.markdown("""
### 1. What is Information Retrieval (IR)?
Information Retrieval is the science of searching for documents, information within documents, and metadata about documents in large unstructured text collections. The core challenge is ranking documents in descending order of predicted relevance to an ad-hoc free-text user query.

---

### 2. Retrieval Algorithms

#### A. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF measures how important a word is to a document in a collection.
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log \\left( \\frac{|D|}{1 + |\{d \\in D : t \\in d\}|} \\right)$$
Documents and queries are represented as high-dimensional sparse vectors, and ranked using **Cosine Similarity**:
$$\\text{CosineSimilarity}(\\vec{q}, \\vec{d}) = \\frac{\\vec{q} \\cdot \\vec{d}}{\\|\\vec{q}\\| \\|\\vec{d}\\|}$$

#### B. BM25 (Best Matching 25)
BM25 is a non-linear probabilistic retrieval function that addresses term frequency saturation and document length normalization:
$$\\text{Score}(D, Q) = \\sum_{i=1}^{n} \\text{IDF}(q_i) \\cdot \\frac{f(q_i, D) \\cdot (k_1 + 1)}{f(q_i, D) + k_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}$$
- $k_1 = 1.5$: Controls term frequency saturation.
- $b = 0.75$: Controls document length normalization penalty.

#### C. Sentence Transformers (Dense Semantic Retrieval)
Dense retrieval embeds text into low-dimensional continuous vector space using pre-trained Transformer architectures (`all-MiniLM-L6-v2`, 384 dimensions). It captures semantic intent, context, and synonymy beyond exact keyword matches.

#### D. Reciprocal Rank Fusion (RRF)
RRF is a state-of-the-art rank-based fusion technique that combines ranked position lists from multiple distinct retrieval engines without comparing raw incompatible score scales:
$$\\text{RRF\\_Score}(d \\in D) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}$$
where $k = 60$ is the standard smoothing constant and $r_m(d)$ is the 1-based rank position of document $d$ in system $m$.

---

### 3. Evaluation Metrics

- **Precision@10**: Proportion of top-10 retrieved documents that are relevant.
  $$\\text{P@10} = \\frac{|\\text{Retrieved}_{10} \\cap \\text{Relevant}|}{10}$$
- **Recall**: Proportion of total relevant documents retrieved across the corpus.
- **Mean Average Precision (MAP)**: Mean of Average Precision (AP) scores calculated across all evaluation queries.
- **nDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality by penalizing relevant documents appearing lower in the top-10 result list:
  $$\\text{DCG@10} = \\sum_{i=1}^{10} \\frac{\\text{rel}_i}{\\log_2(i + 1)}, \\quad \\text{nDCG@10} = \\frac{\\text{DCG@10}}{\\text{IDCG@10}}$$
""")
