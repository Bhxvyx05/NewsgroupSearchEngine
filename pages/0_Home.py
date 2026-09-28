"""
Streamlit Page 0: Home & Overview.
Displays dynamic corpus KPIs, dataset distribution, and architecture pipeline.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.config import METADATA_CSV, QUERIES_CSV, FIGURES_DIR

st.set_page_config(page_title="Home - 20 Newsgroups IR", page_icon="🏠", layout="wide")

st.title("🏠 20 Newsgroups IR System Dashboard")
st.markdown("##### Comparative Evaluation of TF-IDF, BM25, Sentence Transformer, and Hybrid RRF Models")

# Compute dynamic KPI statistics
doc_count = 18846
cat_count = 20
query_count = 200

if METADATA_CSV.exists():
    df_meta = pd.read_csv(METADATA_CSV)
    doc_count = len(df_meta)
    cat_count = df_meta["category_name"].nunique()

if QUERIES_CSV.exists():
    df_q = pd.read_csv(QUERIES_CSV)
    query_count = len(df_q)

# KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Indexed Documents", f"{doc_count:,}", help="Total raw newsgroup posts in dataset")
col2.metric("Newsgroup Categories", f"{cat_count}", help="Distinct newsgroup topic categories")
col3.metric("Retrieval Models", "4", help="TF-IDF, BM25, Sentence Transformer, Hybrid RRF")
col4.metric("Evaluation Queries", f"{query_count}", help="Topic-based evaluation query suite")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🛠️ System Architecture")
    st.code("""
RAW 20 NEWSGROUPS (18,846 Posts)
               |
               v
PREPROCESSING (Headers/Quotes Removal, Stemming)
               |
    +----------+----------+-----------------------+
    |                     |                       |
    v                     v                       v
 TF-IDF                 BM25             Sentence Transformer
(Cosine)            (Okapi BM25)          (all-MiniLM-L6-v2)
    |                     |                       |
    +----------+----------+-----------------------+
                          |
                          v
               Reciprocal Rank Fusion
                     (RRF k=60)
                          |
                          v
                HYBRID FINAL RANKING
                          |
          +---------------+---------------+
          |                               |
          v                               v
 SEARCH ENGINE UI                 EVALUATION DASHBOARD
                             (P@10, Recall, MAP, nDCG@10)
    """, language="text")

with col_right:
    st.subheader("📊 Corpus Category Distribution")
    cat_dist_img = FIGURES_DIR / "category_distribution.png"
    if cat_dist_img.exists():
        st.image(str(cat_dist_img), use_container_width=True)
    elif METADATA_CSV.exists():
        df_meta = pd.read_csv(METADATA_CSV)
        counts = df_meta["category_name"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        st.bar_chart(counts.set_index("Category"))
    else:
        st.info("Category distribution chart available after data preparation.")
