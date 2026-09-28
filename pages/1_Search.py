"""
Streamlit Page 1: Search Interface.
Interactive search engine calling unified search(query, method, top_k) interface API.
"""

import streamlit as st
import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.retrieval import search

st.set_page_config(page_title="Search Engine - 20 Newsgroups IR", page_icon="🔍", layout="wide")

st.title("🔍 Information Retrieval Search Engine")
st.markdown("Search across 18,846 newsgroup documents using sparse, probabilistic, dense, or hybrid rank fusion models.")

# Preset example queries
st.markdown("💡 **Example Queries (Click to try):**")
example_cols = st.columns(5)

example_query = ""
if example_cols[0].button("🚀 NASA Space Launch"):
    example_query = "NASA space shuttle launch"
if example_cols[1].button("💻 3D Graphics"):
    example_query = "computer graphics rendering software"
if example_cols[2].button("🏍️ Motorcycle Oil"):
    example_query = "motorcycle engine oil change"
if example_cols[3].button("⚾ World Series"):
    example_query = "baseball world series game win"
if example_cols[4].button("🔐 RSA Encryption"):
    example_query = "encryption RSA secret key security"

# Search controls
query_input = st.text_input(
    "Enter Search Query:",
    value=example_query if example_query else "NASA space shuttle launch",
    placeholder="e.g. NASA space shuttle launch mission orbit"
)

col_m, col_k = st.columns([3, 1])
with col_m:
    method_display = st.selectbox(
        "Select Retrieval Algorithm:",
        ["Hybrid (RRF)", "BM25 (Okapi)", "Sentence Transformer (Semantic)", "TF-IDF (Cosine Sim)"],
        index=0
    )
with col_k:
    top_k = st.slider("Top K Results:", min_value=1, max_value=50, value=10)

method_map = {
    "Hybrid (RRF)": "hybrid",
    "BM25 (Okapi)": "bm25",
    "Sentence Transformer (Semantic)": "semantic",
    "TF-IDF (Cosine Sim)": "tfidf"
}

method_key = method_map[method_display]

if st.button("🔍 Search Corpus", type="primary"):
    if not query_input.strip():
        st.warning("⚠️ Please enter a non-empty query string.")
    else:
        with st.spinner(f"Querying corpus via {method_display}..."):
            t0 = time.perf_counter()
            results = search(query_input, method=method_key, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            st.warning("No matching documents found for your query. Try broadening search terms.")
        else:
            st.success(f"Found {len(results)} ranked documents in **{elapsed_ms:.2f} ms** using **{method_display}**.")
            st.divider()

            for res in results:
                rank = res["rank"]
                doc_id = res["doc_id"]
                score = res["score"]
                cat = res["category"]
                snippet = res["snippet"]
                raw_text = res["text"]

                with st.container():
                    c_badge, c_title = st.columns([1, 5])
                    with c_badge:
                        st.markdown(f"### Rank #{rank}")
                        st.caption(f"**Score**: `{score:.6f}`")
                    with c_title:
                        st.markdown(f"#### 📄 Document `{doc_id}` &nbsp; | &nbsp; Category: `:blue[{cat}]`")
                        st.markdown(f"> *\"{snippet}\"*")
                        
                        with st.expander("📖 View Full Document Text"):
                            st.text_area("Original Newsgroup Post", value=raw_text, height=250, key=f"txt_{doc_id}_{rank}")
                st.divider()
