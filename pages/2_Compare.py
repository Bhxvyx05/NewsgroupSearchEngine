"""
Streamlit Page 2: Multi-Model Comparison View.
Executes a single user query simultaneously across TF-IDF, BM25, Semantic, and Hybrid models
and presents top-5 results side-by-side.
"""

import streamlit as st
import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from retrieval.retrieval import search

st.set_page_config(page_title="Compare Models - 20 Newsgroups IR", page_icon="⚖️", layout="wide")

st.title("⚖️ Side-by-Side Model Comparison View")
st.markdown("Execute a single query across all four retrieval engines to qualitatively compare rank position and score distribution differences.")

query_input = st.text_input(
    "Enter Comparison Query:",
    value="computer graphics 3d rendering software",
    placeholder="e.g. computer graphics 3d rendering software"
)

if st.button("⚖️ Run Multi-Model Comparison", type="primary"):
    if not query_input.strip():
        st.warning("⚠️ Please enter a non-empty query string.")
    else:
        st.markdown("### Top-5 Ranked Results per Model")
        st.caption("Same query, four retrieval strategies")

        methods = [
            ("TF-IDF", "tfidf", "Sparse Cosine Sim"),
            ("BM25", "bm25", "Probabilistic Okapi"),
            ("Sentence Transformer", "semantic", "all-MiniLM-L6-v2"),
            ("Hybrid RRF", "hybrid", "Reciprocal Rank Fusion k=60")
        ]

        cols = st.columns(4)

        for idx, (label, key, subtitle) in enumerate(methods):
            with cols[idx]:
                st.subheader(label)
                st.caption(subtitle)

                t0 = time.perf_counter()
                results = search(query_input, method=key, top_k=5)
                ms = (time.perf_counter() - t0) * 1000.0

                st.caption(f"⚡ Latency: `{ms:.1f} ms`")
                st.divider()

                if not results:
                    st.write("*No results returned*")
                else:
                    for res in results:
                        st.markdown(f"**Rank #{res['rank']}** &nbsp; | &nbsp; `{res['doc_id']}`")
                        st.markdown(f"**Category**: `:blue[{res['category']}]`")
                        st.markdown(f"**Score**: `{res['score']:.4f}`")
                        st.caption(f"*\"{res['snippet'][:120]}...\"*")
                        st.divider()
