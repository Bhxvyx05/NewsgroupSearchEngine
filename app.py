"""
Streamlit Web Application Main Launcher.

20 Newsgroups Information Retrieval System
Comparative Evaluation of TF-IDF, BM25, Sentence Transformer, and Hybrid RRF Models.
"""

import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

st.set_page_config(
    page_title="20 Newsgroups IR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("🔍 20 Newsgroups Information Retrieval System")
    st.markdown("""
    Welcome to the **Academic Information Retrieval & Hybrid Search Architecture** built over the official **20 Newsgroups Corpus**.
    
    ### 📌 Navigation Guide
    Use the left sidebar to navigate across application modules:
    
    - **🏠 Home**: System overview, corpus statistics, and end-to-end architecture pipeline.
    - **🔍 Search**: Interactive search engine supporting **TF-IDF**, **BM25**, **Sentence Transformer**, and **Hybrid RRF**.
    - **⚖️ Compare**: Side-by-side multi-model comparison view for single queries.
    - **📊 Evaluation Dashboard**: Quantitative benchmark metrics (**Precision@10**, **Recall**, **MAP**, **nDCG@10**) & **PR-Curves**.
    - **📖 Methodology**: Theoretical foundations, formula breakdowns, and IR concepts.
    """)

    st.divider()

    st.info("👈 Select a page from the sidebar on the left to begin exploring!")


if __name__ == "__main__":
    main()
