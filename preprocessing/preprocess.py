"""
Text Preprocessing Module for the 20 Newsgroups Information Retrieval System.

Pipeline:
1. Header removal (strip_headers)
2. Quoted-reply removal (strip_quotes)
3. Lowercasing
4. Text cleaning / noise reduction (preserving technical terms)
5. Tokenization
6. English stop-word removal
7. Stemming (SnowballStemmer)

Choice of Stemmer:
SnowballStemmer ("english") is selected for its speed, deterministic output, 
and strong empirical performance in IR term normalization over English corpora.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Ensure NLTK data dependencies are loaded quietly
try:
    NLTK_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOPWORDS = set(stopwords.words("english"))

# Initialize deterministic stemmer
STEMMER = SnowballStemmer("english")

# Header line pattern detector
HEADER_KEYS = {
    "from:", "subject:", "organization:", "lines:", "nntp-posting-host:",
    "reply-to:", "followup-to:", "distribution:", "expires:", "article-i.d.:",
    "summary:", "keywords:", "path:", "newsgroups:", "approved:", "sender:"
}


def strip_headers(text: str) -> str:
    """
    Remove email-style newsgroup post headers from raw text.
    Headers precede the message body and are typically separated by double newlines.
    """
    if not text or not isinstance(text, str):
        return ""

    # Check if text begins with standard newsgroup header fields
    lines = text.splitlines()
    if not lines:
        return ""

    first_few_lines = [l.strip().lower() for l in lines[:10] if l.strip()]
    has_header = any(
        any(line.startswith(key) for key in HEADER_KEYS)
        for line in first_few_lines
    )

    if has_header:
        # Split at the first double newline separating header block from body
        parts = re.split(r'\n\s*\n|\r\n\s*\r\n', text, maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()

        # Fallback: line-by-line header stripping if no double newline separator found
        body_lines = []
        in_header = True
        for line in lines:
            if in_header:
                lower_line = line.strip().lower()
                if any(lower_line.startswith(key) for key in HEADER_KEYS):
                    continue
                if lower_line == "":
                    in_header = False
                    continue
                # If line doesn't match header format, header ended
                if not re.match(r'^[A-Za-z0-9\-]+:\s', line):
                    in_header = False
                    body_lines.append(line)
            else:
                body_lines.append(line)
        return "\n".join(body_lines).strip()

    return text.strip()


def strip_quotes(text: str) -> str:
    """
    Remove quoted reply lines starting with '>' or attribution headers (e.g., 'writes:').
    """
    if not text or not isinstance(text, str):
        return ""

    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Filter out quote lines starting with > or |>
        if stripped.startswith(">") or stripped.startswith("|>"):
            continue
        # Filter out common quote attribution lines
        if re.search(r'(writes|wrote|said):\s*$', stripped, re.IGNORECASE):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def clean_and_tokenize(text: str) -> List[str]:
    """
    Convert text to lowercase, clean noise, and extract tokens preserving technical terms.
    
    Preserves:
    - Alphanumeric technical terms (e.g. win31, x11, 3d, vga, 486, c++)
    - Words with internal hyphens or numbers
    """
    if not text:
        return []

    # Lowercase
    text_lower = text.lower()

    # Replace URLs and emails with space
    text_no_urls = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', ' ', text_lower)

    # Extract tokens: alphanumeric words, including tech terms with hyphens/numbers/plus (e.g. c++, x-windows)
    tokens = re.findall(r'\b[a-z0-9]+(?:[\-+._][a-z0-9]+)*\b', text_no_urls)

    return tokens


def preprocess_text(
    text: str, 
    remove_headers: bool = True, 
    remove_quotes: bool = True,
    apply_stemming: bool = True
) -> str:
    """
    Main deterministic preprocessing pipeline.

    Pipeline stages:
    1. Header removal (optional)
    2. Quoted reply removal (optional)
    3. Lowercasing & noise cleaning
    4. Tokenization (preserving technical terms)
    5. English stop-word removal
    6. Stemming (SnowballStemmer)

    Args:
        text: Raw document or query text.
        remove_headers: Whether to strip email-style newsgroup headers.
        remove_quotes: Whether to strip quoted reply lines.
        apply_stemming: Whether to apply Snowball stemming.

    Returns:
        Cleaned, space-joined string of normalized tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    processed = text

    if remove_headers:
        processed = strip_headers(processed)

    if remove_quotes:
        processed = strip_quotes(processed)

    # Tokenize & Clean
    raw_tokens = clean_and_tokenize(processed)

    # Remove English stopwords and short noise tokens (< 2 chars unless numeric/tech like x or 3)
    filtered_tokens = [
        token for token in raw_tokens
        if token not in NLTK_STOPWORDS and (len(token) > 1 or token.isdigit())
    ]

    # Apply Stemming
    if apply_stemming:
        final_tokens = [STEMMER.stem(token) for token in filtered_tokens]
    else:
        final_tokens = filtered_tokens

    return " ".join(final_tokens)


def preprocess_debug(text: str) -> Dict[str, str]:
    """
    Debugging and demonstration function showing step-by-step transformations.
    Useful for live viva demos and academic reports.

    Returns:
        Dictionary with text representations at each stage.
    """
    raw = text if isinstance(text, str) else ""
    no_headers = strip_headers(raw)
    no_quotes = strip_quotes(no_headers)
    final_cleaned = preprocess_text(raw, remove_headers=True, remove_quotes=True, apply_stemming=True)

    return {
        "raw_text": raw,
        "after_header_removal": no_headers,
        "after_quote_removal": no_quotes,
        "final_cleaned": final_cleaned
    }
