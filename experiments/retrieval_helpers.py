"""
experiments/retrieval_helpers.py
=================================
Utilitas bersama yang digunakan oleh semua test case retrieval.

Ekspor utama:
    QUERY_DATASET   — 14 kueri representatif dari berbagai kategori
    compute_scores  — menghitung semua skor retrieval untuk satu pasang (query, chunk)
    raw_query       — melakukan query ChromaDB langsung (tanpa filter accept/reject)
"""

import re
from difflib import SequenceMatcher

from src.vectordb import LessonVectorDB


# ==================================================================
# QUERY DATASET
# ==================================================================

QUERY_DATASET = [
    # Relevant concepts
    {"id": "Q01", "category": "Relevant",    "query": "Apa kegunaan novalidate dalam form HTML?"},
    {"id": "Q02", "category": "Relevant",    "query": "Apa kegunaan tag <audio>?"},
    {"id": "Q03", "category": "Relevant",    "query": "Apa kegunaan atribut value pada input?"},
    {"id": "Q04", "category": "Relevant",    "query": "Apa fungsi child selector dalam CSS?"},
    {"id": "Q05", "category": "Relevant",    "query": "Apa kegunaan atribut action pada form?"},
    {"id": "Q06", "category": "Relevant",    "query": "Apa fungsi addEventListener?"},
    {"id": "Q07", "category": "Relevant",    "query": "Apa itu CSS?"},
    {"id": "Q08", "category": "Relevant",    "query": "Apa kegunaan tag <a>?"},
    {"id": "Q09", "category": "Relevant",    "query": "Apa fungsi href?"},
    # Typo queries
    {"id": "Q10", "category": "Typo",        "query": "Apa kegunaan novalidat dalam form HTML?"},
    {"id": "Q11", "category": "Typo",        "query": "Apa kegunaan addEventListner?"},
    {"id": "Q12", "category": "Typo",        "query": "Apa kegunaan valur pada input?"},
    # Unsupported
    {"id": "Q13", "category": "Unsupported", "query": "Apa itu Python?"},
    {"id": "Q14", "category": "Unsupported", "query": "Bagaimana cara membuat REST API?"},
]


# ==================================================================
# INTERNAL HELPERS (mirrors LessonRetriever logic)
# ==================================================================

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_programming_tokens(text: str) -> set:
    normalized = _normalize(text)
    tokens = set()

    for tag in re.findall(r"<([a-zA-Z][a-zA-Z0-9-]*)", normalized):
        tokens.add(f"<{tag}>")

    for attr in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*(?=\s*=)", normalized):
        tokens.add(attr.lower().strip())

    for m in re.findall(r"\b[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(", normalized):
        tokens.add(m.replace("(", "").strip().lower())

    for ident in re.findall(r"\b[a-zA-Z_$][a-zA-Z0-9_$-]{2,}\b", normalized):
        tokens.add(ident.lower().strip())

    return tokens


def _extract_keywords(query: str) -> set:
    stopwords = {
        "apa","itu","yang","dan","di","ke","dari","untuk","dalam","pada",
        "dengan","adalah","fungsi","kegunaan","tujuan","cara","bagaimana",
        "mengapa","kenapa","sebutkan","jelaskan","digunakan","penggunaan",
        "sebuah","suatu","bisa","dapat","agar","akan","sebagai",
        "the","is","what","how","why","of","in","on","for","with","a",
        "an","to","and","does","do","are","used","use","purpose","function",
        "explain","define",
    }
    normalized = _normalize(query)
    programming_tokens = _extract_programming_tokens(query)
    words = re.findall(r"[a-zA-Z0-9_-]+", normalized)
    keywords = {w for w in words if w not in stopwords and len(w) >= 3}
    keywords.update(programming_tokens)
    return keywords


# ==================================================================
# PUBLIC: compute_scores
# ==================================================================

def compute_scores(
    query: str,
    content: str,
    distance: float,
    max_distance: float,
) -> dict:
    """
    Hitung semua skor untuk satu pasang (query, chunk).

    Returns:
        dict dengan kunci: distance, semantic, keyword,
        exact_token, fuzzy_token, html_tag, relevance
    """

    # Semantic
    if distance >= max_distance:
        semantic = 0.0
    else:
        semantic = max(0.0, min(1.0, 1.0 - distance / max_distance))

    # Keyword
    qk = _extract_keywords(query)
    ck = _extract_keywords(content)
    if not qk or not ck:
        keyword = 0.0
    else:
        exact_kw = qk & ck
        if exact_kw:
            keyword = len(exact_kw) / max(len(qk), 1)
        else:
            fuzzy_kw = sum(
                1 for qw in qk
                if not qw.startswith("<") and len(qw) >= 4
                and max(
                    (SequenceMatcher(None, qw, cw).ratio()
                     for cw in ck if not cw.startswith("<")),
                    default=0.0,
                ) >= 0.82
            )
            keyword = fuzzy_kw / max(len(qk), 1)

    # Token helpers
    def _norm_tok(t):
        return t.lower().strip().rstrip("()")

    def _tok_sim(qt, ct):
        qt, ct = _norm_tok(qt), _norm_tok(ct)
        if not qt or not ct:
            return 0.0
        if qt == ct:
            return 1.0
        if qt.startswith("<") or ct.startswith("<"):
            return 0.0
        if len(qt) < 5:
            return 0.0
        return SequenceMatcher(None, qt, ct).ratio()

    qt = _extract_programming_tokens(query)
    ct = _extract_programming_tokens(content)

    # Exact token
    exact_token = (
        len(qt & ct) / max(len(qt), 1)
        if qt and ct else 0.0
    )

    # Fuzzy token
    if not qt or not ct:
        fuzzy_token = 0.0
    else:
        matched_fuzzy = sum(
            1 for qtt in qt
            if max((_tok_sim(qtt, ctt) for ctt in ct), default=0.0) >= 0.82
        )
        fuzzy_token = matched_fuzzy / max(len(qt), 1)

    # HTML tag
    q_tags = {
        f"<{t}>"
        for t in re.findall(r"<([a-zA-Z][a-zA-Z0-9-]*)", query.lower())
    }
    c_tags = {
        f"<{t}>"
        for t in re.findall(r"<([a-zA-Z][a-zA-Z0-9-]*)", content.lower())
    }
    html_tag = (
        len(q_tags & c_tags) / max(len(q_tags), 1)
        if q_tags else 0.0
    )

    # Relevance (weighted)
    token_score = max(exact_token, fuzzy_token)
    relevance = round(
        (semantic * 0.50) + (keyword * 0.25) + (token_score * 0.25),
        3,
    )

    return {
        "distance":    round(distance, 4),
        "semantic":    round(semantic, 3),
        "keyword":     round(keyword, 3),
        "exact_token": round(exact_token, 3),
        "fuzzy_token": round(fuzzy_token, 3),
        "html_tag":    round(html_tag, 3),
        "relevance":   relevance,
    }


# ==================================================================
# PUBLIC: raw_query
# ==================================================================

def raw_query(vectordb: LessonVectorDB, query: str, k: int = 50):
    """
    Query ChromaDB langsung (tanpa filter accept/reject).
    Mengembalikan list of (Document, distance).
    """
    return (
        vectordb
        .get_db()
        .similarity_search_with_score(query, k=k)
    )


# ==================================================================
# ACCEPTANCE RULE (mirrors LessonRetriever._is_relevant)
# ==================================================================

def is_accepted(scores: dict, min_relevance: float = 0.34) -> bool:
    """
    Terapkan aturan penerimaan yang sama dengan produksi.
    """
    return (
        scores["exact_token"] >= 0.5
        or scores["fuzzy_token"] >= 0.5
        or scores["html_tag"]   >= 0.5
        or scores["relevance"]  >= min_relevance
        or (scores["distance"] <= 0.25 and scores["keyword"] >= 0.10)
        or (scores["distance"] <= 0.30 and scores["exact_token"] > 0.0)
    )
