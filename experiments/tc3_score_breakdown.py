"""
experiments/tc3_score_breakdown.py
====================================
TC-3: Score Breakdown Report

Untuk setiap kueri dan setiap chunk kandidat (theory-only),
menampilkan tabel breakdown skor secara lengkap:

    dist | sem | kw | exact | fuzzy | html | rel | ok | content preview

Kolom penjelasan:
    dist  — cosine distance dari ChromaDB (↓ lebih baik)
    sem   — semantic score  = 1 - dist / max_distance
    kw    — keyword score   (lexical overlap setelah stopword filter)
    exact — exact programming token score
    fuzzy — fuzzy programming token score (SequenceMatcher ≥ 0.82)
    html  — HTML tag intersection score
    rel   — final relevance = sem×0.5 + kw×0.25 + max(exact,fuzzy)×0.25
    ok    — ✓ diterima / ✗ ditolak oleh _is_relevant()

Cara menjalankan:
    python -m experiments.tc3_score_breakdown
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from experiments.retrieval_helpers import (
    QUERY_DATASET,
    compute_scores,
    raw_query,
    is_accepted,
)


# ==================================================================
# CONFIGURATION
# ==================================================================

MAX_DISTANCE  = 0.38   # nilai yang ingin dievaluasi
MIN_RELEVANCE = 0.34
CANDIDATE_K   = 50
CONTENT_PREVIEW_LEN = 80


# ==================================================================
# TABLE HEADER
# ==================================================================

_COL = {
    "dist":  8,
    "sem":   7,
    "kw":    7,
    "exact": 7,
    "fuzzy": 7,
    "html":  7,
    "rel":   7,
    "ok":    4,
}

_HEADER = (
    f"{'dist':>{_COL['dist']}}"
    f"  {'sem':>{_COL['sem']}}"
    f"  {'kw':>{_COL['kw']}}"
    f"  {'exact':>{_COL['exact']}}"
    f"  {'fuzzy':>{_COL['fuzzy']}}"
    f"  {'html':>{_COL['html']}}"
    f"  {'rel':>{_COL['rel']}}"
    f"  {'ok':>{_COL['ok']}}"
    f"  content preview"
)

_SEP = "─" * 130


# ==================================================================
# TEST CASE
# ==================================================================

def run(vectordb: LessonVectorDB):

    print()
    print("=" * 80)
    print("TC-3  SCORE BREAKDOWN REPORT")
    print(f"  max_distance  = {MAX_DISTANCE}")
    print(f"  min_relevance = {MIN_RELEVANCE}")
    print(f"  candidate_k   = {CANDIDATE_K}")
    print(f"  Queries       = {len(QUERY_DATASET)}")
    print("=" * 80)

    for entry in QUERY_DATASET:
        qid      = entry["id"]
        category = entry["category"]
        query    = entry["query"]

        print()
        print(_SEP)
        print(f"[{qid}] [{category}]  {query}")
        print(_SEP)
        print(_HEADER)
        print(_SEP)

        results = raw_query(vectordb, query, k=CANDIDATE_K)
        row_count = 0

        for doc, distance in results:
            content  = (getattr(doc, "page_content", "") or "").strip()
            metadata = getattr(doc, "metadata", {}) or {}

            # Theory-only filter
            if str(metadata.get("content_type", "")).lower().strip() != "theory":
                continue

            s  = compute_scores(query, content, distance, MAX_DISTANCE)
            ok = is_accepted(s, MIN_RELEVANCE)

            ok_str  = "✓" if ok else "✗"
            preview = content[:CONTENT_PREVIEW_LEN].replace("\n", " ")

            row = (
                f"{s['distance']:>{_COL['dist']}.4f}"
                f"  {s['semantic']:>{_COL['sem']}.3f}"
                f"  {s['keyword']:>{_COL['kw']}.3f}"
                f"  {s['exact_token']:>{_COL['exact']}.3f}"
                f"  {s['fuzzy_token']:>{_COL['fuzzy']}.3f}"
                f"  {s['html_tag']:>{_COL['html']}.3f}"
                f"  {s['relevance']:>{_COL['rel']}.3f}"
                f"  {ok_str:>{_COL['ok']}}"
                f"  {preview}"
            )
            print(row)
            row_count += 1

        if row_count == 0:
            print("  (Tidak ada kandidat theory ditemukan dalam top-50)")

    # ------------------------------------------------------------------
    # Legenda
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Legenda:")
    print("  dist  = cosine distance ChromaDB        (↓ lebih dekat = lebih relevan)")
    print("  sem   = semantic score                  (1 - dist / max_distance)")
    print("  kw    = keyword score                   (lexical overlap ÷ total keywords)")
    print("  exact = exact programming token score   (irisan eksak token teknis)")
    print("  fuzzy = fuzzy programming token score   (SequenceMatcher ≥ 0.82)")
    print("  html  = HTML tag intersection score     (irisan tag HTML eksak)")
    print("  rel   = relevance = sem×0.50 + kw×0.25 + max(exact,fuzzy)×0.25")
    print("  ok    = ✓ diterima  /  ✗ ditolak oleh aturan _is_relevant()")
    print()
    print("Aturan penerimaan (salah satu harus terpenuhi):")
    print("  • exact  ≥ 0.50")
    print("  • fuzzy  ≥ 0.50")
    print("  • html   ≥ 0.50")
    print(f"  • rel    ≥ {MIN_RELEVANCE}")
    print("  • dist   ≤ 0.25  AND  kw ≥ 0.10")
    print("  • dist   ≤ 0.30  AND  exact > 0")
    print()
    print("=" * 80)
    print("TC-3 SELESAI")
    print("=" * 80)


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 80)
    print("CONCEPTA — TC-3: Score Breakdown Report")
    print("=" * 80)
    print("Initializing embedding model and vector database...")

    embedding = EmbeddingModel().get_model()
    vectordb  = LessonVectorDB(embedding)

    print("Ready.\n")

    run(vectordb)


if __name__ == "__main__":
    main()
