"""
experiments/tc1_top5_chunks.py
================================
TC-1: Top-5 Chunk Inspection

Untuk setiap kueri, menampilkan 5 chunk teratas yang diterima
oleh production retriever (dengan semua filter aktif) beserta
metadata lengkap dan preview konten.

Cara menjalankan:
    python -m experiments.tc1_top5_chunks
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.retriever import LessonRetriever
from experiments.retrieval_helpers import QUERY_DATASET


# ==================================================================
# CONFIGURATION
# ==================================================================

RETRIEVER_CONFIG = {
    "top_k":        5,
    "candidate_k":  50,
    "min_relevance": 0.34,
    "max_distance":  0.38,
}

CONTENT_PREVIEW_LEN = 200


# ==================================================================
# TEST CASE
# ==================================================================

def run(retriever: LessonRetriever):

    print()
    print("=" * 80)
    print("TC-1  TOP-5 CHUNK INSPECTION")
    print(f"  top_k         = {RETRIEVER_CONFIG['top_k']}")
    print(f"  max_distance  = {RETRIEVER_CONFIG['max_distance']}")
    print(f"  min_relevance = {RETRIEVER_CONFIG['min_relevance']}")
    print(f"  Queries       = {len(QUERY_DATASET)}")
    print("=" * 80)

    for entry in QUERY_DATASET:
        qid      = entry["id"]
        category = entry["category"]
        query    = entry["query"]

        print()
        print(f"{'─' * 80}")
        print(f"[{qid}] [{category}]")
        print(f"Query: {query}")
        print(f"{'─' * 80}")

        docs = retriever.retrieve(query)

        if not docs:
            print("  ⚠  Tidak ada chunk yang diterima retriever.")
            continue

        for rank, doc in enumerate(docs[:5], start=1):
            content  = (getattr(doc, "page_content", "") or "").strip()
            metadata = getattr(doc, "metadata", {}) or {}

            lesson  = metadata.get("lesson",       "—")
            section = metadata.get("section",      "—")
            page    = metadata.get("page",         "—")
            source  = metadata.get("source",       "—")
            ctype   = metadata.get("content_type", "—")

            preview = content[:CONTENT_PREVIEW_LEN].replace("\n", " ")
            if len(content) > CONTENT_PREVIEW_LEN:
                preview += "..."

            print(f"\n  Rank   : #{rank}")
            print(f"  Lesson : {lesson}")
            print(f"  Section: {section}")
            print(f"  Page   : {page}")
            print(f"  Source : {source}")
            print(f"  Type   : {ctype}")
            print(f"  Preview: {preview}")

    print()
    print("=" * 80)
    print("TC-1 SELESAI")
    print("=" * 80)


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 80)
    print("CONCEPTA — TC-1: Top-5 Chunk Inspection")
    print("=" * 80)
    print("Initializing retriever...")

    retriever = LessonRetriever(**RETRIEVER_CONFIG)

    print("Retriever ready.\n")

    run(retriever)


if __name__ == "__main__":
    main()
