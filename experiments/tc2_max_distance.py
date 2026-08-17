"""
experiments/tc2_max_distance.py
=================================
TC-2: Max-Distance Sweep

Menjalankan grid search atas nilai max_distance dari 0.20 hingga
0.55.  Untuk setiap nilai, menghitung jumlah chunk theory yang
diterima retriever per kueri.

Tujuan: membantu menentukan nilai max_distance yang optimal —
cukup longgar untuk menangkap konsep relevan, cukup ketat untuk
memblokir topik di luar silabus.

Target:
    Kueri Relevant   → 3–5 chunk diterima
    Kueri Unsupported→ 0   chunk diterima

Cara menjalankan:
    python -m experiments.tc2_max_distance
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

SWEEP_VALUES  = [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55]
MIN_RELEVANCE = 0.34   # fixed; sama dengan konfigurasi produksi
CANDIDATE_K   = 50


# ==================================================================
# TEST CASE
# ==================================================================

def run(vectordb: LessonVectorDB):

    print()
    print("=" * 80)
    print("TC-2  MAX-DISTANCE SWEEP")
    print(f"  Sweep         : {SWEEP_VALUES}")
    print(f"  min_relevance : {MIN_RELEVANCE} (fixed)")
    print(f"  candidate_k   : {CANDIDATE_K}")
    print(f"  Queries       : {len(QUERY_DATASET)}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Tabel header
    # ------------------------------------------------------------------
    COL_W = 10   # lebar kolom max_dist
    QID_W = 6    # lebar kolom per kueri

    header = f"{'max_dist':>{COL_W}}"
    for entry in QUERY_DATASET:
        header += f"  {entry['id']:>{QID_W}}"
    header += f"  {'AVG_REL':>7}"
    header += f"  {'AVG_UNS':>7}"

    print()
    print(header)
    print("─" * (COL_W + (QID_W + 2) * len(QUERY_DATASET) + 20))

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    for max_dist in SWEEP_VALUES:

        row     = f"{max_dist:>{COL_W}.2f}"
        counts  = []

        relevant_counts    = []
        unsupported_counts = []

        for entry in QUERY_DATASET:
            query    = entry["query"]
            category = entry["category"]

            results  = raw_query(vectordb, query, k=CANDIDATE_K)

            accepted = 0
            for doc, distance in results:
                content  = (getattr(doc, "page_content", "") or "").strip()
                metadata = getattr(doc, "metadata", {}) or {}

                if str(metadata.get("content_type", "")).lower().strip() != "theory":
                    continue

                scores = compute_scores(query, content, distance, max_dist)
                if is_accepted(scores, MIN_RELEVANCE):
                    accepted += 1

            counts.append(accepted)
            row += f"  {accepted:>{QID_W}}"

            if category in ("Relevant", "Typo"):
                relevant_counts.append(accepted)
            elif category == "Unsupported":
                unsupported_counts.append(accepted)

        avg_rel = (
            sum(relevant_counts) / len(relevant_counts)
            if relevant_counts else 0.0
        )
        avg_uns = (
            sum(unsupported_counts) / len(unsupported_counts)
            if unsupported_counts else 0.0
        )

        row += f"  {avg_rel:>7.2f}"
        row += f"  {avg_uns:>7.2f}"

        # Tandai baris optimal (sweet spot)
        is_sweet = (2.0 <= avg_rel <= 6.0 and avg_uns <= 0.5)
        if is_sweet:
            row += "  ← kandidat optimal"

        print(row)

    # ------------------------------------------------------------------
    # Legenda
    # ------------------------------------------------------------------
    print()
    print("Keterangan kolom:")
    print(f"  Q01–Q09  = kueri Relevant (target: 3–5 diterima)")
    print(f"  Q10–Q12  = kueri Typo     (target: ≥ 1 diterima)")
    print(f"  Q13–Q14  = kueri Unsupported (target: 0 diterima)")
    print(f"  AVG_REL  = rata-rata chunk diterima untuk Relevant+Typo")
    print(f"  AVG_UNS  = rata-rata chunk diterima untuk Unsupported")
    print(f"  ← kandidat optimal: AVG_REL 2-6, AVG_UNS ≤ 0.5")
    print()
    print("=" * 80)
    print("TC-2 SELESAI")
    print("=" * 80)


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 80)
    print("CONCEPTA — TC-2: Max-Distance Sweep")
    print("=" * 80)
    print("Initializing embedding model and vector database...")

    embedding = EmbeddingModel().get_model()
    vectordb  = LessonVectorDB(embedding)

    print("Ready.\n")

    run(vectordb)


if __name__ == "__main__":
    main()
