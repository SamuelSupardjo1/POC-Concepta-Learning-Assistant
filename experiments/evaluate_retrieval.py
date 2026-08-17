"""
experiments/evaluate_retrieval.py
====================================
Orchestrator — menjalankan semua TC retrieval secara berurutan.

TC-1: Top-5 Chunk Inspection    → experiments/tc1_top5_chunks.py
TC-2: Max-Distance Sweep        → experiments/tc2_max_distance.py
TC-3: Score Breakdown Report    → experiments/tc3_score_breakdown.py

Cara menjalankan SEMUA test case sekaligus:
    python -m experiments.evaluate_retrieval

Cara menjalankan satu test case saja:
    python -m experiments.tc1_top5_chunks
    python -m experiments.tc2_max_distance
    python -m experiments.tc3_score_breakdown
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.embedding import EmbeddingModel
from src.retriever import LessonRetriever
from src.vectordb import LessonVectorDB

import experiments.tc1_top5_chunks    as tc1
import experiments.tc2_max_distance   as tc2
import experiments.tc3_score_breakdown as tc3


def main():
    print("=" * 80)
    print("CONCEPTA — RETRIEVAL EVALUATION SUITE (ALL TEST CASES)")
    print("TC-1: Top-5 Chunks | TC-2: Max-Distance | TC-3: Score Breakdown")
    print("=" * 80)
    print("Initializing shared resources...")

    embedding = EmbeddingModel().get_model()
    vectordb  = LessonVectorDB(embedding)
    retriever = LessonRetriever(
        top_k=5,
        candidate_k=50,
        min_relevance=0.34,
        max_distance=0.38,
    )

    print("Ready.\n")

    # ------------------------------------------------------------------
    tc1.run(retriever)

    # ------------------------------------------------------------------
    tc2.run(vectordb)

    # ------------------------------------------------------------------
    tc3.run(vectordb)

    print()
    print("=" * 80)
    print("SEMUA TEST CASE SELESAI DIJALANKAN.")
    print("=" * 80)


if __name__ == "__main__":
    main()
