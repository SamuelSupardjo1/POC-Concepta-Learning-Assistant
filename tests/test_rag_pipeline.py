"""
Unified Diagnostic Robustness Test Suite (T01 - T30)
Intelligent Learning Assistant - CONCEPTA

Purpose:
    Runs all 30 test cases in a single unified script:
    - Performs diagnostic testing on both normal concepts and edge cases
    - Evaluates semantic grounding, fallback behavior, typos, and prohibited inputs
    - Outputs a clear, comprehensive status table and final pass rate
"""

import sys
import re
from src.retriever import LessonRetriever
from src.prompt import PromptBuilder
from src.llm import OllamaLLM
from src.rag_pipeline import RAGPipeline

# ------------------------------------------------------------
# UTF-8 ENCODING FOR WINDOWS CONSOLE STABILITY
# ------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
FALLBACK_ANSWER = "The requested information is not available in the lesson."

# ------------------------------------------------------------
# TEST CASES DEFINITION (T01 - T30)
# ------------------------------------------------------------
TEST_CASES = [
    # --- T01 - T10 (Original RAG Pipeline Diagnostics) ---
    {
        "id": "T01",
        "category": "Relevant concept",
        "question": "Apa kegunaan novalidate dalam form HTML?",
        "expected": "ANSWER",
        "keywords": ["novalidate", "mengabaikan validasi data"],
    },
    {
        "id": "T02",
        "category": "Unsupported topic",
        "question": "Apa itu variable dalam Python?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T03",
        "category": "Relevant concept",
        "question": "Apa kegunaan tag <audio>?",
        "expected": "ANSWER",
        "keywords": ["audio"],
    },
    {
        "id": "T04",
        "category": "Relevant concept",
        "question": "Apa kegunaan atribut value pada input?",
        "expected": "ANSWER",
        "keywords": ["value"],
    },
    {
        "id": "T05",
        "category": "Relevant concept",
        "question": "Apa fungsi child selector dalam CSS?",
        "expected": "ANSWER",
        "keywords": ["child", "selector"],
    },
    {
        "id": "T06",
        "category": "Unsupported topic",
        "question": "Apa kegunaan class dalam Python?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T07",
        "category": "Unsupported topic",
        "question": "Bagaimana cara membuat API menggunakan FastAPI?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T08",
        "category": "Mixed topic (Partial)",
        "question": "Apa kegunaan novalidate dalam form HTML dan bagaimana cara menggunakannya pada framework React?",
        "expected": "PARTIAL",
        "keywords": ["novalidate"],
    },
    {
        "id": "T09",
        "category": "Prohibited: code generation",
        "question": "Buatkan kode HTML menggunakan novalidate.",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T10",
        "category": "Prohibited: debugging",
        "question": "Kenapa kode HTML saya error?",
        "expected": "FALLBACK",
        "keywords": [],
    },

    # --- T11 - T30 (Original Robustness Test Suite) ---
    {
        "id": "T11",
        "category": "Normal concept",
        "question": "Apa kegunaan atribut action pada form?",
        "expected": "ANSWER",
        "keywords": ["action"],
    },
    {
        "id": "T12",
        "category": "Normal concept",
        "question": "Apa fungsi atribut value pada input?",
        "expected": "ANSWER",
        "keywords": ["value"],
    },
    {
        "id": "T13",
        "category": "Normal concept",
        "question": "Apa kegunaan DOM method?",
        "expected": "ANSWER",
        "keywords": ["dom"],
    },
    {
        "id": "T14",
        "category": "Normal concept",
        "question": "Apa fungsi addEventListener?",
        "expected": "ANSWER",
        "keywords": ["addeventlistener"],
    },
    {
        "id": "T15",
        "category": "Normal concept",
        "question": "Apa itu CSS?",
        "expected": "ANSWER",
        "keywords": ["css"],
    },
    {
        "id": "T16",
        "category": "Unsupported topic",
        "question": "Apa itu Python?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T17",
        "category": "Unsupported topic",
        "question": "Apa itu React?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T18",
        "category": "Unsupported topic",
        "question": "Bagaimana cara membuat REST API?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T19",
        "category": "Unsupported topic",
        "question": "Apa itu machine learning?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T20",
        "category": "Unsupported topic",
        "question": "Bagaimana cara menggunakan database MySQL?",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T21",
        "category": "Typo / fuzzy",
        "question": "Apa kegunaan novalidat dalam form HTML?",
        "expected": "ANSWER",
        "keywords": ["novalidate"],
    },
    {
        "id": "T22",
        "category": "Typo / fuzzy",
        "question": "Apa kegunaan addEventListner?",
        "expected": "ANSWER",
        "keywords": ["addeventlistener"],
    },
    {
        "id": "T23",
        "category": "Typo / fuzzy",
        "question": "Apa kegunaan valur pada input?",
        "expected": "ANSWER",
        "keywords": ["value"],
    },
    {
        "id": "T24",
        "category": "Programming identifier",
        "question": "Apa kegunaan tag <a>?",
        "expected": "ANSWER",
        "keywords": ["<a>"],
    },
    {
        "id": "T25",
        "category": "Programming identifier",
        "question": "Apa fungsi href?",
        "expected": "ANSWER",
        "keywords": ["href"],
    },
    {
        "id": "T26",
        "category": "Programming identifier",
        "question": "Apa kegunaan value pada input?",
        "expected": "ANSWER",
        "keywords": ["value"],
    },
    {
        "id": "T27",
        "category": "Mixed topic (Partial)",
        "question": "Apa kegunaan novalidate pada form HTML dan bagaimana cara menggunakannya di React?",
        "expected": "PARTIAL",
        "keywords": ["novalidate"],
    },
    {
        "id": "T28",
        "category": "Mixed topic (Partial)",
        "question": "Apa fungsi value pada input dan bagaimana cara membuat validasi menggunakan JavaScript?",
        "expected": "PARTIAL",
        "keywords": ["value"],
    },
    {
        "id": "T29",
        "category": "Prohibited: code generation",
        "question": "Buatkan kode HTML lengkap menggunakan novalidate.",
        "expected": "FALLBACK",
        "keywords": [],
    },
    {
        "id": "T30",
        "category": "Prohibited: debugging",
        "question": "Debug kode form HTML saya yang error.",
        "expected": "FALLBACK",
        "keywords": [],
    },
]

# ------------------------------------------------------------
# EVALUATION UTILITIES
# ------------------------------------------------------------
def normalize(text):
    """Normalize text by converting to lowercase and stripping extra whitespaces."""
    return " ".join((text or "").lower().strip().split())


def evaluate(expected, keywords, answer):
    """
    Evaluate actual answer against expected behavior.
    Handles FALLBACK, ANSWER/SUPPORTED, and PARTIAL.
    """
    # Normalize answer, strip any enclosing double/single quotes
    cleaned_answer = (answer or "").strip()
    if len(cleaned_answer) >= 2:
        if (cleaned_answer.startswith('"') and cleaned_answer.endswith('"')) or (
            cleaned_answer.startswith("'") and cleaned_answer.endswith("'")
        ):
            cleaned_answer = cleaned_answer[1:-1].strip()

    a_norm = normalize(cleaned_answer)
    fallback_norm = normalize(FALLBACK_ANSWER)

    if expected == "FALLBACK":
        return a_norm == fallback_norm

    # Check for keywords matching helper
    keyword_ok = all(normalize(k) in a_norm for k in keywords)

    # Standard answer containing lesson concept
    if expected in ("ANSWER", "SUPPORTED"):
        return a_norm != fallback_norm and keyword_ok

    # Partial / mixed support: answer supported part + append fallback sentence
    if expected == "PARTIAL":
        return a_norm != fallback_norm and keyword_ok and fallback_norm in a_norm

    return False


# ------------------------------------------------------------
# MAIN EXECUTION RUNNER
# ------------------------------------------------------------
def main():
    print("=" * 75)
    print("CONCEPTA UNIFIED ROBUSTNESS TEST SUITE T01-T30")
    print("=" * 75)
    print("Initializing RAG Pipeline...")

    try:
        # Initializing the pipeline with override settings for safety
        retriever = LessonRetriever(
            top_k=3,
            candidate_k=50,
            min_relevance=0.34,
            max_distance=0.38,
        )
        prompt_builder = PromptBuilder()
        llm = OllamaLLM()
        
        pipeline = RAGPipeline(
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm
        )
        print("Pipeline initialized successfully.\n")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    results = []
    passed_count = 0

    for test in TEST_CASES:
        t_id = test["id"]
        category = test["category"]
        question = test["question"]
        expected = test["expected"]
        keywords = test["keywords"]

        print("*" * 90)
        print(f"CASE {t_id} | Category: {category}")
        print(f"QUESTION : {question}")
        print(f"EXPECTED : {expected}")
        if keywords:
            print(f"KEYWORDS : {', '.join(keywords)}")
        print("-" * 50)

        try:
            answer = pipeline.ask(question)
            ok = evaluate(expected, keywords, answer)
        except Exception as err:
            answer = f"ERROR EXCEPTION: {err}"
            ok = False

        status = "PASS" if ok else "FAIL"
        if ok:
            passed_count += 1

        results.append({
            "id": t_id,
            "category": category,
            "expected": expected,
            "status": status
        })

        # Safeguard print encoding errors
        try:
            print(f"ANSWER   : {answer}")
        except UnicodeEncodeError:
            print(f"ANSWER   : {answer.encode('ascii', errors='replace').decode('ascii')}")
        print(f"STATUS   : {status}")
        print()

    # ------------------------------------------------------------
    # PRINT END-TO-END SUMMARY TABLE
    # ------------------------------------------------------------
    total = len(TEST_CASES)
    failed_count = total - passed_count
    pass_rate = (passed_count / total) * 100

    print("=" * 75)
    print("         UNIFIED ROBUSTNESS TEST STATUS SUMMARY TABLE")
    print("=" * 75)
    print(f"{'ID':<6} | {'Category':<30} | {'Expected':<12} | {'Status':<8}")
    print("-" * 75)
    for res in results:
        print(f"{res['id']:<6} | {res['category']:<30} | {res['expected']:<12} | {res['status']:<8}")
    print("=" * 75)

    print(f"TOTAL TESTS : {total}")
    print(f"PASSED      : {passed_count}")
    print(f"FAILED      : {failed_count}")
    print(f"PASS RATE   : {pass_rate:.1f}%")
    print("=" * 75)

    if failed_count > 0:
        print("RESULT      : SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("RESULT      : ALL TESTS PASSED SUCCESSFULLY! 🎉")
        sys.exit(0)


if __name__ == "__main__":
    main()
