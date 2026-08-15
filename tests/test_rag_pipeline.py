"""
RAG Pipeline Black Box Test Suite

Tests:
1. Relevant lesson question
2. Unsupported question
3. Code-related question
4. Empty question
5. Completely unrelated question
6. Paraphrased relevant question
7. Another paraphrased question
8. Unsupported programming topic
9. English relevant question
10. Another unrelated question
11. Code snippet with relevant attribute
12. Code generation request
13. Debugging request
14. Code modification request
15. Programming exercise request
16. Partial / mixed unsupported question
17. Multiple concepts question
18. Typo in relevant question
19. Different wording / synonym
20. Non-programming question

Expected behavior:
- Relevant questions -> answer from lesson context
- Unsupported questions -> exact fallback response
- Code generation/debugging/modification/exercise -> must not perform the prohibited task
"""

from pathlib import Path
import sys
import re

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# IMPORT
# ============================================================

from src.rag_pipeline import RAGPipeline
from src.retriever import LessonRetriever

# ============================================================
# CONSTANT
# ============================================================

FALLBACK = "The requested information is not available in the lesson."


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. Relevant lesson question
    # --------------------------------------------------------
    {
        "name": "Relevant lesson question",
        "question": "Apa kegunaan novalidate dalam form HTML?",
        "expected": "ANSWER",
        "required_keywords": [
            "novalidate",
            "mengabaikan validasi data"
        ],
    },

    # --------------------------------------------------------
    # 2. Supported HTML concept question
    # --------------------------------------------------------
    {
        "name": "HTML concept question",
        "question": "what is html",
        "expected": "ANSWER",
        "required_keywords": [
            "html",
            "hypertext markup"
        ],
    },

    # --------------------------------------------------------
    # 3. Definition question for HTML heading concept
    # --------------------------------------------------------
    {
        "name": "Header definition question",
        "question": "Apa itu header?",
        "expected": "ANSWER",
        "required_keywords": [
            "header"
        ],
    },

    # --------------------------------------------------------
    # 4. Footer concept question
    # --------------------------------------------------------
    {
        "name": "Footer concept question",
        "question": "Apa itu footer?",
        "expected": "ANSWER",
        "required_keywords": [
            "footer"
        ],
    },

    # --------------------------------------------------------
    # 5. Unsupported question
    # --------------------------------------------------------
    {
        "name": "Unsupported question",
        "question": "Apa itu variable dalam Python?",
        "expected": "FALLBACK",
    },

    # --------------------------------------------------------
    # 5. Definition question for anchor tag
    # --------------------------------------------------------
    {
        "name": "Anchor tag definition question",
        "question": "Apa itu <a>?",
        "expected": "ANSWER",
        "required_keywords": [
            "hyperlink",
            "link"
        ],
    },

    # --------------------------------------------------------
    # 6. Code-related question
    # --------------------------------------------------------
    {
        "name": "Code-related question",
        "question": """Apa fungsi atribut novalidate pada kode berikut?

<form novalidate>
""",
        "expected": "CODE_ANSWER",
        "required_keywords": [
            "novalidate",
            "mengabaikan validasi data"
        ],
    },

    # --------------------------------------------------------
    # 4. Empty question
    # --------------------------------------------------------
    {
        "name": "Empty question",
        "question": "",
        "expected": "FALLBACK",
    },

    # --------------------------------------------------------
    # 5. Completely unrelated question
    # --------------------------------------------------------
    {
        "name": "Completely unrelated question",
        "question": "Siapa presiden pertama Indonesia?",
        "expected": "FALLBACK",
    },

    # --------------------------------------------------------
    # 6. Paraphrased relevant question
    # --------------------------------------------------------
    {
        "name": "Paraphrased relevant question",
        "question": "Untuk apa atribut novalidate digunakan?",
        "expected": "ANSWER",
        "required_keywords": [
            "mengabaikan validasi data"
        ],
    },

    # --------------------------------------------------------
    # 7. Another paraphrase
    # --------------------------------------------------------
    {
        "name": "Another paraphrased question",
        "question": "Apa fungsi dari novalidate?",
        "expected": "ANSWER",
        "required_keywords": [
            "novalidate",
            "mengabaikan validasi data"
        ],
    },

    # --------------------------------------------------------
    # 8. Unsupported programming topic
    # --------------------------------------------------------
    {
        "name": "Unsupported programming topic",
        "question": "Bagaimana cara membuat API menggunakan Node.js?",
        "expected": "FALLBACK",
    },

    # --------------------------------------------------------
    # 9. English relevant question
    # --------------------------------------------------------
    {
        "name": "English relevant question",
        "question": "What is the purpose of the novalidate attribute?",
        "expected": "ANSWER",
        "required_keywords": [
            "novalidate",
            "ignore data validation"
        ],
    },

    # --------------------------------------------------------
    # 10. Another unrelated question
    # --------------------------------------------------------
    {
        "name": "Another unrelated question",
        "question": "Berapa ibu kota Jepang?",
        "expected": "FALLBACK",
    },

    # --------------------------------------------------------
    # 11. Code snippet with relevant attribute
    # --------------------------------------------------------
    {
        "name": "Relevant attribute inside code snippet",
        "question": """Jelaskan atribut novalidate pada kode berikut:

<form action="/submit" novalidate>
    <input type="text">
</form>
""",
        "expected": "CODE_ANSWER",
        "required_keywords": [
            "novalidate",
            "mengabaikan validasi data"
        ],
    },

    # --------------------------------------------------------
    # 12. Code generation request
    # --------------------------------------------------------
    {
        "name": "Code generation request",
        "question": """
Buatkan kode HTML form yang menggunakan atribut novalidate.
""",
        "expected": "NO_CODE_GENERATION",
    },

    # --------------------------------------------------------
    # 13. Debugging request
    # --------------------------------------------------------
    {
        "name": "Debugging request",
        "question": """
Kenapa kode HTML form saya error?

<form novalidate>
    <input type="text"
</form>
""",
        "expected": "NO_DEBUGGING",
    },

    # --------------------------------------------------------
    # 14. Code modification request
    # --------------------------------------------------------
    {
        "name": "Code modification request",
        "question": """
Perbaiki kode berikut agar form menggunakan novalidate:

<form>
    <input type="text">
</form>
""",
        "expected": "NO_CODE_MODIFICATION",
    },

    # --------------------------------------------------------
    # 15. Programming exercise request
    # --------------------------------------------------------
    {
        "name": "Programming exercise request",
        "question": """
Kerjakan exercise HTML Form berikut dan berikan kode lengkapnya.
""",
        "expected": "NO_EXERCISE_SOLUTION",
    },

    # --------------------------------------------------------
    # 16. Mixed relevant + unsupported question
    # --------------------------------------------------------
    {
        "name": "Mixed relevant and unsupported question",
        "question": """
Apa kegunaan novalidate dalam form HTML dan bagaimana cara
menggunakannya pada framework React?
""",
        "expected": "ANSWER_WITHOUT_OUTSIDE_KNOWLEDGE",
        "required_keywords": [
            "novalidate",
            "mengabaikan validasi data"
        ],
        "forbidden_keywords": [
            "react",
            "jsx",
            "component",
            "useform",
            "formik"
        ],
    },

    # --------------------------------------------------------
    # 17. Multiple concepts question
    # --------------------------------------------------------
    {
        "name": "Multiple concepts question",
        "question": """
Apa kegunaan novalidate dan apa fungsi atribut value?
""",
        "expected": "MULTI_CONCEPT",
        "required_keywords": [
            "novalidate",
            "validasi data",
        ],
    },

    # --------------------------------------------------------
    # 18. Typo in relevant question
    # --------------------------------------------------------
    {
        "name": "Relevant question with typo",
        "question": """
Apa kegunaan novalidte dalam form HTML?
""",
        "expected": "ANSWER",
        "required_keywords": [
            "novalidate",
            "validasi data"
        ],
    },

    # --------------------------------------------------------
    # 19. Different wording / synonym
    # --------------------------------------------------------
    {
        "name": "Relevant question using different wording",
        "question": """
Apa tujuan penggunaan atribut novalidate pada HTML Form?
""",
        "expected": "ANSWER",
        "required_keywords": [
            "novalidate",
            "validasi data"
        ],
    },

    # --------------------------------------------------------
    # 20. Non-programming question
    # --------------------------------------------------------
    {
        "name": "Non-programming question",
        "question": """
Bagaimana cara memasak nasi goreng?
""",
        "expected": "FALLBACK",
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Keyword synonyms mapping (Indonesian <-> English)
KEYWORD_SYNONYMS = {
    "mengabaikan validasi data": ["ignore data validation", "mengabaikan validasi data"],
    "ignore data validation":    ["ignore data validation", "mengabaikan validasi data"],
    "validasi data":             ["data validation", "validasi data"],
    "data validation":           ["data validation", "validasi data"],
    "novalidate":                ["novalidate"],
    "value":                     ["value"],
    # Anchor tag / hyperlink synonyms
    "anchor":                    ["anchor", "hyperlink", "hyperlink adalah"],
    "link":                      ["link", "hyperlink"],
    "hyperlink":                 ["hyperlink", "anchor", "link"],
    # Footer synonyms
    "footer":                    ["footer", "kaki website", "kaki halaman"],
    "kaki website":              ["footer", "kaki website"],
}


def normalize_text(text: str) -> str:
    """
    Normalize answer for flexible comparison.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def expand_keywords(keywords: list) -> set:
    """
    Expand keywords to include their synonyms.
    
    For example:
    - "mengabaikan validasi data" → includes "ignore data validation"
    - "validasi data" → includes "data validation"
    """
    expanded = set()
    
    for keyword in keywords:
        expanded.add(normalize_text(keyword))
        
        # Add synonyms from mapping
        if keyword.lower() in KEYWORD_SYNONYMS:
            for synonym in KEYWORD_SYNONYMS[keyword.lower()]:
                expanded.add(normalize_text(synonym))
    
    return expanded


def contains_keywords(answer: str, keywords: list) -> bool:
    """
    Check whether all required keywords (or their synonyms) exist in answer.
    
    For each keyword, we check if ANY of its synonyms appear in the answer.
    All keywords must be satisfied (with at least one of their synonyms).
    """
    normalized = normalize_text(answer)

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        
        # Get all possible synonym versions of this keyword
        possible_matches = {keyword_norm}
        
        if keyword.lower() in KEYWORD_SYNONYMS:
            for synonym in KEYWORD_SYNONYMS[keyword.lower()]:
                possible_matches.add(normalize_text(synonym))
        
        # Check if ANY synonym of this keyword exists in the answer
        keyword_found = any(
            match in normalized
            for match in possible_matches
        )
        
        if not keyword_found:
            return False
    
    return True


def contains_forbidden_keywords(answer: str, keywords: list) -> bool:
    """
    Check whether forbidden keywords exist.
    """
    normalized = normalize_text(answer)

    return any(
        normalize_text(keyword) in normalized
        for keyword in keywords
    )


def contains_code(answer: str) -> bool:
    """
    Detect whether the assistant generated actual code.

    This is intentionally simple for black-box testing.
    """
    code_patterns = [
        r"<form\b",
        r"<input\b",
        r"<html\b",
        r"<body\b",
        r"<script\b",
        r"```",
        r"function\s+\w+\s*\(",
        r"const\s+\w+\s*=",
        r"let\s+\w+\s*=",
        r"var\s+\w+\s*=",
    ]

    return any(
        re.search(pattern, answer, re.IGNORECASE)
        for pattern in code_patterns
    )


# ============================================================
# TEST RUNNER
# ============================================================

def run_test(pipeline, test_number: int, test: dict) -> bool:
    print("=" * 70)
    print("=" * 70)
    print("=" * 70)
    print("=" * 70)
    print("=" * 70)
    print(f"[TEST {test_number}] {test['name']}")
    print("=" * 70)

    question = test["question"]

    print()
    print("Question:")
    print(question)

    print()
    print(f"Expected: {test['expected']}")

    # --------------------------------------------------------
    # Execute RAG
    # --------------------------------------------------------

    try:
        answer = pipeline.ask(question)
    except Exception as e:
        print()
        print("[ERROR] Pipeline execution failed.")
        print(f"Error: {e}")
        return False

    if answer is None:
        answer = ""

    answer = str(answer)

    print()
    print("=== ANSWER ===")
    print(answer)

    normalized_answer = normalize_text(answer)

    expected = test["expected"]

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if expected == "FALLBACK":

        if normalized_answer == normalize_text(FALLBACK):
            print()
            print("[PASS] RAG correctly rejected unsupported context.")
            return True

        print()
        print("[FAIL] RAG should return the fallback response.")
        return False

    # --------------------------------------------------------
    # ANSWER / CODE_ANSWER
    # --------------------------------------------------------

    if expected in [
        "ANSWER",
        "CODE_ANSWER",
        "ANSWER_WITHOUT_OUTSIDE_KNOWLEDGE",
        "MULTI_CONCEPT",
    ]:

        required_keywords = test.get(
            "required_keywords",
            []
        )

        if required_keywords:

            if not contains_keywords(
                answer,
                required_keywords
            ):
                print()
                print(
                    "[FAIL] Answer does not contain "
                    "the required lesson information."
                )

                print()
                print("Required keywords:")
                for keyword in required_keywords:
                    print(f"  - {keyword}")

                return False

        forbidden_keywords = test.get(
            "forbidden_keywords",
            []
        )

        if forbidden_keywords:

            if contains_forbidden_keywords(
                answer,
                forbidden_keywords
            ):
                print()
                print(
                    "[FAIL] Answer contains information "
                    "that should not come from outside the lesson."
                )

                print()
                print("Forbidden keywords found:")
                for keyword in forbidden_keywords:
                    if normalize_text(keyword) in normalized_answer:
                        print(f"  - {keyword}")

                return False

        print()
        print(
            "[PASS] RAG answered using supported "
            "lesson information."
        )

        return True

    # --------------------------------------------------------
    # NO CODE GENERATION
    # --------------------------------------------------------

    if expected == "NO_CODE_GENERATION":

        if contains_code(answer):
            print()
            print(
                "[FAIL] RAG generated code "
                "despite the restriction."
            )
            return False

        print()
        print(
            "[PASS] RAG did not generate complete code."
        )

        return True

    # --------------------------------------------------------
    # NO DEBUGGING
    # --------------------------------------------------------

    if expected == "NO_DEBUGGING":

        if contains_code(answer):
            print()
            print(
                "[FAIL] RAG appears to provide "
                "modified/debugged code."
            )
            return False

        print()
        print(
            "[PASS] RAG did not provide debugging code."
        )

        return True

    # --------------------------------------------------------
    # NO CODE MODIFICATION
    # --------------------------------------------------------

    if expected == "NO_CODE_MODIFICATION":

        if contains_code(answer):
            print()
            print(
                "[FAIL] RAG modified/generated code."
            )
            return False

        print()
        print(
            "[PASS] RAG did not modify or generate code."
        )

        return True

    # --------------------------------------------------------
    # NO EXERCISE SOLUTION
    # --------------------------------------------------------

    if expected == "NO_EXERCISE_SOLUTION":

        if contains_code(answer):
            print()
            print(
                "[FAIL] RAG generated an exercise solution."
            )
            return False

        print()
        print(
            "[PASS] RAG did not solve the programming exercise."
        )

        return True

    # --------------------------------------------------------
    # Unknown expectation
    # --------------------------------------------------------

    print()
    print(
        f"[FAIL] Unknown expected result: {expected}"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAG PIPELINE BLACK BOX TEST SUITE")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    try:
        retriever = LessonRetriever(
            top_k=3,
            distance_threshold=0.38,
        )

        pipeline = RAGPipeline(
            retriever=retriever
        )

    except Exception as e:

        print("[ERROR] Failed to initialize RAG pipeline.")
        print(f"Error: {e}")

        return

    # --------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------

    passed = 0
    failed = 0

    results = []

    for number, test in enumerate(
        TEST_CASES,
        start=1
    ):

        try:

            result = run_test(
                pipeline,
                number,
                test
            )

        except Exception as e:

            print()
            print("[ERROR] Unexpected test error.")
            print(f"Error: {e}")

            result = False

        results.append(
            {
                "number": number,
                "name": test["name"],
                "passed": result,
            }
        )

        if result:
            passed += 1
        else:
            failed += 1

        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(TEST_CASES)

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Total  : {total}")

    print()

    # --------------------------------------------------------
    # Detailed result
    # --------------------------------------------------------

    print("TEST RESULTS")
    print("-" * 70)

    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"Test {result['number']}: "
            f"{result['name']}"
        )

    print()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if failed == 0:

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    else:

        print("=" * 70)
        print("SOME TESTS FAILED")
        print("=" * 70)


if __name__ == "__main__":
    main()
