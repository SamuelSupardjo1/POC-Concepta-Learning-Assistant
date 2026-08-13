from src.retriever import LessonRetriever
from src.rag_pipeline import RAGPipeline


FALLBACK = RAGPipeline.FALLBACK_ANSWER


TEST_CASES = [
    {
        "name": "Relevant lesson question",
        "question": "Apa kegunaan novalidate dalam form HTML?",
        "expected": "ANSWER",
    },
    {
        "name": "Unsupported question",
        "question": "Apa itu variable dalam Python?",
        "expected": "FALLBACK",
    },
    {
        "name": "Code-related question",
        "question": """Apa fungsi atribut novalidate pada kode berikut?

<form novalidate>
""",
        "expected": "CODE_ANSWER",
    },
    {
        "name": "Empty question",
        "question": "",
        "expected": "FALLBACK",
    },
    {
        "name": "Completely unrelated question",
        "question": "Siapa presiden pertama Indonesia?",
        "expected": "FALLBACK",
    },
    {
        "name": "Paraphrased relevant question",
        "question": "Untuk apa atribut novalidate digunakan?",
        "expected": "ANSWER",
    },
    {
        "name": "Another paraphrased question",
        "question": "Apa fungsi dari novalidate?",
        "expected": "ANSWER",
    },
    {
        "name": "Unsupported programming topic",
        "question": "Bagaimana cara membuat API menggunakan Node.js?",
        "expected": "FALLBACK",
    },
    {
        "name": "English relevant question",
        "question": "What is the purpose of the novalidate attribute?",
        "expected": "ANSWER",
    },
    {
        "name": "Another unrelated question",
        "question": "Berapa ibu kota Jepang?",
        "expected": "FALLBACK",
    },
]


def classify_result(answer: str, expected: str) -> bool:
    """
    Determine whether the generated answer matches
    the expected behavior.
    """

    answer_lower = answer.lower().strip()

    if expected == "FALLBACK":
        return answer.strip() == FALLBACK

    if expected in ("ANSWER", "CODE_ANSWER"):

        # Must not return fallback.
        if answer.strip() == FALLBACK:
            return False

        # Must contain relevant lesson terminology.
        if "novalidate" not in answer_lower:
            return False

        return True

    return False


def run_test(
    test_number: int,
    test_case: dict,
    pipeline: RAGPipeline,
) -> bool:

    print("\n" + "=" * 70)
    print(
        f"[TEST {test_number}] "
        f"{test_case['name']}"
    )
    print("=" * 70)

    question = test_case["question"]
    expected = test_case["expected"]

    print("\nQuestion:")
    print(question)

    print(f"\nExpected: {expected}")

    try:
        answer = pipeline.ask(question)

        print("\n=== ANSWER ===")
        print(answer)

        passed = classify_result(
            answer,
            expected,
        )

        if passed:

            if expected == "FALLBACK":
                print(
                    "\n[PASS] RAG correctly rejected "
                    "unsupported context."
                )

            elif expected == "CODE_ANSWER":
                print(
                    "\n[PASS] RAG answered the "
                    "code-related question."
                )

            else:
                print(
                    "\n[PASS] RAG answered using "
                    "lesson context."
                )

        else:
            print(
                "\n[FAIL] RAG returned an unexpected answer."
            )

        return passed

    except Exception as error:

        print(
            f"\n[ERROR] Test raised an exception:"
        )
        print(error)

        return False


def main():

    print("=" * 70)
    print("RAG PIPELINE TEST SUITE")
    print("=" * 70)

    retriever = LessonRetriever()

    pipeline = RAGPipeline(
        retriever
    )

    passed = 0
    failed = 0

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        result = run_test(
            index,
            test_case,
            pipeline,
        )

        if result:
            passed += 1
        else:
            failed += 1

    total = len(TEST_CASES)

    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Total  : {total}")

    print("=" * 70)

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()