from src.content_segmenter import ContentSegmenter
from src.content_classifier import ContentClassifier


def main():

    text = """
    Pertemuan 6 - Link in HTML and About Us Section

    Hyperlink

    <a href="https://www.amazon.com/">
    Lihat Amazon
    </a>

    Codes

    Relative URL adalah alamat untuk menuju
    ke halaman yang ada di dalam website yang sama.

    <a href="about.html">
    Lihat About Us
    </a>

    Codes

    Anchor Link adalah cara untuk menautkan link
    ke bagian tertentu di halaman website yang sama.

    1. Buat project baru pada Replit
    """

    segmenter = ContentSegmenter()
    classifier = ContentClassifier()

    blocks = segmenter.segment(text)

    print("=" * 70)
    print("SEGMENTATION + CONTENT CLASSIFICATION TEST")
    print("=" * 70)

    print(f"\nTotal blocks: {len(blocks)}")

    counts = {
        "theory": 0,
        "code": 0,
        "activity": 0,
        "noise": 0,
        "structure": 0,
    }

    for index, block in enumerate(blocks, start=1):

        content_type = classifier.classify(block)

        counts[content_type.value] += 1

        print("\n" + "-" * 70)
        print(f"BLOCK {index}")
        print("-" * 70)

        print("Content:")
        print(block)

        print(
            f"\nClassification: "
            f"{content_type.value.upper()}"
        )

    print("\n" + "=" * 70)
    print("REAL PDF EDGE CASE TEST")
    print("=" * 70)

    edge_cases = [
        (
            "Table of Contents",
            "noise",
        ),
        (
            "Source: https://www.youtube.com/example",
            "noise",
        ),
        (
            "Timedoor Coding Academy",
            "noise",
        ),
        (
            "355",
            "noise",
        ),
    ]

    for index, (content, expected) in enumerate(
        edge_cases,
        start=1,
    ):

        predicted = classifier.classify(content)

        status = (
            "PASS"
            if predicted.value == expected
            else "FAIL"
        )

        print("\n" + "-" * 70)
        print(f"EDGE CASE {index}")
        print("-" * 70)

        print("Content:")
        print(content)

        print(f"\nExpected : {expected}")
        print(f"Predicted: {predicted.value}")
        print(f"Status   : {status}")
        
    print("\n" + "=" * 70)
    print("GENERAL PDF ROBUSTNESS TEST")
    print("=" * 70)

    general_cases = [
        ("TABLE OF CONTENTS", "noise"),
        ("Table of Contents ................................ 1", "noise"),
        ("Contents", "noise"),
        ("Source: https://example.com", "noise"),
        ("https://example.com", "noise"),
        ("www.example.com", "noise"),
        ("Page 1", "noise"),
        ("1", "noise"),

        ("Introduction", "structure"),
        ("Variables", "structure"),
        ("Conditional Statement", "structure"),
        ("What is a variable?", "theory"),
        (
            "A variable is a container used to store data.",
            "theory",
        ),

        (
            "1. Create a new variable",
            "activity",
        ),
        (
            "Buatlah program sederhana",
            "activity",
        ),

        (
            "const score = 80;",
            "code",
        ),
        (
            "<h1>Hello World</h1>",
            "code",
        ),
    ]

    passed = 0

    for index, (content, expected) in enumerate(
        general_cases,
        start=1,
    ):

        predicted = classifier.classify(content)

        status = (
            "PASS"
            if predicted.value == expected
            else "FAIL"
        )

        if status == "PASS":
            passed += 1

        print("\n" + "-" * 70)
        print(f"GENERAL TEST {index}")
        print("-" * 70)

        print("Content:")
        print(content)

        print(f"\nExpected : {expected}")
        print(f"Predicted: {predicted.value}")
        print(f"Status   : {status}")

    print("\n" + "=" * 70)
    print(
        f"GENERAL TEST RESULT: "
        f"{passed}/{len(general_cases)} passed"
    )
    print("=" * 70)

if __name__ == "__main__":
    main()