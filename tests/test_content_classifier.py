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
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)

    for content_type, count in counts.items():

        print(
            f"{content_type.upper():12}: {count}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()