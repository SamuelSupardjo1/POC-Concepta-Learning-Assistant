from src.content_segmenter import ContentSegmenter
from src.theory_extractor import TheoryExtractor


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
    extractor = TheoryExtractor()

    blocks = segmenter.segment(text)

    results = extractor.extract(blocks)

    print("=" * 70)
    print("THEORY EXTRACTION TEST")
    print("=" * 70)

    print(f"\nOriginal blocks : {len(blocks)}")
    print(f"Theory blocks   : {len(results)}")

    for index, item in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print(
            f"Type: {item['content_type']}"
        )

        print(
            f"Content:\n{item['content']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()