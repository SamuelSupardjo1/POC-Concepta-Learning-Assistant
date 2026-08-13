from src.content_segmenter import ContentSegmenter
from src.theory_extractor import TheoryExtractor
from src.structure_identifier import StructureIdentifier


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
    identifier = StructureIdentifier()

    blocks = segmenter.segment(text)

    extracted = extractor.extract(blocks)

    structured = identifier.identify(extracted)

    print("=" * 70)
    print("STRUCTURE IDENTIFICATION TEST")
    print("=" * 70)

    for index, item in enumerate(
        structured,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print(f"Type    : {item['content_type']}")
        print(f"Lesson  : {item['lesson']}")
        print(f"Section : {item['section']}")
        print(f"Content : {item['content']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()