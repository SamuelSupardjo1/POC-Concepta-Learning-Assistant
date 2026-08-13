from src.content_segmenter import ContentSegmenter
from src.theory_extractor import TheoryExtractor
from src.structure_identifier import StructureIdentifier
from src.structure_aware_chunker import StructureAwareChunker


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

    # ==============================================
    # 1. Segmentation
    # ==============================================

    segmenter = ContentSegmenter()

    blocks = segmenter.segment(text)

    # ==============================================
    # 2. Theory extraction
    # ==============================================

    extractor = TheoryExtractor()

    extracted = extractor.extract(blocks)

    # ==============================================
    # 3. Structure identification
    # ==============================================

    identifier = StructureIdentifier()

    structured = identifier.identify(extracted)

    # ==============================================
    # 4. Structure-aware chunking
    # ==============================================

    chunker = StructureAwareChunker()

    chunks = chunker.chunk(structured)

    # ==============================================
    # Output
    # ==============================================

    print("=" * 70)
    print("STRUCTURE-AWARE CHUNKING TEST")
    print("=" * 70)

    print(f"\nStructured blocks : {len(structured)}")
    print(f"Final chunks      : {len(chunks)}")

    for chunk in chunks:

        print("\n" + "-" * 70)
        print(
            f"CHUNK {chunk['chunk_id']}"
        )
        print("-" * 70)

        print("Metadata:")

        for key, value in chunk["metadata"].items():
            print(f"  {key}: {value}")

        print("\nContent:")
        print(chunk["content"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()