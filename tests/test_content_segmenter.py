from src.content_segmenter import ContentSegmenter


def main():

    segmenter = ContentSegmenter()

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

    blocks = segmenter.segment(text)

    print("=" * 70)
    print("CONTENT SEGMENTATION TEST")
    print("=" * 70)

    print(f"\nTotal blocks: {len(blocks)}")

    for index, block in enumerate(blocks, start=1):

        print("\n" + "-" * 70)
        print(f"BLOCK {index}")
        print("-" * 70)
        print(block)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()