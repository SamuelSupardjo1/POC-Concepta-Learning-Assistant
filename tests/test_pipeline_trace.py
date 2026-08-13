from src.loader import LessonLoader
from src.content_segmenter import ContentSegmenter
from src.content_classifier import ContentClassifier
from src.theory_extractor import TheoryExtractor
from src.structure_identifier import StructureIdentifier
from src.structure_aware_chunker import StructureAwareChunker


TARGETS = [
    "transform",
    "relative url",
    "hyperlink",
    "anchor link",
]


def contains_target(text):

    text_lower = text.lower()

    return any(
        target in text_lower
        for target in TARGETS
    )


def print_matches(stage, items):

    matches = []

    for item in items:

        if isinstance(item, dict):

            content = item.get(
                "content",
                "",
            )

        else:

            content = getattr(
                item,
                "page_content",
                "",
            )

        if contains_target(content):

            matches.append(content)

    print("\n" + "-" * 70)
    print(stage)
    print("-" * 70)

    if not matches:

        print("NO MATCH")

        return

    for index, content in enumerate(
        matches,
        start=1,
    ):

        print(f"\nMATCH {index}")
        print(content)


def main():

    print("=" * 70)
    print("CONCEPTA - PIPELINE TRACE TEST")
    print("=" * 70)

    # ==================================================
    # 1. LOAD
    # ==================================================

    loader = LessonLoader(
        "knowledge_base/lesson"
    )

    documents = loader.load()

    print(
        f"\nDocuments loaded: {len(documents)}"
    )

    # ==================================================
    # PROCESS
    # ==================================================

    segmenter = ContentSegmenter()
    classifier = ContentClassifier()
    extractor = TheoryExtractor()
    identifier = StructureIdentifier()
    chunker = StructureAwareChunker()

    all_blocks = []
    all_classified = []
    all_theory = []
    all_structured = []
    all_chunks = []

    for document in documents:

        blocks = segmenter.segment(
            document.page_content
        )

        all_blocks.extend(blocks)

        classified = []

        for block in blocks:

            content_type = classifier.classify(
                block
            )

            classified.append(
                {
                    "content": block,
                    "content_type": content_type,
                }
            )

        all_classified.extend(
            classified
        )

        theory = extractor.extract(
            blocks
        )

        all_theory.extend(
            theory
        )

        structured = identifier.identify(
            theory
        )

        all_structured.extend(
            structured
        )

        chunks = chunker.chunk(
            structured
        )

        all_chunks.extend(
            chunks
        )

    # ==================================================
    # TRACE
    # ==================================================

    print_matches(
        "STAGE 1 - SEGMENTATION",
        all_blocks,
    )

    print_matches(
        "STAGE 2 - CLASSIFICATION",
        all_classified,
    )

    print_matches(
        "STAGE 3 - THEORY EXTRACTION",
        all_theory,
    )

    print("\n" + "-" * 70)
    print("STAGE 4 - STRUCTURE IDENTIFICATION")
    print("-" * 70)

    for item in all_structured:

        content = item.get(
            "content",
            "",
        )

        if contains_target(content):

            print("\nCONTENT:")
            print(content)

            print("\nCONTENT TYPE:")
            print(
                item.get("content_type")
            )

            print("\nLESSON:")
            print(
                item.get("lesson")
            )

            print("\nSECTION:")
            print(
                item.get("section")
            )

    print_matches(
        "STAGE 5 - CHUNKING",
        all_chunks,
    )

    print("\n" + "=" * 70)
    print("PIPELINE TRACE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()