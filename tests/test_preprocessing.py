from src.loader import LessonLoader
from src.preprocessor import TheoryPreprocessor


def main():

    print("=" * 70)
    print("REAL PDF - THEORY PREPROCESSING")
    print("=" * 70)

    loader = LessonLoader("knowledge_base/lesson")

    documents = loader.load()

    print(f"\nOriginal documents : {len(documents)}")

    preprocessor = TheoryPreprocessor()

    processed_documents = preprocessor.process(
        documents
    )

    print(
        f"Processed documents: "
        f"{len(processed_documents)}"
    )

    # Show selected pages
    for index in [75, 76, 77, 82, 83, 84, 85]:

        if index >= len(processed_documents):
            continue

        document = processed_documents[index]

        print("\n" + "=" * 70)
        print(
            f"DOCUMENT INDEX: {index}"
        )
        print("=" * 70)

        print("\nMetadata:")
        print(document.metadata)

        print("\nProcessed content:")
        print(document.page_content[:3000])


if __name__ == "__main__":
    main()