from langchain_core.documents import Document

from src.preprocessor import TheoryPreprocessor


def main():

    documents = [
        Document(
            page_content="""
Lesson 1

What is HTML?

HTML stands for HyperText Markup Language.
It is used to structure content on web pages.

<html>
<body>
Hello World
</body>
</html>

HTML documents consist of elements.
""",
            metadata={
                "lesson": "Lesson 1",
                "page": 1,
            },
        )
    ]

    preprocessor = TheoryPreprocessor()

    processed = preprocessor.process(documents)

    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)

    print("\n===== ORIGINAL =====")
    print(documents[0].page_content)

    print("\n===== PROCESSED =====")
    print(processed[0].page_content)

    print("\n===== METADATA =====")
    print(processed[0].metadata)


if __name__ == "__main__":
    main()