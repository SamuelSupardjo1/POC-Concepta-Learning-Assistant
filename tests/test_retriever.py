from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB


def print_results(query: str, results):

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"RESULT {index}")
        print("-" * 70)

        print(f"Similarity score: {score:.4f}")

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)


def main():

    embedding_model = EmbeddingModel().get_model()

    vectordb = LessonVectorDB(
        embedding_model
    )

    db = vectordb.get_db()

    queries = [
        "Apa itu relative URL?",
        "Apa fungsi anchor link?",
        "Bagaimana cara membuat hyperlink?",
        "Apa fungsi property transform?",
    ]

    for query in queries:

        results = db.similarity_search_with_score(
            query,
            k=5,
        )

        print_results(
            query,
            results,
        )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()