from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB


def main():

    embedding_model = EmbeddingModel().get_model()

    vectordb = LessonVectorDB(
        embedding_model
    )

    db = vectordb.get_db()

    print("=" * 70)
    print("KEYWORD CHECK")
    print("=" * 70)

    results = db.get()

    documents = results["documents"]
    metadatas = results["metadatas"]

    keywords = [
        "transform",
        "hyperlink",
        "relative url",
        "anchor link",
    ]

    for keyword in keywords:

        print("\n" + "-" * 70)
        print(f"KEYWORD: {keyword}")
        print("-" * 70)

        found = 0

        for content, metadata in zip(
            documents,
            metadatas,
        ):

            if keyword.lower() in content.lower():

                found += 1

                print(f"\nMATCH {found}")
                print("Content:")
                print(content)

                print("\nMetadata:")
                print(metadata)

        if found == 0:
            print("NO MATCH FOUND")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()