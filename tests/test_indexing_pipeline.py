from src.indexing_pipeline import IndexingPipeline


def main():

    pipeline = IndexingPipeline(
        "knowledge_base/lesson"
    )

    documents = pipeline.run()

    print("\n" + "=" * 70)
    print("INDEXING RESULT FROM TEST")
    print("=" * 70)

    print(
        f"Indexed chunks: {len(documents)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()