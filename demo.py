from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from src.retriever import LessonRetriever
from src.rag_pipeline import RAGPipeline


def main():

    print("=" * 60)
    print("Concepta Learning Assistant - Proof of Concept")
    print("=" * 60)

    # Load embedding model
    embedding = EmbeddingModel().get_model()

    # Open existing ChromaDB
    vectordb = LessonVectorDB(embedding)

    # Create retriever
    retriever = LessonRetriever(vectordb.get_db())

    # Create pipeline
    pipeline = RAGPipeline(retriever)

    while True:

        question = input("\nQuestion (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer = pipeline.ask(question)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()