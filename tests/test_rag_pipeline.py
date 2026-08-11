from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from src.retriever import LessonRetriever
from src.rag_pipeline import RAGPipeline

print("=" * 70)
print("TEST CASE 3 - RAG PIPELINE")
print("=" * 70)

embedding = EmbeddingModel().get_model()

vectordb = LessonVectorDB(embedding)

retriever = LessonRetriever(vectordb.get_db())

pipeline = RAGPipeline(retriever)

question = "What is HTML?"

print(f"\nQuestion : {question}")

answer = pipeline.ask(question)

print("\nAnswer\n")

print(answer)

print("\n")

print("=" * 70)
print("STATUS : SUCCESS")
print("=" * 70)