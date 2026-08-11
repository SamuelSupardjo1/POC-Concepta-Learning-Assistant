from src.loader import LessonLoader
from src.document_filter import DocumentFilter
from src.cleaner import TextCleaner
from src.metadata import MetadataExtractor
from src.chunker import LessonChunker
from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from src.retriever import LessonRetriever
from src.prompt import PromptBuilder
from src.llm import ask_llm
from src.rag_pipeline import RAGPipeline
from src.preprocessor import TheoryPreprocessor

loader = LessonLoader("knowledge_base/lesson")
documents = loader.load()

documents = DocumentFilter().filter(documents)
documents = TextCleaner().clean(documents)

preprocessor = TheoryPreprocessor()
documents = preprocessor.process(documents)

documents = MetadataExtractor().enrich(documents)

chunker = LessonChunker()

chunks = chunker.split(documents)

embedding = EmbeddingModel().get_model()

vectordb = LessonVectorDB(embedding)

vectordb.add_documents(chunks)

retriever = LessonRetriever(
    vectordb=vectordb.get_db(),
    k=3,
)

results = retriever.search(
    "Apa itu HTML?"
)

pipeline = RAGPipeline(retriever)

question = input("Question: ")

answer = pipeline.ask(question)

docs = retriever.search(question)

print("="*80)

for i, doc in enumerate(docs, start=1):

    print(f"RESULT {i}")

    print(doc.metadata)

    print()

    print(doc.page_content[:600])

    print()

    print("="*80)

print(answer)
