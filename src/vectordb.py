from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class LessonVectorDB:
    """
    Manage ChromaDB operations for lesson documents.
    """

    def __init__(
        self,
        embedding_model: HuggingFaceEmbeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "lesson_collection",
    ) -> None:

        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> None:
        """Insert lesson chunks into ChromaDB."""

        if not documents:
            return

        self.db.add_documents(
            documents
        )

    def clear(self) -> None:
        """Delete all existing documents."""

        try:
            self.db.delete_collection()
        except Exception:
            pass

        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def count(self) -> int:
        """Return total documents stored."""

        return self.db._collection.count()

    def get_db(self) -> Chroma:
        """Return Chroma instance."""
        return self.db