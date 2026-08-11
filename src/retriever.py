from langchain_chroma import Chroma
from langchain_core.documents import Document


class LessonRetriever:
    """
    Wrapper for semantic retrieval using ChromaDB.
    """

    def __init__(
        self,
        vectordb: Chroma,
        k: int = 3,
    ) -> None:

        self.retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def search(
        self,
        query: str,
    ) -> list[Document]:
        """
        Retrieve the most relevant lesson chunks.
        """

        return self.retriever.invoke(query)