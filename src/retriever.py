import re

from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB


class LessonRetriever:
    """
    Retrieve relevant theory chunks from ChromaDB
    based on semantic similarity and keyword relevance.
    """

    def __init__(
        self,
        top_k: int = 3,
        distance_threshold: float = 0.30,
    ) -> None:

        self.top_k = top_k
        self.distance_threshold = distance_threshold

        self.embedding = EmbeddingModel().get_model()

        self.vectordb = LessonVectorDB(
            self.embedding
        )

    def _extract_keywords(self, query: str) -> set[str]:
        """
        Extract meaningful keywords from the question.
        """

        stopwords = {
            "apa",
            "itu",
            "yang",
            "dan",
            "di",
            "ke",
            "dari",
            "untuk",
            "dalam",
            "pada",
            "dengan",
            "adalah",
            "fungsi",
            "kegunaan",
            "cara",
            "bagaimana",
            "the",
            "is",
            "what",
            "how",
            "of",
            "in",
            "a",
            "an",
        }

        words = re.findall(
            r"[a-zA-Z0-9_]+",
            query.lower(),
        )

        return {
            word
            for word in words
            if word not in stopwords
            and len(word) >= 3
        }

    def _is_keyword_relevant(
        self,
        query: str,
        content: str,
    ) -> bool:
        """
        Check whether the retrieved content shares meaningful
        keywords with the student's question.
        """

        query_keywords = self._extract_keywords(query)

        content_keywords = set(
            re.findall(
                r"[a-zA-Z0-9_]+",
                content.lower(),
            )
        )

        if not query_keywords:
            return False

        matched_keywords = (
            query_keywords & content_keywords
        )

        return len(matched_keywords) > 0

    def retrieve(self, query: str):

        if not query or not query.strip():
            return []

        results = self.vectordb.get_db().similarity_search_with_score(
            query,
            k=self.top_k,
        )

        print("\n=== RETRIEVAL DEBUG ===")

        relevant_documents = []

        for document, score in results:

            distance_accepted = (
                score <= self.distance_threshold
            )

            keyword_accepted = (
                self._is_keyword_relevant(
                    query,
                    document.page_content,
                )
            )

            accepted = (
                distance_accepted
                and keyword_accepted
            )

            print(f"Score: {score}")
            print(
                f"Distance accepted: "
                f"{distance_accepted}"
            )
            print(
                f"Keyword accepted: "
                f"{keyword_accepted}"
            )
            print(
                f"Accepted: {accepted}"
            )
            print(
                f"Content: "
                f"{document.page_content[:100]}"
            )
            print(
                f"Metadata: "
                f"{document.metadata}"
            )
            print()

            if accepted:
                relevant_documents.append(
                    document
                )

        print(
            f"Distance threshold: "
            f"{self.distance_threshold}"
        )

        print(
            f"Relevant documents: "
            f"{len(relevant_documents)}"
        )

        return relevant_documents