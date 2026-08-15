import re

from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from difflib import SequenceMatcher

class LessonRetriever:
    """
    Retrieve relevant theory chunks from ChromaDB
    based on semantic similarity and keyword relevance.
    """

    def __init__(
        self,
        top_k: int = 3,
        distance_threshold: float = 0.38,
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

        Special-case HTML tag questions such as "<a>" or "anchor link"
        so they still produce a valid concept keyword for retrieval.
        """

        normalized = query.lower().strip()

        if "<a>" in normalized or "tag a" in normalized or "a tag" in normalized:
            return {"a", "anchor", "hyperlink", "link"}

        if "anchor link" in normalized or "anchor" in normalized:
            return {"anchor", "hyperlink", "link"}

        if "header" in normalized:
            return {"header"}

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
            "membuat",
            "menggunakan",
            "buat",
            "pakai",
            "lakukan",
            "tambahkan",
            "tulis",
            "tuliskan",
            "the",
            "is",
            "what",
            "how",
            "of",
            "in",
            "a",
            "an",
            "create",
            "make",
            "use",
            "do",
            "add",
            "write",
        }

        words = re.findall(
            r"[a-zA-Z0-9_]+",
            normalized,
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
        Check whether retrieved content shares meaningful
        keywords with the student's question.

        Small spelling mistakes are allowed.
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

        # Exact keyword match
        matched_keywords = (
            query_keywords & content_keywords
        )

        if matched_keywords:
            return True

        # Fuzzy matching for small typos
        for query_word in query_keywords:

            if len(query_word) < 5:
                continue

            for content_word in content_keywords:

                similarity = SequenceMatcher(
                    None,
                    query_word,
                    content_word,
                ).ratio()

                if similarity >= 0.80:
                    return True

        return False

    def retrieve(self, query: str):

        if not query or not query.strip():
            return []

        normalized = query.lower().strip()
        candidate_queries = [query]

        if "<a>" in normalized or "tag a" in normalized or "a tag" in normalized:
            candidate_queries.extend([
                "Hyperlink <a> href anchor link",
                "Anchor Link Hyperlink HTML link",
                "Tag a href hyperlink",
            ])
        elif "anchor link" in normalized or "anchor" in normalized:
            candidate_queries.extend([
                "Anchor Link Hyperlink HTML link section",
                "Hyperlink <a> href",
            ])
        elif "header" in normalized:
            candidate_queries.extend([
                "header website page section",
                "Header pada website",
                "Header sebagai kepala website",
            ])
        elif "footer" in normalized:
            candidate_queries.extend([
                "footer website page section",
                "Footer sebagai kaki website",
                "Footer pada website",
            ])

        seen_contents = set()
        relevant_documents = []
        all_results = []

        for retrieval_query in candidate_queries:
            results = self.vectordb.get_db().similarity_search_with_score(
                retrieval_query,
                k=10,
            )
            for document, score in results:
                key = (document.page_content.strip(), document.metadata.get("source"))
                if key in seen_contents:
                    continue
                seen_contents.add(key)
                all_results.append((document, score))

        print("\n=== RETRIEVAL DEBUG ===")

        for document, score in all_results:

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

        if not relevant_documents:
            raw_db = self.vectordb.get_db().get(include=["documents", "metadatas"])
            for idx, content in enumerate(raw_db.get("documents", [])):
                text = (content or "").lower()
                if "<a>" in normalized or "tag a" in normalized or "a tag" in normalized:
                    if "hyperlink" in text or "anchor link" in text or "<a" in text:
                        relevant_documents.append(
                            type("FallbackDoc", (), {"page_content": raw_db["documents"][idx], "metadata": raw_db["metadatas"][idx]})()
                        )
                        break
                if "header" in normalized and "header" in text and ("sebagai" in text or "kepala" in text):
                    relevant_documents.append(
                        type("FallbackDoc", (), {"page_content": raw_db["documents"][idx], "metadata": raw_db["metadatas"][idx]})()
                    )
                    break
                if "footer" in normalized and "footer" in text and ("sebagai" in text or "kaki" in text):
                    relevant_documents.append(
                        type("FallbackDoc", (), {"page_content": raw_db["documents"][idx], "metadata": raw_db["metadatas"][idx]})()
                    )
                    break

        print(
            f"Distance threshold: "
            f"{self.distance_threshold}"
        )

        print(
            f"Relevant documents: "
            f"{len(relevant_documents)}"
        )

        return relevant_documents