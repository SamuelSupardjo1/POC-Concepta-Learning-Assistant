from langchain_core.documents import Document


class DocumentFilter:
    """
    Filter out irrelevant pages before chunking.
    """

    def filter(self, documents: list[Document]) -> list[Document]:
        filtered = []

        for doc in documents:

            text = doc.page_content.strip()

            # Skip blank pages
            if not text:
                continue

            # Skip table of contents
            if "Table of Contents" in text:
                continue

            filtered.append(doc)

        return filtered