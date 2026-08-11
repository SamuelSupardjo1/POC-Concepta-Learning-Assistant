import re

from langchain_core.documents import Document


class TextCleaner:
    """
    Clean document text before chunking.
    """

    def clean(
        self,
        documents: list[Document],
    ) -> list[Document]:

        cleaned_docs = []

        for doc in documents:

            text = doc.page_content

            # Hilangkan multiple newline
            text = re.sub(r"\n{2,}", "\n\n", text)

            # Hilangkan multiple spaces
            text = re.sub(r"[ \t]+", " ", text)

            # Hilangkan spasi di awal/akhir
            text = text.strip()

            doc.page_content = text

            cleaned_docs.append(doc)

        return cleaned_docs