import re

from langchain_core.documents import Document


class MetadataExtractor:
    """
    Extract lesson metadata from each document.
    """

    lesson_pattern = re.compile(r"PERTEMUAN\s+(\d+)", re.IGNORECASE)

    def enrich(
        self,
        documents: list[Document],
    ) -> list[Document]:

        current_lesson = None
        current_title = None

        for doc in documents:

            text = doc.page_content

            lesson = self.lesson_pattern.search(text)

            if lesson:
                current_lesson = lesson.group(1)

                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                if len(lines) >= 3:
                    current_title = lines[2]

            doc.metadata["lesson"] = current_lesson
            doc.metadata["title"] = current_title

        return documents