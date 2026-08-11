from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class LessonLoader:
    """
    Load lesson PDF files from the knowledge base.
    """

    def __init__(self, lesson_folder: str):
        self.lesson_folder = Path(lesson_folder)

    def load(self) -> list[Document]:
        """
        Load all PDF files inside lesson folder.

        Returns:
            list[Document]
        """
        documents: list[Document] = []

        for pdf_file in sorted(self.lesson_folder.glob("*.pdf")):
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        return documents