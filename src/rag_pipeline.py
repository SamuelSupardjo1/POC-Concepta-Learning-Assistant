from src.prompt import PromptBuilder
from src.retriever import LessonRetriever
from src.llm import ask_llm


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    FALLBACK_ANSWER = (
        "The requested information is not available in the lesson."
    )

    def __init__(
        self,
        retriever: LessonRetriever,
    ) -> None:

        self.retriever = retriever
        self.prompt_builder = PromptBuilder()

    def _has_answerable_context(self, documents) -> bool:
        """
        Check whether retrieved documents contain explanatory statements
        that can support an answer.
        """

        if not documents:
            return False

        for document in documents:
            content = document.page_content.strip()

            if not content:
                continue

            # Skip chunks that are only questions.
            if content.endswith("?"):
                continue

            # Skip very short headings or labels.
            if len(content.split()) < 3:
                continue

            # A declarative statement is considered answerable.
            return True

        return False

    def ask(self, question: str) -> str:

        documents = self.retriever.retrieve(question)

        print(
            f"\nDocuments passed to LLM: {len(documents)}"
        )

        if not documents:
            print("No relevant context found.")
            return self.FALLBACK_ANSWER

        # Check whether retrieved context actually contains
        # enough information to support an answer.
        if not self._has_answerable_context(documents):
            print("Retrieved context is not answerable.")
            return self.FALLBACK_ANSWER

        prompt = self.prompt_builder.build(
            question=question,
            contexts=documents,
        )

        print("\n=== GENERATED PROMPT ===")
        print(prompt)
        print("========================")

        answer = ask_llm(prompt)

        return answer