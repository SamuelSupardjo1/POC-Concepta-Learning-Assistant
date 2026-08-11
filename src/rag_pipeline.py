from src.prompt import PromptBuilder
from src.retriever import LessonRetriever
from src.llm import ask_llm


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: LessonRetriever,
    ) -> None:

        self.retriever = retriever
        self.prompt_builder = PromptBuilder()

    def ask(self, question: str) -> str:
        """
        Execute the complete RAG workflow.
        """

        documents = self.retriever.search(question)

        prompt = self.prompt_builder.build(
            question=question,
            contexts=documents,
        )

        answer = ask_llm(prompt)

        return answer