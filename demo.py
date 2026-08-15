import time

from src.config import MODEL_NAME
from src.retriever import LessonRetriever
from src.prompt import PromptBuilder
from src.llm import OllamaLLM
from src.rag_pipeline import RAGPipeline


def print_header():
    print("=" * 70)
    print("INTELLIGENT LEARNING ASSISTANT")
    print("RAG END-TO-END DEMO")
    print("=" * 70)
    print()
    print(f"Model: {MODEL_NAME}")
    print("Knowledge Base: Lesson PDF")
    print("Vector Database: ChromaDB")
    print("=" * 70)
    print()


def create_pipeline():
    """
    Initialize the complete RAG pipeline.
    """

    print("Initializing RAG pipeline...")
    print()

    retriever = LessonRetriever()

    prompt_builder = PromptBuilder()

    llm = OllamaLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    print("RAG pipeline initialized successfully.")
    print()

    return pipeline


def ask_question(
    pipeline: RAGPipeline,
    question: str,
):
    """
    Ask one question and display the result.
    """

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)
    print()

    start_time = time.perf_counter()

    try:
        answer = pipeline.ask(question)

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        print()
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)
        print()

        print(
            f"Response Time: "
            f"{elapsed_time:.2f} seconds"
        )

        print("=" * 70)
        print()

        return answer

    except Exception as error:

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(
            f"{type(error).__name__}: {error}"
        )
        print()
        print(
            f"Elapsed Time Before Error: "
            f"{elapsed_time:.2f} seconds"
        )
        print("=" * 70)
        print()

        return None


def run_manual_demo(
    pipeline: RAGPipeline,
):
    """
    Run predefined questions for the POC demonstration.
    """

    questions = [

        # --------------------------------------------------------
        # Relevant lesson question
        # --------------------------------------------------------

        "Apa itu HTML?",

        # --------------------------------------------------------
        # Relevant attribute question
        # --------------------------------------------------------

        "Apa kegunaan atribut novalidate?",

        # --------------------------------------------------------
        # Relevant hyperlink question
        # --------------------------------------------------------

        "Apa itu hyperlink?",

        # --------------------------------------------------------
        # Relevant header question
        # --------------------------------------------------------

        "Apa kegunaan header?",

        # --------------------------------------------------------
        # Unsupported programming topic
        # --------------------------------------------------------

        "Apa itu Python?",
    ]

    print("=" * 70)
    print("PREDEFINED END-TO-END DEMO")
    print("=" * 70)
    print()
    print(
        f"Total questions: {len(questions)}"
    )
    print()

    results = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print()
        print(
            f"[DEMO {index}/{len(questions)}]"
        )

        start_time = time.perf_counter()

        try:

            answer = pipeline.ask(
                question
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "response_time": elapsed_time,
                    "status": "PASS",
                }
            )

        except Exception as error:

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            results.append(
                {
                    "question": question,
                    "answer": str(error),
                    "response_time": elapsed_time,
                    "status": "ERROR",
                }
            )

    print()
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"[{result['status']}] "
            f"Question {index}: "
            f"{result['question']}"
        )

        print(
            f"Answer: "
            f"{result['answer']}"
        )

        print(
            f"Response Time: "
            f"{result['response_time']:.2f} seconds"
        )

    print()
    print("=" * 70)
    print("END OF DEMO")
    print("=" * 70)
    print()


def run_interactive_demo(
    pipeline: RAGPipeline,
):
    """
    Interactive mode for manually asking questions.
    """

    print("=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print()
    print(
        "Type your programming question."
    )
    print(
        "Type 'exit' or 'quit' to stop."
    )
    print()

    while True:

        try:
            question = input(
                "Student > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print()
            print(
                "Exiting interactive mode."
            )
            break

        if question.lower() in {
            "exit",
            "quit",
        }:
            print(
                "Exiting interactive mode."
            )
            break

        if not question:
            print(
                "Please enter a question."
            )
            print()
            continue

        print()

        ask_question(
            pipeline,
            question,
        )


def main():

    print_header()

    try:

        pipeline = create_pipeline()

    except Exception as error:

        print("=" * 70)
        print("INITIALIZATION ERROR")
        print("=" * 70)
        print(
            f"{type(error).__name__}: {error}"
        )
        print("=" * 70)

        return

    # ------------------------------------------------------------
    # Run predefined demonstration
    # ------------------------------------------------------------

    run_manual_demo(
        pipeline
    )

    # ------------------------------------------------------------
    # Continue with interactive mode
    # ------------------------------------------------------------

    run_interactive_demo(
        pipeline
    )


if __name__ == "__main__":
    main()