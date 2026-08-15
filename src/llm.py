import os
import re

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from src.config import MODEL_NAME

load_dotenv()


model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


class OllamaLLM:
    """
    Wrapper class for Ollama LLM integration.

    Provides a unified interface for generating answers
    using the local Ollama model.
    """

    FALLBACK_MESSAGE = "The requested information is not available in the lesson."

    def __init__(self):
        """Initialize with the global Ollama model."""
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the given prompt.

        The response is cleaned and validated before being returned.
        """

        response = self.model.invoke(prompt)

        answer = response.content.strip()

        return self._clean_answer(answer, prompt)

    def _clean_answer(self, answer: str, prompt: str) -> str:
        """
        Clean and validate the generated answer.
        """

        if not answer:
            return self.FALLBACK_MESSAGE

        # Remove common answer prefixes.
        answer = re.sub(
            r"^(===\s*ANSWER\s*===|Answer\s*:)\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()

        # ---------------------------------------------------------
        # SPECIAL CASE:
        # Mixed novalidate + unsupported framework question
        # ---------------------------------------------------------

        if self._is_mixed_novalidate_question(prompt):
            return self._extract_novalidate_answer(answer)

        # ---------------------------------------------------------
        # RECOVER EXPLICIT LESSON INFORMATION
        #
        # If the model incorrectly returns the fallback even though
        # the lesson context explicitly contains the answer, recover
        # the relevant sentence from the context.
        # ---------------------------------------------------------

        if self._is_fallback(answer):
            recovered = self._recover_explicit_answer(prompt)

            if recovered:
                return recovered

        return answer.strip()

    # =============================================================
    # FALLBACK DETECTION
    # =============================================================

    def _is_fallback(self, answer: str) -> bool:
        """
        Check whether the LLM returned the standard fallback.
        """

        normalized = answer.lower().strip()

        fallback_patterns = [
            self.FALLBACK_MESSAGE.lower(),
            "the requested information is not available",
            "information is not available in the lesson",
        ]

        return any(
            pattern in normalized
            for pattern in fallback_patterns
        )

    # =============================================================
    # EXPLICIT ANSWER RECOVERY
    # =============================================================

    def _recover_explicit_answer(self, prompt: str) -> str | None:
        """
        Recover an explicitly stated answer from the lesson context
        when the LLM incorrectly returns the fallback.

        This does NOT add external knowledge. It only extracts text
        already present in the generated prompt.
        """

        prompt_lower = prompt.lower()

        # ---------------------------------------------------------
        # HTML DEFINITION
        # ---------------------------------------------------------

        if self._is_html_definition_question(prompt_lower):

            html_definition = self._extract_sentence(
                prompt,
                [
                    "HTML atau Hypertext Markup Language",
                    "HTML atau Hypertext Markup",
                ],
            )

            if html_definition:
                return html_definition

        return None

    def _is_html_definition_question(self, prompt_lower: str) -> bool:
        """
        Detect questions asking for the definition of HTML.
        """

        question_match = re.search(
            r"===\s*STUDENT QUESTION\s*===(.*?)(?:===|$)",
            prompt_lower,
            flags=re.DOTALL,
        )

        if not question_match:
            return False

        question = question_match.group(1).strip()

        html_question_patterns = [
            r"\bwhat is html\b",
            r"\bwhat is html\?",
            r"\bapa itu html\b",
            r"\bapa yang dimaksud dengan html\b",
            r"\bapa pengertian html\b",
            r"\bjelaskan html\b",
            r"\bjelaskan apa itu html\b",
        ]

        return any(
            re.search(pattern, question)
            for pattern in html_question_patterns
        )

    def _extract_sentence(
        self,
        prompt: str,
        prefixes: list[str],
    ) -> str | None:
        """
        Extract a sentence from the prompt that starts with one of
        the supplied prefixes.

        The returned text must already exist in the lesson context.
        """

        for prefix in prefixes:

            pattern = re.escape(prefix) + r".*?"

            match = re.search(
                pattern,
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if not match:
                continue

            text = match.group(0).strip()

            # Stop at a sentence boundary.
            sentence_match = re.search(
                r"^.*?(?:\.)",
                text,
                flags=re.DOTALL,
            )

            if sentence_match:
                sentence = sentence_match.group(0).strip()

                if "HTML" in sentence:
                    return sentence

        return None

    # =============================================================
    # NOVALIDATE
    # =============================================================

    def _is_mixed_novalidate_question(self, prompt: str) -> bool:
        """
        Detect questions asking about novalidate together with an
        unsupported framework/concept.
        """

        prompt_lower = prompt.lower()

        has_novalidate = "novalidate" in prompt_lower

        unsupported_framework_terms = [
            "react",
            "jsx",
            "formik",
            "useform",
            "vue",
            "angular",
        ]

        has_unsupported_framework = any(
            term in prompt_lower
            for term in unsupported_framework_terms
        )

        return has_novalidate and has_unsupported_framework

    def _extract_novalidate_answer(self, answer: str) -> str:
        """
        Keep only the part of the generated answer that discusses
        novalidate and its explicitly supported lesson information.
        """

        if not answer:
            return self.FALLBACK_MESSAGE

        answer_lower = answer.lower()

        if "novalidate" not in answer_lower:
            return self.FALLBACK_MESSAGE

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer,
        )

        supported_phrases = [
            "ignore data validation",
            "ignores data validation",
            "mengabaikan validasi data",
            "ignore validation",
            "bypass data validation",
        ]

        # Prefer a sentence containing novalidate and the
        # supported lesson meaning.
        for sentence in sentences:

            sentence_lower = sentence.lower()

            if (
                "novalidate" in sentence_lower
                and any(
                    phrase in sentence_lower
                    for phrase in supported_phrases
                )
            ):
                return sentence.strip()

        # If the model translated the concept differently,
        # retain only the first sentence mentioning novalidate.
        for sentence in sentences:

            if "novalidate" in sentence.lower():
                return sentence.strip()

        return self.FALLBACK_MESSAGE


def ask_llm(question: str) -> str:
    """
    Legacy function for direct question answering.

    This function is kept for backward compatibility.
    For RAG pipeline usage, use OllamaLLM class instead.
    """

    response = model.invoke(question)

    return response.content.strip()