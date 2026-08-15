import re


class AnswerValidator:
    """
    Validate LLM answers against the retrieved lesson context.
    """

    FALLBACK = (
        "The requested information is not available in the lesson."
    )

    def validate(
        self,
        question: str,
        answer: str,
        contexts: list,
    ) -> str:

        if not answer or not answer.strip():
            return self.FALLBACK

        if not contexts:
            return self.FALLBACK

        context_text = " ".join(
            doc.page_content
            for doc in contexts
        )

        # Normalize text for comparison
        normalized_context = self._normalize(context_text)
        normalized_answer = self._normalize(answer)

        # Remove common unsupported introductory phrases
        normalized_answer = self._remove_common_phrases(
            normalized_answer
        )

        # Check whether the answer contains information
        # that cannot reasonably be found in the lesson context.
        if self._contains_unsupported_information(
            normalized_answer,
            normalized_context,
        ):
            return self.FALLBACK

        return answer.strip()

    def _normalize(self, text: str) -> str:
        """
        Normalize text for simple lexical comparison.
        """

        text = text.lower()

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _remove_common_phrases(
        self,
        text: str,
    ) -> str:

        phrases = [
            "the novalidate attribute is used to",
            "the purpose of the novalidate attribute is to",
            "novalidate is used to",
            "atribut novalidate digunakan untuk",
        ]

        for phrase in phrases:
            text = text.replace(
                phrase,
                "",
            )

        return text.strip()

    def _contains_unsupported_information(
        self,
        answer: str,
        context: str,
    ) -> bool:

        answer_words = set(
            word
            for word in answer.split()
            if len(word) >= 4
        )

        context_words = set(
            word
            for word in context.split()
            if len(word) >= 4
        )

        if not answer_words:
            return False

        overlap = (
            answer_words & context_words
        )

        similarity = (
            len(overlap)
            / len(answer_words)
        )

        return similarity < 0.35