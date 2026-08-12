import re
from enum import Enum

from langchain_core.documents import Document


class ContentType(str, Enum):
    THEORY = "theory"
    CODE = "code"
    ACTIVITY = "activity"
    NOISE = "noise"
    STRUCTURE = "structure"


class ContentClassifier:
    """
    Classify segmented document blocks into content categories.

    Categories:
    - THEORY
    - CODE
    - ACTIVITY
    - NOISE
    - STRUCTURE
    """

    # --------------------------------------------------
    # Explicit markers found in the lesson PDF
    # --------------------------------------------------

    CODE_MARKERS = {
        "codes",
        "code",
    }

    ACTIVITY_MARKERS = {
        "questions",
        "question",
        "exercise",
        "exercises",
        "latihan",
        "tugas",
        "activity",
        "activities",
    }

    STRUCTURE_MARKERS = {
        "contoh",
        "output",
        "syntax",
    }

    NOISE_PATTERNS = [
        r"^timedoor coding academy$",
        r"^timedoor academy$",
        r"^source\s*:",
        r"^https?://",
        r"^www\.",
    ]

    def classify(
        self,
        block: str,
        previous_blocks: list[str] | None = None,
    ) -> ContentType:

        text = block.strip()

        if not text:
            return ContentType.NOISE

        normalized = self.normalize(text)

        # --------------------------------------------------
        # 1. Explicit NOISE
        # --------------------------------------------------

        if self.is_noise(text):
            return ContentType.NOISE

        # --------------------------------------------------
        # 2. Explicit CODE
        # --------------------------------------------------

        if self.is_code(text):
            return ContentType.CODE

        # --------------------------------------------------
        # 3. Explicit activity / question
        # --------------------------------------------------

        if self.is_activity(text):
            return ContentType.ACTIVITY

        # --------------------------------------------------
        # 4. Lesson / heading structure
        # --------------------------------------------------

        if self.is_structure(text):
            return ContentType.STRUCTURE

        # --------------------------------------------------
        # 5. Default
        # --------------------------------------------------

        return ContentType.THEORY

    # ======================================================
    # CLASSIFICATION RULES
    # ======================================================

    def is_noise(self, text: str) -> bool:

        normalized = self.normalize(text)

        # Explicit marker
        if normalized in self.CODE_MARKERS:
            return True

        if normalized in self.ACTIVITY_MARKERS:
            return True

        # Page number
        if re.fullmatch(r"\d+", normalized):
            return True

        # Roman page number
        if re.fullmatch(r"[ivxlcdm]+", normalized):
            return True

        # URL / source / repeated header
        for pattern in self.NOISE_PATTERNS:

            if re.fullmatch(
                pattern,
                normalized,
                re.IGNORECASE,
            ):
                return True

        return False

    def is_code(self, text: str) -> bool:

        normalized = self.normalize(text)

        # --------------------------------------------------
        # HTML
        # --------------------------------------------------

        if re.search(
            r"<\s*/?\s*[a-zA-Z][^>]*>",
            text,
        ):
            return True

        # --------------------------------------------------
        # CSS
        # --------------------------------------------------

        if re.search(
            r"[.#]?[a-zA-Z][\w-]*\s*\{[^}]*\}",
            text,
            re.DOTALL,
        ):
            return True

        # --------------------------------------------------
        # JavaScript / programming syntax
        # --------------------------------------------------

        if re.search(
            r"\b(function|const|let|var|return|if|else|for|while)\b",
            normalized,
        ):
            return True

        # Common programming symbols
        if (
            ("{" in text and "}" in text)
            or (";" in text and len(text.split()) < 80)
        ):
            return True

        return False

    def is_activity(self, text: str) -> bool:

        normalized = self.normalize(text)

        # Explicit activity marker
        if normalized in self.ACTIVITY_MARKERS:
            return True

        # "Yuk kita coba"
        if re.search(
            r"\b(yuk|ayo)\s+(kita\s+)?coba\b",
            normalized,
        ):
            return True

        # Common activity instructions
        activity_patterns = [
            r"^buatlah\b",
            r"^buat\s",
            r"^tambahkan\b",
            r"^carilah\b",
            r"^cari\s",
            r"^buka\s",
            r"^buat\s+project\b",
            r"^kerjakan\b",
            r"^gunakan\b",
            r"^lakukan\b",
        ]

        for pattern in activity_patterns:

            if re.search(
                pattern,
                normalized,
            ):
                return True

        # Numbered instructions
        if re.match(
            r"^\d+[\.\)]\s+",
            normalized,
        ):
            instruction_words = [
                "buat",
                "tambahkan",
                "carilah",
                "cari",
                "buka",
                "gunakan",
                "kerjakan",
                "lakukan",
            ]

            if any(
                word in normalized
                for word in instruction_words
            ):
                return True

        return False

    def is_structure(self, text: str) -> bool:

        normalized = self.normalize(text)

        # --------------------------------------------------
        # Lesson title
        # --------------------------------------------------

        if re.search(
            r"^pertemuan\s+\d+",
            normalized,
        ):
            return True

        # --------------------------------------------------
        # Explicit structural markers
        # --------------------------------------------------

        if normalized in self.STRUCTURE_MARKERS:
            return True

        # --------------------------------------------------
        # Short title-like text
        #
        # Conservative rule:
        # - <= 8 words
        # - no sentence punctuation
        # --------------------------------------------------

        words = normalized.split()

        if (
            1 <= len(words) <= 8
            and not normalized.endswith(".")
            and not normalized.endswith("?")
            and not normalized.endswith("!")
            and not normalized.endswith(":")
        ):
            return True

        return False

    # ======================================================
    # HELPERS
    # ======================================================

    def normalize(self, text: str) -> str:

        text = text.replace("\u200b", " ")
        text = text.replace("\xa0", " ")

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip().lower()


class ClassifiedBlock:
    """
    Represents a block together with its classification.
    """

    def __init__(
        self,
        content: str,
        content_type: ContentType,
    ):
        self.content = content
        self.content_type = content_type

    def __repr__(self) -> str:

        return (
            f"ClassifiedBlock("
            f"type={self.content_type.value}, "
            f"content={self.content[:50]!r}"
            f")"
        )