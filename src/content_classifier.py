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

    NOISE_EXACT = {
        "timedoor coding academy",
        "timedoor academy",
        "table of contents",
        "contents",
    }

    NOISE_PATTERNS = [
        r"^source\s*:",
        r"^https?://",
        r"^www\.",
        r"^page\s+\d+",
        r"^table\s+of\s+contents\b",
    ]

    def classify(self, text: str) -> ContentType:

        if not text or not text.strip():
            return ContentType.NOISE

        text = text.strip()

        # ==================================================
        # 1. NOISE
        # ==================================================

        if self.is_noise(text):
            return ContentType.NOISE

        # ==================================================
        # 2. CODE
        # ==================================================

        if self.is_code(text):
            return ContentType.CODE

        # ==================================================
        # 3. ACTIVITY
        # ==================================================

        if self.is_activity(text):
            return ContentType.ACTIVITY

        # ==================================================
        # 4. STRUCTURE
        # ==================================================

        if self.is_structure(text):
            return ContentType.STRUCTURE

        # ==================================================
        # 5. DEFAULT
        # ==================================================

        return ContentType.THEORY

    def is_noise(self, text: str) -> bool:

        normalized = self.normalize(text)

        # --------------------------------------------------
        # Exact noise markers
        # --------------------------------------------------

        if normalized in self.NOISE_EXACT:
            return True

        # --------------------------------------------------
        # Code labels
        #
        # "Codes" is a label in the PDF, not actual code.
        # --------------------------------------------------

        if normalized in self.CODE_MARKERS:
            return True

        # --------------------------------------------------
        # Activity labels
        # --------------------------------------------------

        if normalized in self.ACTIVITY_MARKERS:
            return True

        # --------------------------------------------------
        # Page number
        # --------------------------------------------------

        if re.fullmatch(r"\d+", normalized):
            return True

        # --------------------------------------------------
        # Roman page number
        # --------------------------------------------------

        if re.fullmatch(r"[ivxlcdm]+", normalized):
            return True

        # --------------------------------------------------
        # URL / source / repeated header
        #
        # IMPORTANT:
        # use re.match(), not re.fullmatch()
        # because "Source:" is followed by a URL.
        # --------------------------------------------------

        for pattern in self.NOISE_PATTERNS:

            if re.match(
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

        # --------------------------------------------------
        # Common programming symbols
        # --------------------------------------------------

        if (
            ("{" in text and "}" in text)
            or (
                ";" in text
                and len(text.split()) < 80
            )
        ):
            return True

        return False

    def is_activity(self, text: str) -> bool:

        normalized = self.normalize(text)

        # --------------------------------------------------
        # Explicit activity marker
        # --------------------------------------------------

        if normalized in self.ACTIVITY_MARKERS:
            return True

        # --------------------------------------------------
        # "Yuk kita coba"
        # --------------------------------------------------

        if re.search(
            r"\b(yuk|ayo)\s+(kita\s+)?coba\b",
            normalized,
        ):
            return True

        # --------------------------------------------------
        # Common activity instructions
        # --------------------------------------------------

        activity_patterns = [
            # Indonesian
            r"^buatlah\b",
            r"^buat\s",
            r"^tambahkan\b",
            r"^carilah\b",
            r"^cari\s",
            r"^buka\s",
            r"^kerjakan\b",
            r"^gunakan\b",
            r"^lakukan\b",

            # English
            r"^create\s",
            r"^create\b",
            r"^add\s",
            r"^find\s",
            r"^open\s",
            r"^make\s",
            r"^write\s",
            r"^build\s",
            r"^use\s",
            r"^complete\s",
        ]

        for pattern in activity_patterns:

            if re.search(
                pattern,
                normalized,
            ):
                return True

        # --------------------------------------------------
        # Numbered instructions
        # --------------------------------------------------

        if re.match(
            r"^\d+[\.\)]\s+",
            normalized,
        ):

            instruction_words = [
                # Indonesian
                "buat",
                "tambahkan",
                "carilah",
                "cari",
                "buka",
                "gunakan",
                "kerjakan",
                "lakukan",

                # English
                "create",
                "add",
                "find",
                "open",
                "make",
                "write",
                "build",
                "use",
                "complete",
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
        # Only classify short text as structure when it
        # looks like an actual heading.
        # --------------------------------------------------

        words = normalized.split()

        if not (
            1 <= len(words) <= 8
        ):
            return False

        # Sentence punctuation
        if (
            normalized.endswith(".")
            or normalized.endswith("?")
            or normalized.endswith("!")
            or normalized.endswith(":")
        ):
            return False

        # --------------------------------------------------
        # Prevent obvious non-structure content
        # --------------------------------------------------

        if normalized.startswith("source"):
            return False

        if normalized.startswith("http"):
            return False

        if normalized.startswith("www."):
            return False

        if normalized == "table of contents":
            return False

        # --------------------------------------------------
        # A short phrase without sentence punctuation
        # can be treated as a heading.
        # --------------------------------------------------

        return True

    def normalize(self, text: str) -> str:

        text = text.replace(
            "\u200b",
            " ",
        )

        text = text.replace(
            "\xa0",
            " ",
        )

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