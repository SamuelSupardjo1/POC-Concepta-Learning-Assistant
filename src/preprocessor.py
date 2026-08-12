import re

from langchain_core.documents import Document


class TheoryPreprocessor:
    """
    Preprocess programming lesson documents.

    Goal:
    - Keep programming theory
    - Keep lesson headings
    - Remove code blocks
    - Remove obvious activity/instruction content
    - Remove document noise
    """

    CODE_MARKERS = {
        "codes",
        "code",
    }

    ACTIVITY_MARKERS = {
        "questions",
        "question",
        "exercise",
        "latihan",
        "tugas",
        "activities",
        "activity",
    }

    NOISE_PATTERNS = [
        r"^timedoor coding academy$",
        r"^timedoor academy$",
        r"^source\s*:",
    ]

    def process(self, documents: list[Document]) -> list[Document]:
        processed_documents = []

        for document in documents:

            cleaned_text = self.clean_text(
                document.page_content
            )

            if not cleaned_text.strip():
                continue

            metadata = document.metadata.copy()

            metadata["content_type"] = "theory"

            processed_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=metadata,
                )
            )

        return processed_documents

    def clean_text(self, text: str) -> str:

        # --------------------------------------------------
        # 1. Normalize text
        # --------------------------------------------------

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # --------------------------------------------------
        # 2. Remove fenced code blocks
        # --------------------------------------------------

        text = re.sub(
            r"```[\s\S]*?```",
            "",
            text,
        )

        # --------------------------------------------------
        # 3. Remove obvious source URLs
        # --------------------------------------------------

        text = re.sub(
            r"(?im)^\s*source\s*:\s*https?://\S+\s*$",
            "",
            text,
        )

        # --------------------------------------------------
        # 4. Split document into blocks
        # --------------------------------------------------

        blocks = self.split_into_blocks(text)

        cleaned_blocks = []

        skip_code = False
        skip_activity = False

        for block in blocks:

            normalized = self.normalize_marker(block)

            # ----------------------------------------------
            # CODE MARKER
            # ----------------------------------------------

            if normalized in self.CODE_MARKERS:
                skip_code = True
                continue

            # ----------------------------------------------
            # ACTIVITY MARKER
            # ----------------------------------------------

            if normalized in self.ACTIVITY_MARKERS:
                skip_activity = True
                continue

            # ----------------------------------------------
            # Stop skipping when a new heading appears
            # ----------------------------------------------

            if skip_code:

                if self.is_section_heading(block):
                    skip_code = False
                    cleaned_blocks.append(block.strip())

                continue

            if skip_activity:

                if self.is_section_heading(block):
                    skip_activity = False
                    cleaned_blocks.append(block.strip())

                continue

            # ----------------------------------------------
            # Remove obvious noise
            # ----------------------------------------------

            if self.is_noise(block):
                continue

            cleaned_blocks.append(block.strip())

        # --------------------------------------------------
        # 5. Reconstruct document
        # --------------------------------------------------

        result = "\n\n".join(cleaned_blocks)

        # --------------------------------------------------
        # 6. Normalize whitespace
        # --------------------------------------------------

        result = re.sub(
            r"[ \t]+",
            " ",
            result,
        )

        result = re.sub(
            r"\n{3,}",
            "\n\n",
            result,
        )

        return result.strip()

    # ======================================================
    # BLOCK UTILITIES
    # ======================================================

    def split_into_blocks(self, text: str) -> list[str]:
        """
        Split extracted PDF text into logical blocks.
        """

        return [
            block.strip()
            for block in re.split(
                r"\n\s*\n",
                text,
            )
            if block.strip()
        ]

    def normalize_marker(self, block: str) -> str:
        """
        Normalize a block for marker comparison.
        """

        return re.sub(
            r"\s+",
            " ",
            block.strip().lower(),
        )

    def is_noise(self, block: str) -> bool:
        """
        Detect obvious document noise.
        """

        normalized = self.normalize_marker(block)

        # Page number
        if re.fullmatch(
            r"(page\s*)?\d+",
            normalized,
        ):
            return True

        # Roman page number
        if re.fullmatch(
            r"[ivxlcdm]+",
            normalized,
        ):
            return True

        # Known header/footer
        for pattern in self.NOISE_PATTERNS:

            if re.fullmatch(
                pattern,
                normalized,
                re.IGNORECASE,
            ):
                return True

        return False

    def is_section_heading(self, block: str) -> bool:
        """
        Conservative heading detection.

        This does not try to fully understand the lesson.
        Structure identification will be handled separately.
        """

        normalized = self.normalize_marker(block)

        if not normalized:
            return False

        # Known structural markers
        if normalized in {
            "continue website",
            "font awesome",
            "output",
            "intro to website",
        }:
            return True

        # Short title-like blocks
        words = normalized.split()

        if len(words) <= 8 and not normalized.endswith("."):
            return True

        return False