import re


class ChunkQualityFilter:
    """
    Final quality filter for knowledge chunks.

    Removes chunks that technically passed the previous
    classification stages but do not contain meaningful
    learning material.
    """

    def filter(self, chunks: list[dict]) -> list[dict]:

        clean_chunks = []

        for chunk in chunks:

            content = chunk["content"].strip()

            if not content:
                continue

            if self.is_low_quality(content):
                continue

            clean_chunks.append(chunk)

        return clean_chunks

    def is_low_quality(self, content: str) -> bool:

        normalized = self.normalize(content)

        # ----------------------------------------------
        # Empty content
        # ----------------------------------------------

        if not normalized:
            return True

        # ----------------------------------------------
        # Only numbers
        # ----------------------------------------------

        if re.fullmatch(r"[\d\s\.\-\(\)]+", normalized):
            return True

        # ----------------------------------------------
        # Very short numbered fragments
        #
        # Example:
        # "Continue Website 1. 2."
        # ----------------------------------------------

        if re.fullmatch(
            r"(continue\s+website\s*)?(\d+[\.\)]?\s*)+",
            normalized,
            re.IGNORECASE,
        ):
            return True

        # ----------------------------------------------
        # Common UI / document fragments
        # ----------------------------------------------

        low_quality_markers = {
            "codes",
            "code",
            "continue website",
            "continue",
        }

        if normalized in low_quality_markers:
            return True

        # ----------------------------------------------
        # Header / footer fragments
        # ----------------------------------------------

        header_footer_markers = {
            "timedoor coding academy",
            "timedoor academy",
        }

        if normalized in header_footer_markers:
            return True

        # ----------------------------------------------
        # URL-only content
        # ----------------------------------------------

        if re.fullmatch(
            r"(https?://\S+|www\.\S+)",
            normalized,
        ):
            return True

        # ----------------------------------------------
        # Source-only content
        # ----------------------------------------------

        if normalized.startswith("source:"):
            return True

        return False

    def normalize(self, text: str) -> str:

        text = text.replace("\u200b", " ")
        text = text.replace("\xa0", " ")

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip().lower()