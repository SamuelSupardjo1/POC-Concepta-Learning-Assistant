import re


class ContentSegmenter:

    def segment(self, text: str) -> list[str]:

        if not text:
            return []

        lines = self._normalize_lines(text)

        blocks = []
        current = []

        in_code = False

        for line in lines:

            # ==========================================
            # CODE START
            # ==========================================

            if self._looks_like_code_start(line):

                if current:
                    blocks.append(
                        self._build_block(current)
                    )
                    current = []

                current.append(line)
                in_code = True
                continue

            # ==========================================
            # CODE CONTINUATION
            # ==========================================

            if in_code:

                current.append(line)

                if self._looks_like_code_end(line):

                    blocks.append(
                        self._build_block(current)
                    )

                    current = []
                    in_code = False

                continue

            # ==========================================
            # CODE MARKER
            # ==========================================

            if line.strip().lower() in {
                "codes",
                "code",
            }:

                if current:
                    blocks.append(
                        self._build_block(current)
                    )
                    current = []

                blocks.append(line)
                continue

            # ==========================================
            # LESSON HEADING
            # ==========================================

            if re.match(
                r"^Pertemuan\s+\d+",
                line,
                re.IGNORECASE,
            ):

                if current:
                    blocks.append(
                        self._build_block(current)
                    )
                    current = []

                blocks.append(line)
                continue

            # ==========================================
            # NUMBERED ACTIVITY
            # ==========================================

            if re.match(
                r"^\d+[\.\)]\s+",
                line,
            ):

                if current:
                    blocks.append(
                        self._build_block(current)
                    )
                    current = []

                blocks.append(line)
                continue

            # ==========================================
            # NORMAL TEXT
            # ==========================================

            current.append(line)

        # ==============================================
        # REMAINING CONTENT
        # ==============================================

        if current:
            blocks.append(
                self._build_block(current)
            )

        return [
            block
            for block in blocks
            if block.strip()
        ]

    # ==================================================
    # CODE DETECTION
    # ==================================================

    def _looks_like_code_start(
        self,
        line: str,
    ) -> bool:

        stripped = line.strip()

        # HTML opening tag
        if re.match(
            r"^<\s*[a-zA-Z][^>]*>",
            stripped,
        ):
            return True

        # CSS selector / block
        if re.match(
            r"^[.#]?[a-zA-Z][\w-]*\s*\{",
            stripped,
        ):
            return True

        # JavaScript
        if re.match(
            r"^(if|else|for|while|function|const|let|var)\b",
            stripped,
        ):
            return True

        return False

    def _looks_like_code_end(
        self,
        line: str,
    ) -> bool:

        stripped = line.strip()

        # HTML closing tag
        if re.match(
            r"^<\s*/[a-zA-Z][^>]*>\s*$",
            stripped,
        ):
            return True

        # Single-line code
        if (
            stripped.endswith(";")
            or stripped.endswith("}")
        ):
            return True

        return False

    # ==================================================
    # HELPERS
    # ==================================================

    def _normalize_lines(
        self,
        text: str,
    ) -> list[str]:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = []

        for line in text.split("\n"):

            line = line.strip()

            if line:
                lines.append(line)

        return lines

    def _build_block(
        self,
        lines: list[str],
    ) -> str:

        return "\n".join(
            line.strip()
            for line in lines
            if line.strip()
        )