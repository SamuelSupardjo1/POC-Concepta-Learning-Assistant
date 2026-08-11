import re
from langchain_core.documents import Document


class TheoryPreprocessor:
    """
    Preprocess lesson documents by removing obvious code blocks
    and document noise while preserving programming theory.
    """

    def process(self, documents: list[Document]) -> list[Document]:
        processed_documents = []

        for document in documents:
            cleaned_text = self.clean_text(document.page_content)

            if not cleaned_text.strip():
                continue

            processed_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=document.metadata.copy(),
                )
            )

        return processed_documents

    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text while preserving theoretical content.
        """

        # --------------------------------------------------
        # 1. Normalize line endings
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
        # 3. Remove common document noise
        # --------------------------------------------------

        text = self.remove_page_numbers(text)
        text = self.remove_repeated_noise(text)

        # --------------------------------------------------
        # 4. Process text blocks
        # --------------------------------------------------

        blocks = self.split_into_blocks(text)

        cleaned_blocks = []

        for block in blocks:

            if self.is_code_block(block):
                continue

            if self.is_noise_block(block):
                continue

            cleaned_blocks.append(block.strip())

        # --------------------------------------------------
        # 5. Reconstruct document
        # --------------------------------------------------

        text = "\n\n".join(cleaned_blocks)

        # --------------------------------------------------
        # 6. Normalize excessive spaces
        # --------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ======================================================
    # BLOCK PROCESSING
    # ======================================================

    def split_into_blocks(self, text: str) -> list[str]:
        """
        Split document into paragraphs/blocks.

        Empty lines are treated as boundaries between blocks.
        """

        blocks = re.split(r"\n\s*\n", text)

        return [
            block.strip()
            for block in blocks
            if block.strip()
        ]

    # ======================================================
    # CODE DETECTION
    # ======================================================

    def is_code_block(self, block: str) -> bool:
        """
        Detect whether a block is likely to be source code.

        The detection is language-agnostic and uses multiple
        indicators instead of checking for one programming language.
        """

        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        code_score = 0

        # --------------------------------------------------
        # Indicator 1: HTML/XML-like complete tags
        # --------------------------------------------------

        html_lines = sum(
            bool(re.match(r"^</?[A-Za-z][^>]*>$", line))
            for line in lines
        )

        if html_lines >= 2:
            code_score += 2

        # --------------------------------------------------
        # Indicator 2: Braces commonly used in code blocks
        # --------------------------------------------------

        brace_lines = sum(
            "{" in line or "}" in line
            for line in lines
        )

        if brace_lines >= 2:
            code_score += 2

        # --------------------------------------------------
        # Indicator 3: Strong programming syntax patterns
        # --------------------------------------------------

        syntax_patterns = [
            r"^\s*(if|else|elif|for|while|switch|case)\s*\(",
            r"^\s*(function|class|def)\s+\w+",
            r"^\s*(var|let|const)\s+\w+\s*=",
            r"^\s*import\s+[\w{]",
            r"^\s*from\s+\S+\s+import\s+",
            r"^\s*#include\s*[<\"]",
            r"^\s*SELECT\s+.+\s+FROM\s+",
            r"^\s*public\s+(class|static|void)",
            r"^\s*return\s+.+;",
        ]

        syntax_matches = 0

        for line in lines:
            for pattern in syntax_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    syntax_matches += 1
                    break

        if syntax_matches >= 2:
            code_score += 2

        # --------------------------------------------------
        # Indicator 4: Code-like indentation
        # --------------------------------------------------

        indented_lines = sum(
            len(line) != len(line.lstrip())
            for line in block.splitlines()
            if line.strip()
        )

        if len(lines) >= 3 and indented_lines >= 2:
            code_score += 1

        # --------------------------------------------------
        # Indicator 5: High concentration of code symbols
        # --------------------------------------------------

        symbol_lines = sum(
            bool(
                re.search(
                    r"[;{}]\s*$|=>|&&|\|\||\+\+|--",
                    line,
                )
            )
            for line in lines
        )

        if len(lines) >= 3 and symbol_lines >= 2:
            code_score += 1

        # --------------------------------------------------
        # Final decision
        # --------------------------------------------------

        return code_score >= 3

    # ======================================================
    # NOISE DETECTION
    # ======================================================

    def is_noise_block(self, block: str) -> bool:
        """
        Detect obvious document noise.
        """

        normalized = block.strip().lower()

        if not normalized:
            return True

        # Very short numeric blocks
        if re.fullmatch(r"\d+", normalized):
            return True

        # Common page labels
        if re.fullmatch(
            r"(page\s*)?\d+",
            normalized,
        ):
            return True

        # Common repeated headers/footers
        noise_patterns = [
            r"timedoor coding academy",
            r"timedoor academy",
        ]

        for pattern in noise_patterns:
            if re.fullmatch(pattern, normalized):
                return True

        return False

    # ======================================================
    # PAGE NUMBER CLEANING
    # ======================================================

    def remove_page_numbers(self, text: str) -> str:
        """
        Remove standalone page numbers.
        """

        return re.sub(
            r"^\s*(page\s*)?\d+\s*$",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    # ======================================================
    # HEADER / FOOTER CLEANING
    # ======================================================

    def remove_repeated_noise(self, text: str) -> str:
        """
        Remove known repeated document headers/footers.

        Only removes exact standalone lines.
        """

        patterns = [
            r"^\s*Timedoor Coding Academy\s*$",
            r"^\s*Timedoor Academy\s*$",
        ]

        for pattern in patterns:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

        return text