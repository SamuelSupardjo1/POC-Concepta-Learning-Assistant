class StructureIdentifier:
    """
    Identify the structural context of theory content.

    STRUCTURE blocks define the current lesson/section.
    THEORY blocks inherit the latest structural context.
    """

    def identify(self, extracted_blocks: list[dict]) -> list[dict]:

        results = []

        current_lesson = None
        current_section = None

        for block in extracted_blocks:

            content = block["content"]
            content_type = block["content_type"]

            if content_type != "structure":
                results.append({
                    "content": content,
                    "content_type": content_type,
                    "lesson": current_lesson,
                    "section": current_section,
                })

                continue

            # ------------------------------------------
            # Lesson heading
            # ------------------------------------------

            if self._is_lesson_heading(content):

                current_lesson = content
                current_section = None

                continue

            # ------------------------------------------
            # Other structure = section
            # ------------------------------------------

            current_section = content

        return results

    def _is_lesson_heading(self, text: str) -> bool:

        normalized = text.strip().lower()

        return normalized.startswith("pertemuan ")