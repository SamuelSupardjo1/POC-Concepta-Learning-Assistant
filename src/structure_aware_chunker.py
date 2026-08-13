class StructureAwareChunker:
    """
    Create chunks based on lesson and section structure.

    Each theory block becomes a semantic chunk while
    preserving its structural context.
    """

    def chunk(
        self,
        structured_blocks: list[dict],
    ) -> list[dict]:

        chunks = []

        for index, item in enumerate(
            structured_blocks,
            start=1,
        ):

            # ==================================================
            # Only theory becomes a knowledge chunk
            # ==================================================

            if item["content_type"] != "theory":
                continue

            # ==================================================
            # Lesson is required
            # ==================================================

            lesson = item.get("lesson")

            if not lesson:
                continue

            # ==================================================
            # Content must exist
            # ==================================================

            content = item.get(
                "content",
                "",
            ).strip()

            if not content:
                continue

            # ==================================================
            # Create chunk
            # ==================================================

            chunk = {
                "chunk_id": index,
                "content": content,
                "metadata": {
                    "lesson": lesson,
                    "section": item.get(
                        "section"
                    ),
                    "content_type": "theory",
                },
            }

            chunks.append(chunk)

        return chunks