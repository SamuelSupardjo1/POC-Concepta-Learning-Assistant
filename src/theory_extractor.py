from src.content_classifier import (
    ContentClassifier,
    ContentType,
)


class TheoryExtractor:
    """
    Extract theory-related content from classified blocks.

    THEORY and STRUCTURE are retained.
    CODE, ACTIVITY, and NOISE are removed.
    """

    def __init__(self):

        self.classifier = ContentClassifier()

    def extract(
        self,
        blocks: list[str],
    ) -> list[dict]:

        results = []

        for block in blocks:

            content_type = self.classifier.classify(
                block
            )

            if content_type in {
                ContentType.THEORY,
                ContentType.STRUCTURE,
            }:

                results.append(
                    {
                        "content": block,
                        "content_type": content_type.value,
                    }
                )

        return results