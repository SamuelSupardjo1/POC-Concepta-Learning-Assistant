import re
from difflib import SequenceMatcher
from typing import List


from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB


class LessonRetriever:
    """
    Hybrid retriever for CONCEPTA.

    Retrieval strategy:
        1. Semantic retrieval
        2. Lexical keyword matching
        3. Exact programming-token matching
        4. Fuzzy matching for spelling errors
        5. Strict relevance scoring
        6. Theory-only filtering

    No lesson-specific concept mapping is used.
    """

    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 50,
        min_relevance: float = 0.34,
        max_distance: float = 0.38,
    ) -> None:

        self.top_k = top_k
        self.candidate_k = candidate_k
        self.min_relevance = min_relevance
        self.max_distance = max_distance

        self.embedding = EmbeddingModel().get_model()

        self.vectordb = LessonVectorDB(
            self.embedding
        )

    def _exact_html_tag_score(
        self,
        query: str,
        content: str,
    ) -> float:

        query_tags = set(
            f"<{tag}>"
            for tag in re.findall(
                r"<([a-zA-Z][a-zA-Z0-9-]*)",
                query.lower(),
            )
        )

        if not query_tags:
            return 0.0

        content_tag_names = set(
            f"<{tag}>"
            for tag in re.findall(
                r"<([a-zA-Z][a-zA-Z0-9-]*)",
                content.lower(),
            )
        )

        if not content_tag_names:
            return 0.0

        matched = query_tags & content_tag_names

        return (
            len(matched)
            / len(query_tags)
        )

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ============================================================
    # PROGRAMMING TOKEN NORMALIZATION
    # ============================================================

    def _normalize_programming_token(
        self,
        token: str,
    ) -> str:
        """
        Normalize programming tokens while preserving
        important syntax such as HTML tags.

        Examples:
            <A> -> <a>
            addEventListner -> addeventlistner
            addEventListener -> addeventlistener
            querySelector() -> queryselector
        """

        token = token.lower().strip()

        # Remove function parentheses.
        token = re.sub(
            r"\(\s*\)$",
            "",
            token
        )

        # Remove unnecessary whitespace.
        token = re.sub(
            r"\s+",
            "",
            token
        )

        return token

    # ============================================================
    # PROGRAMMING TOKEN EXTRACTION
    # ============================================================

    def _extract_programming_tokens(
        self,
        text: str,
    ) -> set[str]:

        normalized = self._normalize_text(text)

        tokens = set()

        # --------------------------------------------------------
        # HTML TAGS
        # --------------------------------------------------------

        html_tags = re.findall(
            r"<([a-zA-Z][a-zA-Z0-9-]*)",
            normalized
        )

        for tag in html_tags:
            tokens.add(
                self._normalize_programming_token(f"<{tag}>")
            )

        # --------------------------------------------------------
        # HTML ATTRIBUTES
        # --------------------------------------------------------

        attributes = re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9-]*(?=\s*=)",
            normalized
        )

        for attribute in attributes:
            tokens.add(
                self._normalize_programming_token(
                    attribute
                )
            )

        # --------------------------------------------------------
        # JAVASCRIPT / PROGRAMMING METHODS
        # --------------------------------------------------------

        methods = re.findall(
            r"\b[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(",
            normalized
        )

        for method in methods:

            method = method.replace(
                "(",
                ""
            ).strip()

            tokens.add(
                self._normalize_programming_token(
                    method
                )
            )

        # --------------------------------------------------------
        # PROGRAMMING IDENTIFIERS
        # --------------------------------------------------------

        identifiers = re.findall(
            r"\b[a-zA-Z_$][a-zA-Z0-9_$-]{2,}\b",
            normalized,
        )

        for identifier in identifiers:

            tokens.add(
                self._normalize_programming_token(
                    identifier
                )
            )

        return tokens

    # ============================================================
    # FUZZY PROGRAMMING TOKEN SCORE
    # ============================================================

    def _programming_token_similarity(
        self,
        query_token: str,
        content_token: str,
    ) -> float:
        """
        Calculate similarity between programming tokens.

        Used for small spelling mistakes such as:

            addEventListner
            addEventListener
        """

        query_token = self._normalize_programming_token(
            query_token
        )

        content_token = self._normalize_programming_token(
            content_token
        )

        if not query_token or not content_token:
            return 0.0

        # Exact match.
        if query_token == content_token:
            return 1.0

        # HTML tags should use strict matching.
        if (
            query_token.startswith("<")
            or content_token.startswith("<")
        ):
            return 0.0

        # Short programming identifiers should not
        # be aggressively fuzzy matched.
        if len(query_token) < 5:
            return 0.0

        return SequenceMatcher(
            None,
            query_token,
            content_token,
        ).ratio()

    # ============================================================
    # NATURAL LANGUAGE KEYWORDS
    # ============================================================

    def _extract_keywords(
        self,
        query: str,
    ) -> set[str]:

        normalized = self._normalize_text(query)

        programming_tokens = (
            self._extract_programming_tokens(
                query
            )
        )

        stopwords = {
            # Indonesian
            "apa",
            "itu",
            "yang",
            "dan",
            "di",
            "ke",
            "dari",
            "untuk",
            "dalam",
            "pada",
            "dengan",
            "adalah",
            "fungsi",
            "kegunaan",
            "tujuan",
            "cara",
            "bagaimana",
            "mengapa",
            "kenapa",
            "sebutkan",
            "jelaskan",
            "digunakan",
            "penggunaan",
            "sebuah",
            "suatu",
            "bisa",
            "dapat",
            "agar",
            "akan",
            "sebagai",

            # English
            "the",
            "is",
            "what",
            "how",
            "why",
            "of",
            "in",
            "on",
            "for",
            "with",
            "a",
            "an",
            "to",
            "and",
            "does",
            "do",
            "are",
            "used",
            "use",
            "purpose",
            "function",
            "explain",
            "define",
        }

        words = re.findall(
            r"[a-zA-Z0-9_-]+",
            normalized,
        )

        keywords = {
            word
            for word in words
            if word not in stopwords
            and len(word) >= 3
        }

        # Jangan kehilangan token HTML pendek
        # seperti <a>.
        keywords.update(
            programming_tokens
        )

        return keywords

    # ============================================================
    # EXACT TOKEN SCORE
    # ============================================================

    def _exact_token_score(
        self,
        query: str,
        content: str,
    ) -> float:

        query_tokens = (
            self._extract_programming_tokens(
                query
            )
        )

        if not query_tokens:
            return 0.0

        content_tokens = (
            self._extract_programming_tokens(
                content
            )
        )

        if not content_tokens:
            return 0.0

        matched = (
            query_tokens
            & content_tokens
        )

        return (
            len(matched)
            / max(len(query_tokens), 1)
        )

    # ============================================================
    # FUZZY TOKEN SCORE
    # ============================================================

    def _fuzzy_token_score(
        self,
        query: str,
        content: str,
    ) -> float:

        query_tokens = (
            self._extract_programming_tokens(
                query
            )
        )

        content_tokens = (
            self._extract_programming_tokens(
                content
            )
        )

        if not query_tokens or not content_tokens:
            return 0.0

        matched = 0

        for query_token in query_tokens:

            best_similarity = 0.0

            for content_token in content_tokens:

                similarity = (
                    self._programming_token_similarity(
                        query_token,
                        content_token,
                    )
                )

                best_similarity = max(
                    best_similarity,
                    similarity,
                )

            # 0.82 is intentionally strict enough
            # for small spelling mistakes.
            if best_similarity >= 0.82:
                matched += 1

        return (
            matched
            / max(len(query_tokens), 1)
        )

    # ============================================================
    # KEYWORD SCORE
    # ============================================================

    def _keyword_score(
        self,
        query: str,
        content: str,
    ) -> float:

        query_keywords = (
            self._extract_keywords(query)
        )

        content_keywords = (
            self._extract_keywords(content)
        )

        if not query_keywords:
            return 0.0

        if not content_keywords:
            return 0.0

        exact_matches = (
            query_keywords
            & content_keywords
        )

        if exact_matches:
            return (
                len(exact_matches)
                / max(len(query_keywords), 1)
            )

        # Fuzzy natural-language matching.
        fuzzy_matches = 0

        for query_word in query_keywords:

            # HTML/programming tags handled separately.
            if query_word.startswith("<"):
                continue

            if len(query_word) < 4:
                continue

            best_similarity = 0.0

            for content_word in content_keywords:

                if content_word.startswith("<"):
                    continue

                similarity = SequenceMatcher(
                    None,
                    query_word,
                    content_word,
                ).ratio()

                best_similarity = max(
                    best_similarity,
                    similarity,
                )

            if best_similarity >= 0.82:
                fuzzy_matches += 1

        return (
            fuzzy_matches
            / max(len(query_keywords), 1)
        )

    # ============================================================
    # SEMANTIC SCORE
    # ============================================================

    def _semantic_score(
        self,
        distance: float,
    ) -> float:

        if distance >= self.max_distance:
            return 0.0

        score = (
            1.0
            - (
                distance
                / self.max_distance
            )
        )

        return max(
            0.0,
            min(1.0, score)
        )

    # ============================================================
    # FINAL RELEVANCE SCORE
    # ============================================================

    def _calculate_relevance(
        self,
        distance: float,
        keyword_score: float,
        exact_token_score: float,
        fuzzy_token_score: float,
    ) -> float:

        semantic_score = (
            self._semantic_score(
                distance
            )
        )

        # Exact programming token is strongest.
        token_score = max(
            exact_token_score,
            fuzzy_token_score,
        )

        relevance = (
            (semantic_score * 0.50)
            + (keyword_score * 0.25)
            + (token_score * 0.25)
        )

        return round(
            relevance,
            3
        )

    # ============================================================
    # ACCEPTANCE RULE
    # ============================================================

    def _is_relevant(
        self,
        distance: float,
        keyword_score: float,
        exact_token_score: float,
        fuzzy_token_score: float,
        relevance: float,
        html_tag_score: float,
    ) -> bool:

        # Exact programming token.
        if exact_token_score >= 0.5:
            return True

        # Small programming typo.
        if fuzzy_token_score >= 0.5:
            return True

        if html_tag_score >= 0.5:
            return True

        # Strong hybrid relevance.
        if relevance >= self.min_relevance:
            return True

        # Very strong semantic evidence.
        if (
            distance <= 0.25
            and keyword_score >= 0.10
        ):
            return True

        # Semantically close document that contains at least
        # one exact programming token from the query.
        # This handles long multi-part queries where the
        # token ratio is diluted but the core concept matches.
        if (
            distance <= 0.30
            and exact_token_score > 0.0
        ):
            return True

        return False

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query: str,
    ) -> List:

        if not query or not query.strip():
            return []

        query = query.strip()

        # Retrieve candidates with query expansion for technical tokens.
        # This builds robustness when queries contain specific HTML tags, attribute, or JS method typos.
        seen_contents = {}

        # Primary search using full user query
        primary_results = (
            self.vectordb
            .get_db()
            .similarity_search_with_score(
                query,
                k=self.candidate_k,
            )
        )
        for doc, score in primary_results:
            content = getattr(doc, "page_content", "")
            if content:
                seen_contents[content] = (doc, score)

        # Secondary search using individual programming tokens/tags from query
        query_tokens = self._extract_programming_tokens(query)
        for token in query_tokens:
            is_html_tag = token.startswith("<") and token.endswith(">")
            clean_token = token.replace("<", "").replace(">", "").strip()
            if not clean_token:
                continue
            if not is_html_tag and len(clean_token) < 3:
                continue

            token_results = (
                self.vectordb
                .get_db()
                .similarity_search_with_score(
                    token,
                    k=10,
                )
            )
            for doc, score in token_results:
                content = getattr(doc, "page_content", "")
                if content and content not in seen_contents:
                    seen_contents[content] = (doc, score)

        results = list(seen_contents.values())

        scored_documents = []

        print(
            "\n=== RETRIEVAL DEBUG ==="
        )

        for document, distance in results:

            content = (
                getattr(
                    document,
                    "page_content",
                    "",
                )
                or ""
            ).strip()

            if not content:
                continue

            metadata = (
                getattr(
                    document,
                    "metadata",
                    {},
                )
                or {}
            )

            # ----------------------------------------------------
            # THEORY ONLY
            # ----------------------------------------------------

            content_type = str(
                metadata.get(
                    "content_type",
                    "",
                )
            ).lower().strip()

            if content_type != "theory":

                print(
                    "Skipped non-theory:",
                    content_type
                )

                continue

            # ----------------------------------------------------
            # SCORES
            # ----------------------------------------------------

            keyword_score = (
                self._keyword_score(
                    query,
                    content,
                )
            )

            exact_token_score = (
                self._exact_token_score(
                    query,
                    content,
                )
            )

            fuzzy_token_score = (
                self._fuzzy_token_score(
                    query,
                    content,
                )
            )

            html_tag_score = self._exact_html_tag_score(
                query,
                content,
            )

            semantic_score = (
                self._semantic_score(
                    distance
                )
            )

            relevance = (
                self._calculate_relevance(
                    distance,
                    keyword_score,
                    exact_token_score,
                    fuzzy_token_score,
                )
            )

            accepted = (
                self._is_relevant(
                    distance,
                    keyword_score,
                    exact_token_score,
                    fuzzy_token_score,
                    html_tag_score,
                    relevance,
                )
            )

            # ----------------------------------------------------
            # DEBUG
            # ----------------------------------------------------

            print(
                f"Distance: {distance:.4f}"
            )

            print(
                f"Semantic score: "
                f"{semantic_score:.3f}"
            )

            print(
                f"Keyword score: "
                f"{keyword_score:.3f}"
            )

            print(
                f"Exact token score: "
                f"{exact_token_score:.3f}"
            )

            print(
                f"Fuzzy token score: "
                f"{fuzzy_token_score:.3f}"
            )

            print(
                f"Relevance: "
                f"{relevance:.3f}"
            )

            print(
                f"Accepted: "
                f"{accepted}"
            )

            print(
                f"Content: "
                f"{content[:200]}"
            )

            print(
                f"Metadata: "
                f"{metadata}"
            )

            print()

            if not accepted:
                continue

            scored_documents.append(
                (
                    document,
                    distance,
                    relevance,
                    exact_token_score,
                    fuzzy_token_score,
                    keyword_score,
                )
            )

        # --------------------------------------------------------
        # RANKING
        # --------------------------------------------------------

        scored_documents.sort(
            key=lambda item: (
                item[2],   # relevance
                item[3],   # exact token
                item[4],   # fuzzy token
                item[5],   # keyword
                -item[1],  # semantic distance
            ),
            reverse=True,
        )

        final_documents = [
            item[0]
            for item in scored_documents[
                :self.top_k
            ]
        ]

        # --------------------------------------------------------
        # FINAL DEBUG
        # --------------------------------------------------------

        print(
            "======================================"
        )

        print(
            "Distance threshold:",
            self.max_distance,
        )

        print(
            "Minimum relevance:",
            self.min_relevance,
        )

        print(
            "Candidate documents:",
            len(results),
        )

        print(
            "Relevant documents:",
            len(final_documents),
        )

        print(
            "======================================"
        )

        return final_documents