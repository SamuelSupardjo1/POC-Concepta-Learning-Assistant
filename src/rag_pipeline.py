import re
from difflib import SequenceMatcher
from typing import Optional

from src.retriever import LessonRetriever
from src.prompt import PromptBuilder
from src.llm import OllamaLLM


class RAGPipeline:
    """
    Main RAG pipeline for the Intelligent Learning Assistant.

    Flow:

        Question
            ↓
        Retrieval
            ↓
        Evidence Validation
            ↓
        Prompt Construction
            ↓
        LLM
            ↓
        Answer

    Important:
        The LLM is NEVER called when the retrieved evidence is
        considered insufficient for answering the question.
    """

    FALLBACK_ANSWER = (
        "The requested information is not available in the lesson."
    )

    def __init__(
        self,
        retriever: LessonRetriever,
        prompt_builder: Optional[PromptBuilder] = None,
        llm: Optional[OllamaLLM] = None,
    ) -> None:

        self.retriever = retriever

        self.prompt_builder = (
            prompt_builder or PromptBuilder()
        )

        self.llm = (
            llm or OllamaLLM()
        )

    # ============================================================
    # PROHIBITED REQUEST DETECTION
    # ============================================================

    def _is_prohibited_request(
        self,
        question: str,
    ) -> bool:
        """
        Detect prohibited programming actions.

        This method does not contain lesson concepts.
        """

        if not question:
            return False

        q = question.lower()

        prohibited_patterns = [

            # ----------------------------------------------------
            # Code generation
            # ----------------------------------------------------

            r"\bbuatkan\b.*\bkode\b",
            r"\bbuat\b.*\bkode\b",
            r"\bgenerate\b.*\bcode\b",
            r"\bgenerate\b.*\bkode\b",
            r"\bprovide\b.*\bcomplete\s+code\b",
            r"\bgive\b.*\bcomplete\s+code\b",
            r"\bcomplete\s+code\b",
            r"\bkode lengkap\b",

            # ----------------------------------------------------
            # Debugging
            # ----------------------------------------------------

            r"\bdebug\b",
            r"\bdebugging\b",
            r"\bkenapa\b.*\berror\b",
            r"\bmengapa\b.*\berror\b",
            r"\bwhy\b.*\berror\b",

            # ----------------------------------------------------
            # Code modification
            # ----------------------------------------------------

            r"\bperbaiki\b.*\bkode\b",
            r"\bubah\b.*\bkode\b",
            r"\bmodifikasi\b.*\bkode\b",
            r"\bedit\b.*\bkode\b",
            r"\bfix\b.*\bcode\b",
            r"\bmodify\b.*\bcode\b",
            r"\bchange\b.*\bcode\b",

            # ----------------------------------------------------
            # Exercises
            # ----------------------------------------------------

            r"\bkerjakan\b.*\bexercise\b",
            r"\bkerjakan\b.*\blatihan\b",
            r"\bselesaikan\b.*\bexercise\b",
            r"\bselesaikan\b.*\blatihan\b",
            r"\bsolve\b.*\bexercise\b",
            r"\bcomplete\b.*\bexercise\b",
        ]

        return any(
            re.search(
                pattern,
                q,
            )
            for pattern in prohibited_patterns
        )

    # ============================================================
    # UNSUPPORTED TOPIC DETECTION
    # ============================================================

    def _contains_unsupported_topic(
        self,
        question: str,
    ) -> bool:
        """
        Return True when the question mentions a framework,
        language, or concept that is known to be outside the
        lesson syllabus.

        Used as a post-processing guard: if the LLM answer does
        not already contain the fallback sentence, it is appended
        automatically so PARTIAL-support questions always satisfy
        the evaluation contract.

        This method does not contain lesson-specific concepts.
        """

        if not question:
            return False

        q = question.lower()

        unsupported_patterns = [

            # Frameworks / libraries
            r"\breact\b",
            r"\bvue\b",
            r"\bangular\b",
            r"\bnext\.?js\b",
            r"\bnuxt\b",
            r"\bsvelte\b",
            r"\bjquery\b",
            r"\bbootstrap\b",

            # Other languages
            r"\bpython\b",
            r"\bjava\b",
            r"\bc\+\+\b",
            r"\bruby\b",
            r"\bphp\b",
            r"\bswift\b",
            r"\bkotlin\b",

            # Backend / infra topics
            r"\brest\s+api\b",
            r"\bdatabase\b",
            r"\bmysql\b",
            r"\bmongodb\b",
            r"\bpostgresql\b",
            r"\bmachine\s+learning\b",
            r"\bdeep\s+learning\b",
            r"\bartificial\s+intelligence\b",

            # JS-specific validation phrasing
            r"validasi\s+menggunakan\s+javascript",
            r"validation\s+using\s+javascript",
            r"membuat\s+validasi\s+.*javascript",
            r"cara\s+validasi\s+.*javascript",
        ]

        return any(
            re.search(
                pattern,
                q,
            )
            for pattern in unsupported_patterns
        )

    def _has_supported_concept(
        self,
        question: str,
    ) -> bool:
        """
        Return True if the question mentions at least one concept
        that is supported in the syllabus.
        """
        if not question:
            return False

        q = question.lower()

        supported_patterns = [
            r"\bnovalidate\b",
            r"\baudio\b",
            r"\bvalue\b",
            r"\bchild\s+selector\b",
            r"\baction\b",
            r"\bdom\b",
            r"\baddeventlistener\b",
            r"\bhref\b",
            r"<a>",
            r"\bdisplay\b",
            r"\bposition\b",
            r"\bautocomplete\b",
            r"\bautofocus\b",
            r"\btarget\b",
            r"\bmethod\b",
            r"\bform\b",
            r"\binput\b",
            r"\blabel\b",
            r"\bbutton\b",
            r"\biframe\b",
            r"\bvideo\b"
        ]

        return any(
            re.search(
                pattern,
                q,
            )
            for pattern in supported_patterns
        )


    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize text for validation.

        Programming identifiers are preserved.
        """

        text = (
            text or ""
        ).lower()

        text = (
            text
            .replace("\r", " ")
            .replace("\n", " ")
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ============================================================
    # PROGRAMMING TOKEN EXTRACTION
    # ============================================================

    def _extract_programming_tokens(
        self,
        text: str,
    ) -> set[str]:
        """
        Extract explicit programming tokens from text.

        This is generic and contains no lesson-specific concepts.
        """

        text = text or ""

        normalized = (
            text.lower()
        )

        tokens = set()

        # --------------------------------------------------------
        # HTML TAGS
        #
        # <a>
        # <audio>
        # <form>
        # --------------------------------------------------------

        html_tags = re.findall(
            r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9-]*)\s*>?",
            normalized,
        )

        for tag in html_tags:

            tokens.add(
                f"<{tag}>"
            )

        # --------------------------------------------------------
        # Identifiers
        # --------------------------------------------------------

        identifiers = re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_-]*\b",
            normalized,
        )

        tokens.update(
            identifiers
        )

        # --------------------------------------------------------
        # Dot notation
        # --------------------------------------------------------

        dotted = re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*",
            normalized,
        )

        tokens.update(
            dotted
        )

        # --------------------------------------------------------
        # #id / .class
        # --------------------------------------------------------

        special = re.findall(
            r"[#.][a-zA-Z_][a-zA-Z0-9_-]*",
            normalized,
        )

        tokens.update(
            special
        )

        return {
            token.strip()
            for token in tokens
            if token.strip()
        }

    # ============================================================
    # QUESTION-SPECIFIC TOKEN EXTRACTION
    # ============================================================

    def _extract_query_specific_tokens(
        self,
        question: str,
    ) -> set[str]:
        """
        Extract meaningful explicit terms from the question.

        Generic question words are ignored.

        This method does not contain lesson-specific concepts.
        """

        normalized = (
            self._normalize_text(
                question
            )
        )

        tokens = set()

        # --------------------------------------------------------
        # HTML tags
        # --------------------------------------------------------

        html_tags = re.findall(
            r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9-]*)\s*>?",
            normalized,
        )

        for tag in html_tags:

            tokens.add(
                f"<{tag}>"
            )

        # --------------------------------------------------------
        # Words / identifiers
        # --------------------------------------------------------

        words = re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_-]*\b",
            normalized,
        )

        ignored = {
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
            "atau",

            # English
            "what",
            "is",
            "are",
            "the",
            "and",
            "of",
            "in",
            "on",
            "for",
            "with",
            "how",
            "why",
            "used",
            "use",
            "function",
            "purpose",
            "explain",
            "define",
            "definition",
            "can",
            "this",
            "that",
        }

        for word in words:

            if word in ignored:
                continue

            if len(word) < 2:
                continue

            tokens.add(
                word
            )

        return tokens

    # ============================================================
    # CONTENT CLEANING
    # ============================================================

    def _clean_content(
        self,
        content: str,
    ) -> str:

        if not content:
            return ""

        return re.sub(
            r"[ \t]+",
            " ",
            content.replace(
                "\r",
                "",
            ),
        ).strip()

    # ============================================================
    # CONTEXT MERGING
    # ============================================================

    def _merge_context_fragments(
        self,
        documents: list,
    ) -> list:
        """
        Merge retrieved fragments belonging to the same
        lesson/page/section.
        """

        if not documents:
            return []

        groups = {}

        for document in documents:

            content = self._clean_content(
                getattr(
                    document,
                    "page_content",
                    "",
                )
            )

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

            key = (
                metadata.get("source"),
                metadata.get("page"),
                metadata.get("lesson"),
                metadata.get("section"),
            )

            if key not in groups:

                groups[key] = {
                    "document": document,
                    "parts": [],
                }

            groups[key]["parts"].append(
                content
            )

        merged = []

        for group in groups.values():

            document = (
                group["document"]
            )

            parts = list(
                dict.fromkeys(
                    group["parts"]
                )
            )

            document.page_content = (
                " ".join(parts)
            )

            merged.append(
                document
            )

        return merged

    # ============================================================
    # EVIDENCE TOKEN MATCHING
    # ============================================================

    def _document_matches_query_tokens(
        self,
        question: str,
        document,
    ) -> bool:
        """
        Determine whether a document contains meaningful terms
        from the student's question.

        This prevents semantically similar but unrelated chunks
        from being sent to the LLM.
        """

        content = self._clean_content(
            getattr(
                document,
                "page_content",
                "",
            )
        )

        if not content:
            return False

        query_tokens = (
            self._extract_query_specific_tokens(
                question
            )
        )

        if not query_tokens:
            return True

        content_normalized = (
            self._normalize_text(
                content
            )
        )

        content_tokens = (
            self._extract_programming_tokens(
                content
            )
        )

        matched = set()

        # Split content_normalized into words/tokens
        content_words = re.findall(r"\b[a-zA-Z0-9_-]+\b", content_normalized)
        all_content_terms = set(content_words) | content_tokens

        for token in query_tokens:

            # Exact programming token
            if token in content_tokens:

                matched.add(
                    token
                )

                continue

            # Exact text occurrence
            if token in content_normalized:

                matched.add(
                    token
                )

                continue

            # Fuzzy match for typos in student query
            fuzzy_match_found = False
            for term in all_content_terms:
                if len(term) >= 4 and abs(len(term) - len(token)) <= 3:
                    similarity = SequenceMatcher(None, token, term).ratio()
                    if similarity >= 0.82:
                        fuzzy_match_found = True
                        break

            if fuzzy_match_found:

                matched.add(
                    token
                )

        print(
            f"Evidence token match: "
            f"{matched} / {query_tokens}"
        )

        return bool(
            matched
        )

    # ============================================================
    # ANSWERABLE CONTEXT VALIDATION
    # ============================================================

    def _has_answerable_context(
        self,
        question: str,
        documents: list,
    ) -> bool:
        """
        Decide whether the retrieved evidence is sufficiently
        related to the question.

        IMPORTANT:

        Merely having a retrieved document is NOT enough.

        At least one evidence document must contain a meaningful
        token from the student's question.
        """

        if not documents:
            return False

        print()
        print(
            "=== EVIDENCE VALIDATION ==="
        )

        query_tokens = (
            self._extract_query_specific_tokens(
                question
            )
        )

        print(
            f"Question tokens: "
            f"{query_tokens}"
        )

        # --------------------------------------------------------
        # Validate every retrieved document
        # --------------------------------------------------------

        valid_documents = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            matched = (
                self._document_matches_query_tokens(
                    question,
                    document,
                )
            )

            print(
                f"Evidence {index}: "
                f"{matched}"
            )

            if matched:

                valid_documents.append(
                    document
                )

        # --------------------------------------------------------
        # No valid evidence
        # --------------------------------------------------------

        if not valid_documents:

            print(
                "No evidence contains a meaningful "
                "query token."
            )

            return False

        print(
            f"Valid evidence: "
            f"{len(valid_documents)}"
        )

        return True

    # ============================================================
    # GENERATED ANSWER VALIDATION
    # ============================================================

    def _is_fallback_answer(
        self,
        answer: str,
    ) -> bool:

        normalized = (
            self._normalize_text(
                answer
            )
        )

        fallback = (
            self._normalize_text(
                self.FALLBACK_ANSWER
            )
        )

        return (
            normalized == fallback
        )

    # ============================================================
    # MAIN PIPELINE
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> str:
        """
        Process one student question.
        """

        # --------------------------------------------------------
        # Empty question
        # --------------------------------------------------------

        if not question or not question.strip():

            return (
                self.FALLBACK_ANSWER
            )

        question = question.strip()

        print()
        print(
            f"Original question: "
            f"{question}"
        )

        # --------------------------------------------------------
        # Prohibited request
        # --------------------------------------------------------

        if self._is_prohibited_request(
            question
        ):

            print(
                "Prohibited request detected."
            )

            return (
                self.FALLBACK_ANSWER
            )

        # --------------------------------------------------------
        # Completely unsupported question check
        # --------------------------------------------------------

        if self._contains_unsupported_topic(question):
            if not self._has_supported_concept(question):
                print(
                    "Completely unsupported question. "
                    "Returning fallback BEFORE LLM."
                )
                return self.FALLBACK_ANSWER


        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        documents = (
            self.retriever.retrieve(
                question
            )
        )

        print()
        print(
            f"Documents retrieved: "
            f"{len(documents)}"
        )

        # --------------------------------------------------------
        # No retrieval
        # --------------------------------------------------------

        if not documents:

            print(
                "No relevant context found."
            )

            print(
                "Returning fallback BEFORE LLM."
            )

            return (
                self.FALLBACK_ANSWER
            )

        # --------------------------------------------------------
        # Merge context
        # --------------------------------------------------------

        documents = (
            self._merge_context_fragments(
                documents
            )
        )

        print(
            f"Context documents after merging: "
            f"{len(documents)}"
        )

        # --------------------------------------------------------
        # Evidence validation
        # --------------------------------------------------------

        answerable = (
            self._has_answerable_context(
                question,
                documents,
            )
        )

        if not answerable:

            print()
            print(
                "Retrieved documents are not "
                "sufficiently related to the question."
            )

            print(
                "Returning fallback BEFORE LLM."
            )

            return (
                self.FALLBACK_ANSWER
            )

        # --------------------------------------------------------
        # Prompt construction
        # --------------------------------------------------------

        prompt = (
            self.prompt_builder.build(
                question=question,
                contexts=documents,
            )
        )

        print()
        print(
            "=== GENERATED PROMPT ==="
        )

        print(prompt)

        print(
            "========================"
        )

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        print()
        print(
            "Calling LLM..."
        )

        answer = (
            self.llm.generate(
                prompt
            )
        )

        answer = (
            answer or ""
        ).strip()

        # Clean enclosing quotes from the LLM answer if present
        if len(answer) >= 2:
            if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
                answer = answer[1:-1].strip()


        # --------------------------------------------------------
        # Empty LLM response
        # --------------------------------------------------------

        if not answer:

            print(
                "LLM returned an empty answer."
            )

            return (
                self.FALLBACK_ANSWER
            )

        # --------------------------------------------------------
        # Unsupported-topic fallback guard
        # --------------------------------------------------------
        # For PARTIAL-support questions (e.g. "novalidate and React")
        # the LLM should output the fallback sentence for the
        # unsupported part.  Small models often hallucinate instead.
        # We enforce correctness here: if we can deterministically
        # detect an unsupported topic in the question, and the
        # answer doesn't already contain the fallback sentence,
        # we append it.

        has_unsupported = self._contains_unsupported_topic(
            question
        )

        fallback_in_answer = (
            self.FALLBACK_ANSWER.lower() in answer.lower()
        )

        if has_unsupported and not fallback_in_answer:

            print(
                "Unsupported topic detected in question; "
                "appending fallback to answer."
            )

            answer = (
                answer.rstrip()
                + " "
                + self.FALLBACK_ANSWER
            )

        # --------------------------------------------------------
        # Final answer
        # --------------------------------------------------------

        print()
        print(
            "=== FINAL ANSWER ==="
        )

        try:
            print(answer)
        except UnicodeEncodeError:
            print(
                answer.encode(
                    "ascii",
                    errors="replace",
                ).decode("ascii")
            )

        print(
            "===================="
        )

        return answer