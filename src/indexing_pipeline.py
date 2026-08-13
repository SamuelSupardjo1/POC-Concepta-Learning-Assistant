from langchain_core.documents import Document

from src.loader import LessonLoader
from src.content_segmenter import ContentSegmenter
from src.theory_extractor import TheoryExtractor
from src.structure_identifier import StructureIdentifier
from src.structure_aware_chunker import StructureAwareChunker
from src.embedding import EmbeddingModel
from src.vectordb import LessonVectorDB
from src.chunk_quality_filter import ChunkQualityFilter


class IndexingPipeline:
    """
    End-to-end pipeline for creating the theory knowledge base.

    PDF
    -> Segmentation
    -> Classification
    -> Theory Extraction
    -> Structure Identification
    -> Structure-Aware Chunking
    -> Embedding
    -> ChromaDB
    """

    def __init__(self, lesson_path: str):

        self.lesson_path = lesson_path

        self.segmenter = ContentSegmenter()
        self.extractor = TheoryExtractor()
        self.identifier = StructureIdentifier()
        self.chunker = StructureAwareChunker()
        self.quality_filter = ChunkQualityFilter()

        self.embedding = EmbeddingModel().get_model()

        self.vectordb = LessonVectorDB(
            self.embedding
        )
        

    def run(self):

        print("=" * 70)
        print("CONCEPTA - INDEXING PIPELINE")
        print("=" * 70)

        # ==================================================
        # 1. LOAD DOCUMENTS
        # ==================================================

        print("\n[1] Loading documents...")

        loader = LessonLoader(
            self.lesson_path
        )

        documents = loader.load()

        print(
            f"Documents loaded: {len(documents)}"
        )

        # ==================================================
        # STATISTICS
        # ==================================================

        total_blocks = 0
        total_theory_blocks = 0
        total_structured_blocks = 0
        total_chunks = 0

        # ==================================================
        # 2. PROCESS DOCUMENTS
        # ==================================================

        print("\n[2] Processing documents...")

        all_chunks = []

        for document in documents:

            # ------------------------------
            # Segmentation
            # ------------------------------

            blocks = self.segmenter.segment(
                document.page_content
            )

            total_blocks += len(blocks)

            # ------------------------------
            # Theory extraction
            # ------------------------------

            extracted = self.extractor.extract(
                blocks
            )

            total_theory_blocks += len(
                extracted
            )

            # ------------------------------
            # Structure identification
            # ------------------------------

            structured = self.identifier.identify(
                extracted
            )

            total_structured_blocks += len(
                structured
            )

            # ------------------------------
            # Structure-aware chunking
            # ------------------------------

            chunks = self.chunker.chunk(
                structured
            )

            total_chunks += len(chunks)

            # ------------------------------
            # Add document metadata
            # ------------------------------

            for chunk in chunks:

                chunk["metadata"]["page"] = (
                    document.metadata.get("page")
                )

                chunk["metadata"]["source"] = (
                    document.metadata.get("source")
                )

                all_chunks.append(chunk)

        # ==================================================
        # PROCESSING STATISTICS
        # ==================================================

        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)

        print(
            f"Documents          : {len(documents)}"
        )

        print(
            f"Total blocks       : {total_blocks}"
        )

        print(
            f"Theory blocks      : {total_theory_blocks}"
        )

        print(
            f"Structured blocks  : {total_structured_blocks}"
        )

        print(
            f"Final chunks       : {total_chunks}"
        )
        # ==================================================
        # QUALITY FILTER
        # ==================================================

        print("\n[3] Applying chunk quality filter...")

        chunks_before_filter = len(all_chunks)

        all_chunks = self.quality_filter.filter(
            all_chunks
        )

        chunks_after_filter = len(all_chunks)

        print(
            f"Chunks before filter : {chunks_before_filter}"
        )

        print(
            f"Chunks after filter  : {chunks_after_filter}"
        )

        print(
            f"Chunks removed       : "
            f"{chunks_before_filter - chunks_after_filter}"
        )

        # ==================================================
        # 3. CONVERT TO LANGCHAIN DOCUMENTS
        # ==================================================

        documents_for_db = []

        for chunk in all_chunks:

            documents_for_db.append(
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"],
                )
            )

        # ==================================================
        # 4. STORE IN CHROMADB
        # ==================================================

        print(
            "\n[3] Adding documents to ChromaDB..."
        )
        print("\n[4] Resetting ChromaDB...")

        self.vectordb.clear()


        print("ChromaDB cleared.")

        print("\n[5] Adding documents to ChromaDB...")

        self.vectordb.add_documents(
            documents_for_db
        )

        print(
            "Indexing completed."
        )

        # ==================================================
        # FINAL RESULT
        # ==================================================

        print("\n" + "=" * 70)
        print("INDEXING RESULT")
        print("=" * 70)

        print(
            f"Indexed chunks: {len(documents_for_db)}"
        )

        # ==================================================
        # SAMPLE CHUNKS
        # ==================================================

        if documents_for_db:

            sample_indexes = [
                0,
                len(documents_for_db) // 2,
                len(documents_for_db) - 1,
            ]

            for number, index in enumerate(
                sample_indexes,
                start=1,
            ):

                document = documents_for_db[index]

                print("\n" + "-" * 70)
                print(
                    f"SAMPLE CHUNK {number}"
                )
                print("-" * 70)

                print(
                    document.page_content
                )

                print("\nMetadata:")

                print(
                    document.metadata
                )

        print("=" * 70)

        return documents_for_db