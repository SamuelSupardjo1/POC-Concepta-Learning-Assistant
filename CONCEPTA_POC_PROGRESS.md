# CONCEPTA — POC Progress & Development Guide

## 1. Project Overview

**Project:** CONCEPTA — Intelligent Learning Assistant  
**Research context:** Skripsi  
**Main approach:** Retrieval-Augmented Generation (RAG)

The system is intended to support students in understanding programming concepts from structured course materials.

The system is **NOT** intended to be:
- an AI coding assistant,
- a code generator,
- a debugger,
- a system that directly provides complete coding solutions.

The main purpose is to answer conceptual programming questions using the lesson material as the primary knowledge source and provide relevant lesson context/reference.

---

# 2. Final System Goal

The final system should be able to:

1. Load programming lesson PDFs.
2. Extract and preprocess their content.
3. Separate useful theory from code, activities, structural elements, and noise.
4. Identify lesson and section structure.
5. Create structure-aware semantic chunks.
6. Generate embeddings for theory chunks.
7. Store embeddings and metadata in ChromaDB.
8. Receive a student's conceptual question.
9. Convert the question into an embedding.
10. Retrieve the most relevant theory chunks from ChromaDB.
11. Construct a context-aware prompt.
12. Send the retrieved context to the selected LLM.
13. Generate an educational answer grounded in the retrieved lesson material.
14. Display lesson/section references with the answer when possible.

Target architecture:

PDF / Curriculum / FAQ
        ↓
Document Loading
        ↓
Preprocessing
        ↓
Block Segmentation
        ↓
Content Classification
        ↓
Theory Extraction
        ↓
Structure Identification
        ↓
Structure-Aware Chunking
        ↓
Embedding
        ↓
ChromaDB
        ↓
Student Question
        ↓
Query Embedding
        ↓
Semantic Retrieval
        ↓
Prompt Construction
        ↓
Qwen2.5:1.5b
        ↓
Answer + Lesson Reference

---

# 3. Important Scope Changes

The original idea contained an adaptive-learning element intended to reduce repeated questions from students and make the system easier for teachers.

That adaptive element is currently **removed from the core scope**.

The current focus is:

> Helping students understand programming concepts through curriculum/lesson-aware RAG retrieval.

The system is therefore not primarily designed to adapt lesson difficulty or automate teacher workload.

---

# 4. Knowledge Sources

The intended knowledge base consists of:

- Programming lesson PDFs
- Curriculum information
- FAQ / frequently asked questions when available

FAQ remains a possible knowledge source, but it should not be assumed that every question must come from an FAQ.

The lesson material remains the primary source for conceptual answers.

---

# 5. Current Technical Stack

Current POC technologies:

- Python
- LangChain
- ChromaDB
- Hugging Face / Sentence Transformers
- Embedding model: `intfloat/multilingual-e5-small`
- Local LLM: `qwen2.5:1.5b`
- Ollama for local LLM execution

Previously tested LLMs:
- `qwen3:4b` — too slow for the current hardware/POC
- `qwen2.5:3b` — usable but slower
- `qwen2.5:1.5b` — selected for the current POC because of significantly faster response time
- Gemini was also tested but had availability/quota/model issues.

---

# 6. Progress Completed

## 6.1 Document Loading

The system can load the real lesson PDF.

Current real PDF:

- `Lesson_01.pdf`
- Total pages/documents observed: **355**

The PDF contains programming course material including HTML, CSS, JavaScript, explanations, activities, questions, examples, and supporting material.

---

## 6.2 Preprocessing

A preprocessing stage has been implemented.

The purpose is to clean extracted PDF text and remove irrelevant elements while preserving useful educational content.

Examples of removable/irrelevant content:

- page numbers,
- repeated headers,
- source URLs,
- unnecessary PDF labels,
- code labels such as `Codes`,
- other document noise.

Important:

Preprocessing is intended to be **general for programming lesson PDFs**, not exclusively JavaScript.

---

# 7. Block Segmentation

After preprocessing, document content is divided into blocks.

A block is a relatively small content unit obtained from the document before semantic chunking.

Example:

```text
Pertemuan 6 - Link in HTML and About Us Section

Hyperlink

<a href="https://www.amazon.com/">
Lihat Amazon
</a>

Codes

Relative URL adalah alamat untuk menuju
ke halaman yang ada di dalam website yang sama.

<a href="about.html">
Lihat About Us
</a>

Codes

Anchor Link adalah cara untuk menautkan link
ke bagian tertentu di halaman website yang sama.

1. Buat project baru pada Replit
```

The segmentation test successfully produced:

```text
Total blocks: 9
```

---

# 8. Content Classification

A `ContentClassifier` has been implemented.

Current content categories:

```text
THEORY
CODE
ACTIVITY
NOISE
STRUCTURE
```

The classifier is currently rule-based.

Classification flow:

```text
Block
 ↓
NOISE?
 ↓ no
CODE?
 ↓ no
ACTIVITY?
 ↓ no
STRUCTURE?
 ↓ no
THEORY
```

## Classification Examples

```text
"Relative URL adalah alamat..."
        → THEORY

"<a href='about.html'>"
        → CODE

"1. Buat project baru pada Replit"
        → ACTIVITY

"Codes"
        → NOISE

"Pertemuan 6 - Link in HTML..."
        → STRUCTURE
```

---

# 9. Classifier Design

The classifier uses a combination of:

- exact matching,
- regular-expression/pattern matching,
- structural rules,
- programming syntax detection.

It should NOT rely entirely on exact strings.

Generic rules are preferred where possible.

Examples of generic noise detection:

```text
Table of Contents
Source: ...
https://...
www....
Page 1
page numbers
```

Programming code is detected using patterns for:

- HTML,
- CSS,
- JavaScript/programming syntax,
- common programming symbols.

Activities are detected using instruction patterns in both Indonesian and English.

Examples:

```text
Buatlah program sederhana
1. Buat project baru
Create a new variable
1. Create a new variable
```

---

# 10. Classifier Testing

The classifier has been tested using both lesson-specific and general-purpose cases.

The final general robustness test achieved:

```text
GENERAL TEST RESULT: 17/17 passed
```

The tests covered:

- Table of Contents
- Contents
- Source URLs
- direct URLs
- website URLs
- page numbers
- headings
- theory questions
- theory statements
- Indonesian activities
- English activities
- JavaScript code
- HTML code

This is the current classifier checkpoint.

**Do not unnecessarily modify the classifier now unless a real-PDF test reveals a new failure.**

---

# 11. Theory Extraction

A theory extraction stage has been implemented.

Its purpose is to retain only blocks classified as:

```text
THEORY
```

while preserving structural context where available.

Example:

```text
Original blocks: 9
Theory blocks: 4
```

Result:

```text
1. structure
   Pertemuan 6 - Link in HTML and About Us Section

2. structure
   Hyperlink

3. theory
   Relative URL adalah alamat untuk menuju
   ke halaman yang ada di dalam website yang sama.

4. theory
   Anchor Link adalah cara untuk menautkan link
   ke bagian tertentu di halaman website yang sama.
```

The important concept is:

> Code and activities should not become theory knowledge chunks merely because they occur near theory.

---

# 12. Structure Identification

The system identifies structural information such as:

```text
Lesson
Section
Theory
```

Example:

```text
Lesson:
Pertemuan 6 - Link in HTML and About Us Section

Section:
Hyperlink

Theory:
Relative URL adalah alamat...
```

The goal is to preserve the educational context of a theory statement.

---

# 13. Structure-Aware Chunking

A structure-aware chunker has been implemented.

Current concept:

> Each valid theory block becomes a semantic knowledge chunk while preserving its lesson and section metadata.

Example output:

```text
CHUNK 1

Metadata:
  lesson: Pertemuan 6 - Link in HTML and About Us Section
  section: Hyperlink
  content_type: theory

Content:
Relative URL adalah alamat untuk menuju
ke halaman yang ada di dalam website yang sama.
```

And:

```text
CHUNK 2

Metadata:
  lesson: Pertemuan 6 - Link in HTML and About Us Section
  section: Hyperlink
  content_type: theory

Content:
Anchor Link adalah cara untuk menautkan link
ke bagian tertentu di halaman website yang sama.
```

---

# 14. Important Chunking Concept

The system is NOT intended to use simple fixed-size chunking such as:

```text
Every 500 characters → new chunk
```

The current approach is **structure-aware**.

Conceptually:

```text
Lesson
  ↓
Section
  ↓
Theory block
  ↓
Semantic knowledge chunk
```

Fixed chunking and rule-based classification are different concepts.

The classifier can be rule-based while the resulting chunks remain structure-aware.

---

# 15. Existing Indexing Progress

An indexing pipeline has already been created.

Observed result before the latest classifier improvements:

```text
Documents loaded: 355
Final chunks: 441
Indexed chunks: 441
```

ChromaDB indexing completed successfully.

However, this index should be considered **outdated/intermediate** because the preprocessing/classification pipeline has subsequently been improved.

The full indexing process should therefore be rerun after the preprocessing pipeline is finalized.

A sample old indexed chunk showed:

```text
Table of Contents
Meeting 17
...
```

with:

```text
lesson: None
section: None
content_type: theory
page: 1
```

This indicates that the old indexing result still allowed an irrelevant Table of Contents block into the knowledge base.

This must be fixed by rerunning the improved pipeline.

---

# 16. Current Known Issue / Next Technical Task

The immediate next task is:

## FULL REAL-PDF PREPROCESSING

Run the improved pipeline over all 355 pages.

The output should report statistics such as:

```text
======================================================================
PREPROCESSING RESULT
======================================================================

Original documents : 355
Theory blocks      : XXX
Code blocks        : XXX
Activity blocks    : XXX
Noise blocks       : XXX
Structure blocks   : XXX

Final theory chunks: XXX
======================================================================
```

The exact numbers should be obtained from the real PDF.

Do NOT invent these values.

---

# 17. Next Development Sequence

Continue in this order:

## Phase A — Full preprocessing

```text
355 PDF pages
 ↓
Document preprocessing
 ↓
Block segmentation
 ↓
Content classification
 ↓
Theory extraction
 ↓
Structure identification
 ↓
Structure-aware chunking
```

Verify:

- no obvious Table of Contents chunks,
- no repeated headers,
- no page numbers,
- no source URLs,
- no code-only chunks,
- no activity-only chunks,
- theory retains lesson/section context.

---

## Phase B — Final chunk validation

Inspect a representative sample of chunks.

Each chunk should ideally contain:

```text
content
lesson
section
content_type
page/source when available
```

Do not immediately optimize for chunk count.

The primary concern is **semantic quality and retrieval usefulness**.

---

## Phase C — Embedding

Use:

```text
intfloat/multilingual-e5-small
```

Generate embeddings only for valid knowledge chunks.

The current primary knowledge target is theory.

---

## Phase D — ChromaDB

Store:

```text
chunk content
embedding
lesson metadata
section metadata
content_type
page/source metadata
```

Use a clean/new collection when validating the improved pipeline so old incorrect chunks do not remain.

---

## Phase E — Retrieval

Implement:

```text
Student Question
       ↓
Query Embedding
       ↓
ChromaDB similarity search
       ↓
Top-K relevant theory chunks
```

Initially evaluate Top-5 retrieval.

Example:

```text
Question:
"Apa perbedaan Absolute URL dan Relative URL?"

        ↓

Top 5 retrieved chunks
        ↓
Evaluate whether relevant
```

---

## Phase F — Prompt Construction

Construct a prompt that instructs the LLM to:

- answer based on retrieved lesson context,
- explain programming concepts clearly,
- avoid unsupported information,
- provide lesson/section reference,
- avoid generating a complete direct coding solution when the question asks for conceptual understanding.

---

## Phase G — LLM Answer Generation

Current selected model:

```text
Qwen2.5:1.5b
```

via Ollama.

The LLM should receive:

```text
Student question
+
Retrieved theory context
+
System instructions
```

and generate the final educational response.

---

# 18. Final RAG Flow

The intended final system:

```text
                    KNOWLEDGE BASE
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Lesson PDF        Curriculum         FAQ
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                Document Loading
                         ↓
                  Preprocessing
                         ↓
                 Block Segmentation
                         ↓
                Content Classification
                         ↓
                  Theory Extraction
                         ↓
                Structure Identification
                         ↓
              Structure-Aware Chunking
                         ↓
                     Embedding
                         ↓
                     ChromaDB
                         │
                         │
                         │
                  USER QUESTION
                         ↓
                  Query Embedding
                         ↓
                 Semantic Retrieval
                         ↓
                  Top-K Theory Chunks
                         ↓
                 Prompt Construction
                         ↓
                  Qwen2.5:1.5b
                         ↓
                Educational Answer
                         ↓
             Lesson / Section Reference
```

---

# 19. System Scope

## The system CAN:

- answer conceptual programming questions,
- explain programming concepts,
- explain syntax conceptually,
- explain code snippets when provided by the student,
- retrieve relevant lesson material,
- connect answers to lesson/section context,
- use lesson/curriculum/FAQ knowledge sources,
- provide lesson references.

## The system SHOULD NOT:

- act as an AI coding assistant,
- generate complete solutions to programming assignments,
- function as a general-purpose code generator,
- function primarily as a debugger,
- invent answers unrelated to the retrieved knowledge,
- treat code examples as theory knowledge merely because they are present in the PDF.

---

# 20. Important Design Principle

The system should prioritize:

```text
Curriculum / Lesson Context
        ↓
Relevant Theory
        ↓
Grounded Explanation
```

rather than:

```text
Question
 ↓
LLM general knowledge
 ↓
Answer
```

The RAG component is important because the system should answer according to the learning material used in the course.

---

# 21. What NOT To Do Yet

Do NOT currently:

- replace the classifier with an ML classifier,
- over-engineer the preprocessing,
- add dozens of PDF-specific hardcoded rules,
- switch embedding models without a reason,
- switch the LLM without a reason,
- optimize response speed before retrieval works,
- add adaptive learning features,
- build a code-generation feature,
- build a debugging feature.

First complete a working end-to-end POC.

---

# 22. Development Priority

Priority order:

```text
1. Full real-PDF preprocessing
2. Validate extracted theory
3. Validate structure-aware chunks
4. Rebuild ChromaDB
5. Test Top-5 retrieval
6. Evaluate retrieval relevance
7. Build prompt construction
8. Connect Qwen2.5:1.5b
9. Test end-to-end QA
10. Prepare demonstration
```

---

# 23. Current Checkpoint

As of the current development stage:

```text
Document Loading              ✅
Preprocessing                 ✅
Block Segmentation            ✅
Content Classification        ✅ 17/17
Theory Extraction             ✅
Structure Identification      ✅
Structure-Aware Chunking      ✅
Embedding                     ⏳
ChromaDB final rebuild        ⏳
Retrieval evaluation          ⏳
Prompt construction           ⏳
LLM integration               ⏳
End-to-end QA                 ⏳
```

The immediate objective is to move from:

```text
CLASSIFIER VALIDATION
```

to:

```text
FULL REAL-PDF PIPELINE VALIDATION
```

---

# 24. Instructions for CLI Copilot

When continuing development, preserve the following principles:

1. Do not redesign the architecture unless explicitly requested.
2. Do not remove the current content categories.
3. Do not replace structure-aware chunking with simple fixed-size chunking.
4. Keep classifier rules as generic as possible.
5. Add source-specific rules only when necessary and clearly justified.
6. Preserve lesson and section metadata.
7. Do not embed noise, activities, or code-only content as theory knowledge unless explicitly required by the design.
8. Do not silently invent metadata.
9. Show test output after significant changes.
10. Prefer small, testable changes.
11. Keep the current model choice (`qwen2.5:1.5b`) unless performance testing gives a concrete reason to change it.
12. The final system must remain a curriculum/lesson-aware programming learning assistant, not a code generator/debugger.

---

# 25. Immediate Command

After verifying the latest code, the next action should be:

```text
Run the complete preprocessing pipeline against the real 355-page Lesson_01.pdf.

Report:
- total documents/pages,
- classification counts,
- extracted theory count,
- structure count,
- final chunk count,
- representative chunk samples,
- metadata quality.

Do not index the old 441 chunks as the final result.
Rebuild the vector database using the improved preprocessing output.
```

This document is the current development checkpoint and should be updated whenever a major pipeline stage is completed or its design changes.
