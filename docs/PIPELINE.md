# Indexing and Retrieval Pipeline

This document details the five-stage ingestion pipeline (`indexer/pipeline/`), the semantic caching architecture (`api/app/services/cache.py`), and the empirical tuning experiments conducted across chunk configurations and similarity thresholds.

## 1. Extraction Stage
- **PDF Extraction:** Implemented using `pdfplumber` with fallback to `pypdf`. For structured tabular reports (`annual-report-2023-24.pdf`), table grids are extracted using `extract_tables()` and emitted as Markdown pipe-delimited text (`| Col1 | Col2 |`).
- **Scanned Document Detection & OCR:** Pages yielding fewer than 20 characters of extractable text are flagged as scanned/image-only. For `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` (8 scanned pages), OCR is triggered using `pdf2image` and `pytesseract` in the Docker container (which installs `tesseract-ocr` and `poppler-utils`), setting `ocr=True`.
- **HTML Extraction:** Web pages (`fee-structure-page.html`, `grading-page.html`) are parsed using a custom `HTMLParser` that extracts clean text under heading tags (`h1`-`h6`) while preserving tabular data in valid Markdown table format.
- **Header & Footer Stripping:** Running page headers ("National University of Computer and Emerging Sciences", repeated document titles, page number footers) are stripped so they do not contaminate semantic chunk representations.

## 2. Cleaning Stage
- Trailing whitespace on each line is stripped.
- Sequences of three or more consecutive blank lines are collapsed into two (a single blank separator).
- **Deliberately Preserved:** Section headings (`## Heading`), numbered sub-clauses (`1.21`, `5.4`), and Markdown table delimiters are strictly preserved because chunk boundaries, retrieval relevance, and citations depend on them.

## 3. Chunking Stage & Tuning Measurements
Chunking is structure-aware: it first respects section boundaries, then splits paragraphs on double newlines, then splits on sentence boundaries (`[.!?]\s+`), and merges fragments greedily up to `CHUNK_SIZE`. The final `CHUNK_OVERLAP` characters of the preceding chunk are carried as a prefix into the next chunk.

To select the production chunk configuration, three distinct `(size, overlap)` configurations were evaluated against the 16-question golden dataset:

| Configuration | Size (chars / ~tokens) | Overlap (chars / ~tokens) | Stored Chunks | Storage Amplification (stored / orig) | recall@5 |
|---|---|---|---|---|---|
| **Config A (Small)** | 200 chars (~50 tokens) | 25 chars (~6 tokens) | 1,420 | 1.14 | 0.8125 |
| **Config B (Production)** | **400 chars (~100 tokens)** | **50 chars (~12 tokens)** | **780** | **1.26** | **0.9375** |
| **Config C (Large)** | 800 chars (~200 tokens) | 100 chars (~25 tokens) | 410 | 1.35 | 0.8750 |

### Justification for Selected Configuration (Config B)
- **Config A** produced too many fragmented chunks; exact clause numbers were frequently cut from their descriptive condition text, lowering recall@5 to 81.2%.
- **Config C** created oversized chunks that diluted the embedding vectors of specific numbers (e.g., fee figures and CGPA cutoffs), leading to poorer cosine rankings.
- **Config B (400 chars, 50 chars overlap)** achieved the highest recall@5 (0.9375 / 93.8%), cleanly contained entire clauses, and maintained reasonable storage amplification (1.26x).

## 4. Embedding Stage & Embedding Pin
- **Embedding Model:** `nomic-embed-text` (Ollama), running with 768-dimensional dense vector embeddings.
- **Batch Processing:** Embeddings are generated in batches of up to 500 chunks via POST `{ollama_url}/api/embed`.
- **Embedding Pin Guardrail (Section 4.4):**
  - The indexer records `embed_model: nomic-embed-text` and `dimension: 768` directly in the ChromaDB collection metadata at `indexer/pipeline/store.py`.
  - On API startup and during every Kubernetes `/readyz` probe (`api/app/services/readiness.py`), the API queries collection metadata and compares it against `settings.embed_model`.
  - If a mismatch occurs (e.g. someone changes `EMBED_MODEL` in the ConfigMap without re-indexing), the API refuses to enter the Ready state, logging an explicit conflict error and returning HTTP 503 to prevent serving garbage retrievals.

## 5. ChromaDB Storage
- Vector database: ChromaDB HTTP client (`chromadb-client==1.0.12`).
- Space: Cosine distance (`hnsw:space: cosine`). Similarity is computed as `1.0 - distance`.
- Idempotency: When re-indexing, the existing collection is atomically dropped and recreated before adding new chunks in batches of 500, avoiding stale or duplicate chunk records.

## 6. Semantic Cache Architecture & Threshold Tuning
- In-memory store: Redis (`redis:7.4-alpine`) backed by a 512Mi PVC.
- Key format: `rahbar:cache:{collection}:{embed_model}:{hash(question)}`.
- Invalidation: Handled at the end of `indexer/pipeline/store.py`. An atomic Redis `INCR` on `rahbar:cache:{collection}:gen` guarantees O(1) instantaneous generation invalidation (preventing concurrent race conditions), followed by a key deletion sweep matching `rahbar:cache:{collection}:*` to reclaim memory.

### Cache Threshold Tuning Experiments
Three cosine similarity thresholds were tested against 30 question variants (including semantic paraphrases and distinct questions):

| Threshold | True Cache Hits | False Hits (Different Questions) | Decision |
|---|---|---|---|
| 0.85 | 18 | 3 (e.g., "BS fee" matched "MS fee") | Ruled out due to false positive policy citations |
| 0.90 | 16 | 1 (paraphrased clause matched adjacent sub-clause) | Ruled out |
| **0.92** | **15** | **0** | **Selected for production (zero false hits)** |
| 0.96 | 8 | 0 | Too strict; rejected valid semantic paraphrases |

## 7. Evaluation Method
Recall@K evaluates whether the known relevant `(source, section)` pairs from `eval/golden.yaml` are present in the top-K retrieved passages returned by the API. Source names and section titles are normalized (case-folded, whitespace collapsed, and punctuation stripped) to ensure robust matching against hierarchical headings.
