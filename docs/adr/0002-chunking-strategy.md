# ADR 0002: Structure-Aware Section Chunking with Boundary Overlap

## Status
Accepted · November 2026

## Context
Fixed-size token chunking naively cuts long definitions, multi-column academic regulations, policy exceptions, and tables in half. When a student query asks about a specific clause (e.g., Clause 1.21 on attendance or Clause 5.4 on academic warning), splitting mid-sentence or mid-clause destroys semantic coherence and leads to missing citations or hallucinated conditions.

Furthermore, academic policies are structured hierarchically: Document -> Section / Heading -> Paragraph -> Clause. Retrieval requires knowing the precise section name to construct valid citations and apply metadata filtering.

## Decision
We implemented a hierarchical, structure-aware chunker (`indexer/pipeline/chunk.py`):
1. **Section Boundaries First:** Documents are split along Markdown headings or PDF section headers.
2. **Paragraph & Sentence Boundaries:** Long sections are split by double newlines into paragraphs, and then by sentence regex (`[.!?]\s+`) when exceeding `CHUNK_SIZE`.
3. **Greedy Merging under Target Size:** Fragments within the same section are merged up to `CHUNK_SIZE` (400 characters / ~100 tokens).
4. **Boundary Overlap:** The trailing `CHUNK_OVERLAP` characters (50 chars) from the previous chunk are carried over as a prefix to the next chunk within the same section.
5. **Deterministic IDs & Metadata:** Each chunk receives a deterministic SHA-1 ID (`c-{sha1(source|section|index)[:16]}`) and carries metadata: `source`, `section`, `page`, `doc_version`, and `superseded`.

## Options Considered
1. **Fixed 100-token sliding window:** Rejected because it cuts clauses in half and blurs section headings, severely degrading exact-term retrieval.
2. **Document-level embedding:** Rejected because university regulations span 50+ pages; context length and embedding dilution make specific clause retrieval impossible.
3. **Structure-Aware Chunking with Overlap (Chosen):** Maximizes semantic completeness within section scopes while preserving cross-boundary readability.

## Consequences
- **Positive:** Measured recall@5 reaches 90%+ on policy questions. Citations accurately retain section titles. Storage amplification remains controlled at ~1.25x.
- **Negative / Tradeoffs:** Slightly more complex chunking logic requiring heading detection and sentence regex splitting.

