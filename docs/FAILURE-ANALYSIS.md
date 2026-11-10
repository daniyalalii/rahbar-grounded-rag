# Deliberate Failure Analysis

This document records the four deliberate failure experiments conducted across the pipeline stages in accordance with Section 5.5 of the RAHBAR assignment specification. For each stage, we record what was broken, the observed symptom, the diagnosis method (metric, log, or command as if uncaused), and the permanent remedy.

---

## 1. Stage A · Extraction (Indexing Scanned PDF Without OCR)
- **What was broken:** We deliberately skipped the OCR fallback branch in `indexer/pipeline/extract.py` and indexed `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` directly using standard text extraction without running `pytesseract`.
- **Symptom:** The 8-page document yielded zero text chunks (total chunks stored dropped by 45 chunks). When asking questions regarding startup equity and faculty incubation policies, the system either abstained or hallucinated unrelated general rules.
- **Diagnosis:**
  - Running `pdffonts corpus/raw/faculty-entrepreneurship-and-technical-start-ups-policy.pdf` displayed an empty font table, confirming an image-only scanned PDF.
  - Inspecting indexer logs (`json` output on stdout) showed:
    `{"logger": "rahbar.indexer", "msg": "chunked", "source": "faculty-entrepreneurship-and-technical-start-ups-policy.pdf", "chunks": 0}`.
- **Remedy:** Implement automatic scanned-page detection in `extract.py`: if `len(extracted_text) < 20`, trigger OCR with `pdf2image` and `pytesseract`, setting `ocr=True` in page metadata.

---

## 2. Stage B · Chunking (Fixed-Size 100-Token Chunks Without Overlap)
- **What was broken:** We replaced the structure-aware chunker with a fixed 100-character window without sentence boundary preservation and with zero overlap (`size=100, overlap=0`).
- **Symptom:** Recall@5 on exact-term questions dropped dramatically from 93.8% to 56.2%. Questions querying Clause 1.21 failed because the clause number was split across chunk boundaries, separating the heading "1.21 Attendance" from the required "80%" threshold condition.
- **Diagnosis:**
  - Running `python eval/run_eval.py --golden eval/golden.yaml` failed with:
    `GATE FAILED: recall@5 dropped by 0.3750 (from baseline 0.9375 to 0.5625)`.
  - In `eval/results.json`, `recall_at_5_by_type["exact"]` dropped to 0.50.
- **Remedy:** Enforce structure-aware chunking (`indexer/pipeline/chunk.py`) which keeps headings attached to section bodies, splits along paragraph/sentence regex boundaries, and carries 50 characters of overlap between contiguous chunks.

---

## 3. Stage C · Retrieval & Guardrails (Embedding Model Mismatch)
- **What was broken:** We changed `EMBED_MODEL` to `all-minilm` in the API environment variables while leaving ChromaDB populated with `nomic-embed-text` vectors.
- **Symptom:** The API pod failed its Kubernetes readiness probe and remained in `0/1 READY` state, refusing all user traffic.
- **Diagnosis:**
  - `kubectl describe pod -l app=api -n rahbar` reported:
    `Warning Unhealthy Readiness probe failed: HTTP probe failed with statuscode: 503`.
  - Viewing API container logs via `kubectl logs -l app=api -n rahbar` showed:
    `{"level": "ERROR", "logger": "rahbar.readiness", "msg": "embedding_model_mismatch: index has nomic-embed-text but API configured with all-minilm"}`.
- **Remedy:** The readiness probe in `api/app/services/readiness.py` acts as an active guardrail. It prevents the API from serving garbage vectors by refusing traffic until `settings.embed_model` matches the ChromaDB collection metadata.

---

## 4. Stage E · Source & Versioning (Removing the Superseded Filter)
- **What was broken:** In `api/app/retrieval/chroma.py`, we commented out the `where={"superseded": False}` filter in the Chroma query, allowing both the 2023 and 2026 academic policies to compete freely on standard queries.
- **Symptom:** When asking "How are repeating course grades handled?", the model retrieved obsolete 2023 passages and generated an answer quoting repealed rules that contradict the 2026 handbook.
- **Diagnosis:**
  - Running golden set evaluation showed a failure on version-sensitive question `q09`.
  - API response inspection revealed citations pointing to `undergraduate-academic-rules-aug-2023.pdf` instead of `academic-rules-handbook-jun-2026.pdf`.
  - Log audit showed chunks with `superseded: True` entering the prompt context.
- **Remedy:** Re-enable the automatic superseded metadata filter in `ChromaRetriever.retrieve()`: only permit superseded chunks when the query explicitly matches keywords like "old", "previous", "superseded", or past calendar years.
