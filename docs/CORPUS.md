# Corpus Documentation

Source: FAST-NUCES public website (`nu.edu.pk`) · Documents: 9 · Total PDF Pages: 279 (plus 2 HTML data pages)

This corpus was specifically assembled to satisfy all seven mandatory corpus properties defined in Section 2 of the RAHBAR assignment specification, making every failure mode (A through F) reachable and evaluatable.

## Required Properties Inventory

| # | Required Property | Document(s) that Provide It | Evidence, Notes, and Extraction Behavior |
|---|---|---|---|
| **1** | **At least 6 documents, 40 pages** | All 9 fetched documents in `corpus/raw/` (7 PDFs, 2 HTML). | Exactly 279 PDF pages across 7 PDFs, plus 2 HTML documents (`fee-structure-page.html`, `grading-page.html`). Documented in `corpus/MANIFEST.md` with SHA-256 hashes. |
| **2** | **Two versions of one policy** | (1) `undergraduate-academic-rules-aug-2023.pdf` (version `2023-08`, `superseded: true`) vs (2) `academic-rules-handbook-jun-2026.pdf` (version `2026-06`, `superseded: false`). | Demonstrates failure mode E (outdated source). In 2023, repeating a course grade improvement rules differed; the 2026 revision governs current degree awards. Metadata tagging (`doc_version`, `superseded`) allows the retriever to filter out 2023 rules unless explicitly asked. |
| **3** | **Scanned, image-only PDF** | `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` (version `2022-05`, 8 pages). | `pdffonts` and PDF text extraction reveal 0 extractable text characters across all 8 pages (100% scanned bitmap pages). Demonstrates failure mode A (bad extraction). Our pipeline detects `<20` chars per page and invokes OCR via `pytesseract` / `pdf2image`, setting `ocr=True`. |
| **4** | **Tables carrying real information** | `fee-structure-page.html`, `annual-report-2023-24.pdf`, `grading-page.html`. | HTML and PDF tables containing tuition fees per credit hour (Rs. 12,000 for BS, Rs. 5,000 late fee), graduation stats, and letter grade to grade-point mappings (A+: 4.00, B+: 3.33, etc.). Extracted as structured Markdown tables (`\| Col1 \| Col2 \|`) to prevent column jumbling. |
| **5** | **Long definitions** | `student-handbook-jun-2026.pdf` (64 pages) & `academic-rules-handbook-jun-2026.pdf` (58 pages). | Contains extensive, multi-paragraph definitions of Academic Deficiencies, Cheating & Plagiarism sanctions, and Generative AI usage rules. Structure-aware chunking preserves section headings and uses sentence boundaries to avoid splitting clauses mid-definition. |
| **6** | **Deliberately absent topic** | **Hostel and On-Campus Accommodation Rules** (and bus transport routes/timetables). | Left out on purpose by excluding the university's Transport & Hostel Facilities PDF. The Student Handbook only mentions that hostels exist. Tested via golden set questions `q13` and `q14` to force honest abstention (`answered_from_context: false`). |
| **7** | **Exact-term content** | `academic-rules-handbook-jun-2026.pdf` & `fee-structure-page.html`. | Clause numbers (Clause 1.21 for 80% attendance, Clause 5.4 for Academic Warning CGPA 2.00), exact fee figures (Rs. 12,000 admission fee, Rs. 5,000 semester freeze fee), and degree duration limits (7 years max for 4-year BS). Motivates strict retrieval precision. |

## Raw Corpus Storage and Delivery Mechanism
- In accordance with `STARTER.md`, external university raw documents are downloaded locally into `corpus/raw/` via `python corpus/fetch.py`.
- `corpus/raw/` is gitignored to avoid checking large binary files into the Git repository.
- Verification is performed using `python corpus/fetch.py --verify`, which computes SHA-256 hashes against `corpus/MANIFEST.md`.
- In Kubernetes, the indexer Job accesses the corpus either through an initialized shared storage volume or pre-staged corpus directory, ensuring the indexer remains completely decoupled from API serving.
