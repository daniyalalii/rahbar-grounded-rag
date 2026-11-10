# AI Usage Disclosure

In compliance with the course academic integrity policy and the instructions in Section 8 of the RAHBAR assignment specification, this document details the AI tools utilized during the design, development, and testing of this project.

## Disclosed Tool Usage

| Tool | What It Wrote or Shaped | What We Changed Afterwards, and Why |
|---|---|---|
| **Claude / Anthropic** | Generated initial scaffolding for FastAPI schemas, drafted boilerplate for Kubernetes NetworkPolicy manifests, and proposed regex patterns for heading extraction. | Heavily refactored the extraction logic to handle malformed PDF tables and empty scanned pages; tightened NetworkPolicy rules to explicitly whitelist CoreDNS port 53; rewritten Pydantic schemas to strictly match assignment constraints. |
| **GitHub Copilot** | Autocompleted repetitive unit test assertions in `api/tests/test_real_pipeline.py` and `indexer/tests/test_pipeline.py`. | Reviewed every generated test assertion to ensure tests genuinely execute the code paths rather than passing trivially; corrected mock object returns for ChromaDB and Redis. |
| **ChatGPT / OpenAI** | Provided reference snippets for Pydantic v2 `model_validate_json` custom error formatting and Docker Alpine package upgrade syntax. | Replaced generic OpenAI API generation logic with Ollama structured JSON extraction (`format="json"`) to ensure local execution without requiring paid third-party API keys. |
| **Google Antigravity Agent** | Conducted comprehensive repository audits, generated the Mermaid architectural diagram, assisted in structuring the seven engineering notes, and verified submission checker outputs. | Verified all file paths, line references, and metric numbers against actual command execution outputs and Git history; validated that no secrets or API keys exist in the repository. |

## Verification and Authorship
All implementations, architectural designs, Kubernetes manifests, and evaluation functions were reviewed, executed, and validated by the team members. No AI-generated code was committed without local test execution and verification.
