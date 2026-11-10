# ADR 0004: LLM Provider Architecture, Secret Governance, and Data Privacy

## Status
Accepted · November 2026

## Context
Educational institutions handle student queries that may contain personally identifiable information (PII), student ID numbers, or academic standing details. Routing all student queries to commercial public cloud LLMs risks leaking student queries into third-party training pipelines. Furthermore, academic grading requirements mandate support for offline/local development without paid credit card requirements.

## Decision
We implemented a pluggable LLM provider interface (`api/app/providers/llm/base.py`) selected via `LLM_PROVIDER` in the ConfigMap:
1. **Default: Ollama Host / Local Deployment (`ollama-host`):** Queries run against an on-premises or laptop Ollama instance running `llama3.2:3b`. Zero student data leaves the institutional boundary.
2. **Pluggable Hosted Fallback (`hosted`):** For cloud deployment without GPU nodes, an OpenAI-compatible interface connects to free-tier inference endpoints.
3. **Strict Secret Management:** Secrets (`LLM_API_KEY`) are never committed to Git. In Kubernetes, keys are loaded from Kubernetes Secret `rahbar-secret` created via CLI. `Settings.llm_api_key` specifies `repr=False` so keys never appear in application logs or traceback dumps.
4. **Offline Testing Stub (`stub`):** A deterministic offline stub provider allows full unit and integration testing in CI without network or GPU dependencies.

## Options Considered
1. **Hardcoded Cloud Provider (OpenAI/Anthropic):** Rejected due to credit card barriers, cost, and potential FERPA/PII data governance violations.
2. **Pure Local Transformer Model in API Container:** Rejected due to heavy container image sizes (10GB+) and CPU contention with FastAPI serving threads.
3. **Decoupled Local Ollama Service (Chosen):** Provides the best balance of data sovereignty, local offline development, and zero subscription costs.

## Consequences
- **Positive:** Full data privacy compliance; zero external data leakage in default configuration; secrets are safe from Git leaks; offline development is frictionless.
- **Negative / Tradeoffs:** Running Ollama locally requires exposing `OLLAMA_HOST=0.0.0.0` on port 11434 so Minikube containers can route to the host machine.

