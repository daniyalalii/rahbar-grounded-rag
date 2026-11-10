# ADR 0003: Redis Semantic Caching and Re-Index Invalidation

## Status
Accepted · November 2026

## Context
Serving LLM-generated answers for frequent university queries (e.g., fee structure, attendance thresholds, grading scales) introduces latency (2-6 seconds per request) and consumes cluster compute resources. Exact-string question caching fails because students phrase questions differently (e.g., "What is the attendance rule?" vs "How much attendance is required for exams?").

However, naive semantic caching introduces two critical failure modes:
1. **Serving Stale Answers After Policy Updates:** If the university updates attendance rules from 75% to 80% and the indexer rebuilds the index, serving cached answers from the old policy causes students to fail exams based on obsolete information.
2. **Serving Cached Mistakes / Hallucinations:** Caching an abstention or an answer that failed validation propagates errors to all subsequent users.

## Decision
We implemented a semantic caching layer in Redis (`api/app/services/cache.py`):
1. **Vector Similarity in Redis:** Incoming questions are embedded with `settings.embed_model`. Redis keys are prefixed with `rahbar:cache:{collection}:{embed_model}:*`.
2. **Threshold Gating:** A cache hit requires cosine similarity >= `CACHE_THRESHOLD` (0.92), selected experimentally to eliminate false hits.
3. **Strict Caching Exclusions:** We never cache:
   - Abstentions (`answered_from_context == False`)
   - Responses without valid citations
   - Fallback / low-confidence responses
   - Follow-up questions
4. **Cache Invalidation on Ingestion:** When the indexer Job finishes rebuilding the ChromaDB collection, it connects to Redis and flushes all keys matching `rahbar:cache:{collection}:*`.

## Options Considered
1. **Second ChromaDB Collection as Cache:** Rejected because ChromaDB incurs higher query overhead and lacks atomic TTL expiration.
2. **In-Memory LRU Cache in API Pod:** Rejected because three separate API replicas would each maintain inconsistent caches, and restarts would wipe the cache.
3. **Redis Key-Value with Vector Cosine Matching (Chosen):** Centralized across all API replicas, supports atomic invalidation, and survives API restarts via PVC.

## Consequences
- **Positive:** Latency on frequent questions drops from ~3.5s to <15ms. Invalidation ensures newly indexed policies immediately replace old answers.
- **Negative / Tradeoffs:** Requires Redis Deployment and PVC, and NetworkPolicy rules allowing API egress to Redis port 6379.

