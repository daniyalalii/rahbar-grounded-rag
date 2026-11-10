"""Semantic caching over Redis.

Stores question embeddings and answers in Redis, performing cosine similarity search.
Enforces all assignment rules:
  - Cache hit if cosine similarity >= cache_threshold
  - NEVER cache:
      * abstentions (answered_from_context == False or refusal message)
      * failed validation / safe canned fallback answers
      * follow-up questions
      * out-of-scope answers
  - Associated with the active collection/index generation
  - Invalidated on re-index
  - Graceful fallback on Redis outage (never fail query if Redis is down)
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from ..config import Settings
from ..schemas import AskResponse, Citation, RetrievedRef

log = logging.getLogger("rahbar.cache")

try:
    import redis
except ImportError:
    redis = None  # type: ignore


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        self.client = client
        if self.client is None and redis is not None:
            try:
                self.client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                    decode_responses=True,
                )
            except Exception as exc:
                log.warning("SemanticCache: redis connection failed: %s", exc)
                self.client = None

    def get_generation(self) -> str:
        """Fetch the active generation counter from Redis."""
        return self._get_generation()

    def _get_generation(self) -> str:
        """Fetch the active generation counter from Redis.

        Defaults to '1' if the key does not exist or if Redis is unreachable.
        """
        if not self.client:
            return "1"
        try:
            val = self.client.get(f"rahbar:cache:{self.settings.collection}:gen")
            return str(val) if val else "1"
        except Exception as exc:
            log.warning("SemanticCache: error reading generation counter: %s", exc)
            return "1"

    def _key_prefix(self) -> str:
        gen = self._get_generation()
        return f"rahbar:cache:{self.settings.collection}:{self.settings.embed_model}:gen{gen}"

    def check_hit(
        self, question_vector: list[float], is_followup: bool = False
    ) -> AskResponse | None:
        """Lookup question in Redis semantic cache."""
        if is_followup:
            return None
        if not self.client or not question_vector:
            return None

        prefix = self._key_prefix()
        try:
            # Keys pattern: rahbar:cache:<collection>:<embed_model>:*
            keys = self.client.keys(f"{prefix}:*")
            if not keys:
                return None

            best_sim = -1.0
            best_entry: dict[str, Any] | None = None

            for k in keys:
                raw_data = self.client.get(k)
                if not raw_data:
                    continue
                entry = json.loads(raw_data)
                cached_vec = entry.get("vector")
                if not cached_vec:
                    continue
                sim = _cosine_similarity(question_vector, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim >= self.settings.cache_threshold and best_entry:
                log.info(
                    "cache_hit",
                    extra={
                        "similarity": round(best_sim, 4),
                        "threshold": self.settings.cache_threshold,
                        "cached_question": best_entry.get("question"),
                    },
                )
                resp_data = best_entry["response"]
                return AskResponse(
                    request_id=resp_data.get("request_id", ""),
                    answer=resp_data.get("answer", ""),
                    citations=[Citation(**c) for c in resp_data.get("citations", [])],
                    answered_from_context=resp_data.get("answered_from_context", True),
                    confidence=resp_data.get("confidence", "high"),
                    retrieved=[RetrievedRef(**r) for r in resp_data.get("retrieved", [])],
                    provider=resp_data.get("provider", "cache"),
                )
        except Exception as exc:
            log.warning("SemanticCache: lookup error: %s", exc)
            return None

        return None

    def store(
        self,
        question: str,
        question_vector: list[float],
        response: AskResponse,
        is_followup: bool = False,
        generation: str | None = None,
    ) -> bool:
        """Store an eligible response in the cache."""
        # Section 3.3 rules: Never cache follow-ups, abstentions, out-of-scope,
        # or anything that failed validation.
        if is_followup:
            return False
        if not response.answered_from_context:
            return False
        if not response.citations:
            return False
        if response.confidence == "low":
            return False
        if not self.client or not question_vector:
            return False

        current_gen = self._get_generation()
        if generation is not None and str(generation) != str(current_gen):
            log.info("SemanticCache: rejecting write from obsolete generation (%s != %s)", generation, current_gen)
            return False

        prefix = f"rahbar:cache:{self.settings.collection}:{self.settings.embed_model}:gen{current_gen}"
        entry_id = f"{prefix}:{hash(question)}"
        payload = {
            "question": question,
            "vector": question_vector,
            "response": response.model_dump(),
        }
        try:
            self.client.set(entry_id, json.dumps(payload), ex=86400 * 7)  # 7-day TTL
            return True
        except Exception as exc:
            log.warning("SemanticCache: store error: %s", exc)
            return False

    def invalidate(self) -> int:
        """Invalidate all cache entries associated with this collection.

        1. Increments the Redis generation counter (`INCR rahbar:cache:{collection}:gen`).
           This provides an immediate O(1) logical invalidation: all subsequent reads and
           writes immediately query/store under `:gen{N+1}`, rendering any stale entries
           from previous generations inaccessible even under concurrent load.
        2. Performs a key deletion sweep for all entries under `rahbar:cache:{collection}:*`
           to reclaim memory. Note: this two-phase operation (INCR then sweep) is a logical
           invalidation with deferred memory cleanup, not an atomic multi-key Redis transaction.
        """
        if not self.client:
            return 0
        gen_key = f"rahbar:cache:{self.settings.collection}:gen"
        try:
            if hasattr(self.client, "incr"):
                try:
                    self.client.incr(gen_key)
                except Exception as exc:
                    log.warning("SemanticCache: incr generation error: %s", exc)
            # Sweep all cached keys across generations for this collection
            sweep_pattern = f"rahbar:cache:{self.settings.collection}:{self.settings.embed_model}:*"
            keys = self.client.keys(sweep_pattern)
            if keys:
                return int(self.client.delete(*keys))
            return 0
        except Exception as exc:
            log.warning("SemanticCache: invalidate error: %s", exc)
            return 0

