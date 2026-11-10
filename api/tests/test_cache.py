"""Unit tests for semantic cache functionality in api/app/services/cache.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.config import Settings
from app.schemas import AskResponse, Citation, RetrievedRef
from app.services.cache import SemanticCache


def _make_dummy_response(
    answered_from_context: bool = True,
    confidence: str = "high",
    has_citations: bool = True,
) -> AskResponse:
    citations = (
        [Citation(source="rules.pdf", section="1.21")] if has_citations else []
    )
    return AskResponse(
        request_id="test-req-1",
        answer="You need 80% attendance.",
        citations=citations,
        answered_from_context=answered_from_context,
        confidence=confidence,  # type: ignore
        retrieved=[
            RetrievedRef(
                chunk_id="c-1", source="rules.pdf", section="1.21", score=0.88
            )
        ],
        provider="ollama",
    )


def test_cache_hit_and_miss() -> None:
    settings = Settings(cache_threshold=0.90)
    fake_redis = MagicMock()
    cache = SemanticCache(settings, client=fake_redis)

    # Miss when no keys
    fake_redis.keys.return_value = []
    assert cache.check_hit([1.0, 0.0]) is None

    # Simulate cached key
    cached_payload = {
        "question": "What is the attendance rule?",
        "vector": [1.0, 0.0],
        "response": _make_dummy_response().model_dump(),
    }
    fake_redis.keys.return_value = [
        "rahbar:cache:rahbar:nomic-embed-text:hash123"
    ]
    fake_redis.get.return_value = json.dumps(cached_payload)

    # Hit with identical vector (similarity = 1.0 >= 0.90)
    hit = cache.check_hit([1.0, 0.0])
    assert hit is not None
    assert hit.answer == "You need 80% attendance."

    # Miss with orthogonal vector (similarity = 0.0 < 0.90)
    miss = cache.check_hit([0.0, 1.0])
    assert miss is None


def test_cache_exclusions() -> None:
    settings = Settings(cache_threshold=0.90)
    fake_redis = MagicMock()
    cache = SemanticCache(settings, client=fake_redis)

    # 1. Never cache follow-up
    assert not cache.store(
        "question", [1.0], _make_dummy_response(), is_followup=True
    )

    # 2. Never cache abstentions (answered_from_context=False)
    assert not cache.store(
        "question", [1.0], _make_dummy_response(answered_from_context=False)
    )

    # 3. Never cache responses without citations
    assert not cache.store(
        "question", [1.0], _make_dummy_response(has_citations=False)
    )

    # 4. Never cache low confidence / fallback responses
    assert not cache.store(
        "question", [1.0], _make_dummy_response(confidence="low")
    )

    # 5. Store eligible response
    assert cache.store("question", [1.0], _make_dummy_response())
    assert fake_redis.set.called


def test_cache_invalidation() -> None:
    settings = Settings(collection="rahbar", embed_model="nomic-embed-text")
    fake_redis = MagicMock()
    fake_redis.keys.return_value = ["k1", "k2"]
    cache = SemanticCache(settings, client=fake_redis)

    _ = cache.invalidate()
    fake_redis.delete.assert_called_once_with("k1", "k2")
    fake_redis.incr.assert_called_once_with("rahbar:cache:rahbar:gen")


def test_generation_isolation_on_invalidation() -> None:
    """Verify that bumping generation counter isolates old cache entries even before deletion."""
    settings = Settings(collection="rahbar", embed_model="nomic-embed-text", cache_threshold=0.90)
    store_dict: dict[str, str] = {}
    current_gen = [1]

    fake_redis = MagicMock()
    fake_redis.get.side_effect = lambda k: str(current_gen[0]) if k == "rahbar:cache:rahbar:gen" else store_dict.get(k)
    fake_redis.set.side_effect = lambda k, v, ex=None: store_dict.__setitem__(k, v)
    fake_redis.keys.side_effect = lambda pat: [k for k in store_dict if k.startswith(pat.replace("*", ""))]
    fake_redis.incr.side_effect = lambda k: current_gen.__setitem__(0, current_gen[0] + 1) or current_gen[0]

    cache = SemanticCache(settings, client=fake_redis)

    # Store entry in generation 1
    resp = _make_dummy_response()
    stored = cache.store("attendance rule?", [1.0, 0.0], resp)
    assert stored is True

    # Confirm cache hit in generation 1
    hit = cache.check_hit([1.0, 0.0])
    assert hit is not None
    assert hit.answer == "You need 80% attendance."

    # Invalidate by incrementing generation counter
    cache.invalidate()
    assert current_gen[0] == 2

    # Verify that generation 2 reads do NOT return generation 1 entries
    hit_after_invalidation = cache.check_hit([1.0, 0.0])
    assert hit_after_invalidation is None

    # Verify that a write that started under generation 1 is rejected from polluting generation 2
    stale_write = cache.store("late query?", [1.0, 0.0], resp, generation="1")
    assert stale_write is False


def test_redis_outage_graceful_fallback() -> None:
    """Verify that Redis connection errors degrade gracefully to cache miss without raising."""
    settings = Settings(collection="rahbar", embed_model="nomic-embed-text")
    failing_redis = MagicMock()
    failing_redis.get.side_effect = RuntimeError("Redis connection dropped")
    failing_redis.keys.side_effect = RuntimeError("Redis timeout")
    failing_redis.set.side_effect = RuntimeError("Redis out of memory")

    cache = SemanticCache(settings, client=failing_redis)

    # Read should return None gracefully
    hit = cache.check_hit([1.0, 0.0])
    assert hit is None

    # Write should return False gracefully
    stored = cache.store("test?", [1.0, 0.0], _make_dummy_response())
    assert stored is False

    # Invalidate should return 0 gracefully
    cleared = cache.invalidate()
    assert cleared == 0

