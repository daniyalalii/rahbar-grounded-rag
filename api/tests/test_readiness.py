"""Tests for /readyz and the embedding-pin guardrail using fakes."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.services.readiness import check_readiness


class FakeCollection:
    def __init__(self, count_val: int = 3, metadata: dict[str, object] | None = None) -> None:
        self._count = count_val
        self.metadata = metadata if metadata is not None else {"embed_model": "nomic-embed-text", "dimension": 768}

    def count(self) -> int:
        return self._count


class FakeChromaClient:
    def __init__(
        self,
        heartbeat_ok: bool = True,
        collection: FakeCollection | None = None,
        fail_get_collection: bool = False,
    ) -> None:
        self.heartbeat_ok = heartbeat_ok
        self.collection = collection if collection is not None else FakeCollection()
        self.fail_get_collection = fail_get_collection

    def heartbeat(self) -> int:
        if not self.heartbeat_ok:
            raise ConnectionError("Chroma heartbeat connection refused")
        return 123456789

    def get_collection(self, name: str) -> FakeCollection:
        if self.fail_get_collection:
            raise ValueError(f"Collection '{name}' not found")
        return self.collection


class FakeRedisClient:
    def __init__(self, ping_ok: bool = True) -> None:
        self.ping_ok = ping_ok

    def ping(self) -> bool:
        if not self.ping_ok:
            raise ConnectionError("Redis connection refused")
        return True


# ---------------------------------------------------------------------------
# Fixture: client configured with RETRIEVER=stub (the CI overlay mode).
# Tests that exercise the Chroma path use check_readiness() directly with
# Settings(retriever="chroma") so the stub fast-path is not triggered.
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    from app.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Stub-retriever fast-path: /readyz skips Chroma, only checks Redis.
# This is what happens in the CI overlay (k8s/overlays/ci).
# ---------------------------------------------------------------------------

def test_readyz_stub_mode_ready_when_redis_up(client: TestClient) -> None:
    """With RETRIEVER=stub the API is ready as soon as Redis is reachable."""
    from app.main import app

    # Even a broken Chroma client must not block readiness in stub mode.
    app.state.chroma_client = FakeChromaClient(heartbeat_ok=False)
    app.state.redis_client = FakeRedisClient(ping_ok=True)

    r = client.get("/readyz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert "stub" in data["checks"]["retriever"]
    assert data["checks"]["redis"] == "ok"


def test_readyz_stub_mode_fails_when_redis_down(client: TestClient) -> None:
    """With RETRIEVER=stub readiness still fails when Redis is unreachable."""
    from app.main import app

    app.state.redis_client = FakeRedisClient(ping_ok=False)

    r = client.get("/readyz")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "not ready"
    assert data["failed"] == "redis"


# ---------------------------------------------------------------------------
# Chroma-retriever path: tested via check_readiness() directly with
# Settings(retriever="chroma") so the stub fast-path is not triggered.
# ---------------------------------------------------------------------------

def _chroma_settings(**kwargs: object) -> Settings:
    """Return Settings with retriever=chroma; override other fields as needed."""
    return Settings(retriever="chroma", embed_model="nomic-embed-text", **kwargs)  # type: ignore[arg-type]


def test_readyz_returns_200_when_all_ready(client: TestClient) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(
        heartbeat_ok=True,
        collection=FakeCollection(count_val=3, metadata={"embed_model": "nomic-embed-text", "dimension": 768}),
    )
    redis = FakeRedisClient(ping_ok=True)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "ready"
    checks = res["checks"]
    assert isinstance(checks, dict)
    assert checks["chroma"] == "ok"
    assert checks["redis"] == "ok"
    assert checks["count"] == 3
    assert checks["embed_model"] == "nomic-embed-text"


def test_readyz_fails_when_chroma_down(client: TestClient) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(heartbeat_ok=False)
    redis = FakeRedisClient(ping_ok=True)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "not ready"
    assert res["failed"] == "chroma"
    assert "detail" in res


def test_readyz_fails_when_collection_empty(client: TestClient) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(
        heartbeat_ok=True,
        collection=FakeCollection(count_val=0, metadata={"embed_model": "nomic-embed-text"}),
    )
    redis = FakeRedisClient(ping_ok=True)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "not ready"
    assert res["failed"] == "collection-empty"
    assert "empty" in str(res["detail"])


def test_readyz_fails_when_collection_missing(client: TestClient) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(heartbeat_ok=True, fail_get_collection=True)
    redis = FakeRedisClient(ping_ok=True)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "not ready"
    assert res["failed"] == "collection-empty"


def test_readyz_fails_when_redis_down(client: TestClient) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(
        heartbeat_ok=True,
        collection=FakeCollection(count_val=5, metadata={"embed_model": "nomic-embed-text"}),
    )
    redis = FakeRedisClient(ping_ok=False)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "not ready"
    assert res["failed"] == "redis"
    assert "detail" in res


def test_readyz_fails_on_embedding_mismatch_and_logs_error(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _chroma_settings()
    chroma = FakeChromaClient(
        heartbeat_ok=True,
        collection=FakeCollection(count_val=5, metadata={"embed_model": "dummy-768", "dimension": 768}),
    )
    redis = FakeRedisClient(ping_ok=True)

    with caplog.at_level(logging.ERROR):
        res = check_readiness(settings, chroma_client=chroma, redis_client=redis)

    assert res["status"] == "not ready"
    assert res["failed"] == "embedding-pin"
    assert "mismatch" in str(res["detail"])

    # Verify ONE JSON line / message at ERROR level
    error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_logs) == 1
    assert "embedding model mismatch" in error_logs[0].message
    assert "dummy-768" in error_logs[0].message
    assert "nomic-embed-text" in error_logs[0].message


def test_healthz_untouched_and_unaffected_by_dependencies(client: TestClient) -> None:
    from app.main import app

    # Dependencies are broken, but /healthz must still return 200 "alive"
    app.state.chroma_client = FakeChromaClient(heartbeat_ok=False)
    app.state.redis_client = FakeRedisClient(ping_ok=False)

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_direct_service_checks() -> None:
    settings = _chroma_settings()

    # Chroma down
    chroma = FakeChromaClient(heartbeat_ok=False)
    redis = FakeRedisClient(ping_ok=True)
    res = check_readiness(settings, chroma_client=chroma, redis_client=redis)
    assert res["status"] == "not ready"
    assert res["failed"] == "chroma"

    # Redis ping returns False
    class FalseRedis:
        def ping(self) -> bool:
            return False

    chroma = FakeChromaClient(heartbeat_ok=True, collection=FakeCollection(count_val=1))
    res = check_readiness(settings, chroma_client=chroma, redis_client=FalseRedis())
    assert res["status"] == "not ready"
    assert res["failed"] == "redis"

