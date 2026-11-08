"""The real retriever. (sections 3.2 and 4.4).

Queries the Chroma vector store after embedding the question via Ollama /api/embed.
Applies a metadata filter to exclude superseded documents unless the question explicitly
mentions an older year or version.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ..config import Settings
from ..schemas import Passage

log = logging.getLogger("rahbar.retriever.chroma")

_OLDER_KEYWORDS = [
    "superseded",
    "older",
    "previous",
    "former",
    "past",
    "prior",
    "old rule",
    "old rules",
    "old version",
    "old edition",
    "earlier",
]


def mentions_older_version(question: str) -> bool:
    """Detect whether a question is asking about a superseded/historical policy."""
    q = question.lower()
    if any(k in q for k in _OLDER_KEYWORDS):
        return True
    # 4-digit years before current policy year (2026)
    years = re.findall(r"\b(19\d\d|20[0-2]\d)\b", q)
    for y_str in years:
        if int(y_str) < 2026:
            return True
    return False


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
        else:
            import chromadb  # the thin client, from chromadb-client in requirements.txt
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def _embed(self, question: str) -> list[float]:
        """Embed the question with Ollama /api/embed using EMBED_MODEL."""
        url = f"{self.settings.ollama_url.rstrip('/')}/api/embed"
        response = httpx.post(
            url,
            json={"model": self.settings.embed_model, "input": [question]},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError(f"No embeddings returned from {url}")
        return embeddings[0]

    def retrieve(self, question: str, k: int) -> list[Passage]:
        """Embed question, query Chroma top k, filter superseded unless older requested."""
        vector = self._embed(question)

        try:
            collection = self.client.get_collection(name=self.settings.collection)
        except Exception as exc:
            log.warning("Failed to get collection %r: %s", self.settings.collection, exc)
            return []

        try:
            count = collection.count()
        except Exception as exc:
            log.warning("Failed to get collection count: %s", exc)
            return []

        if count == 0:
            return []

        top_k = k if k > 0 else self.settings.top_k
        limit = min(top_k, count)

        include_superseded = mentions_older_version(question)
        where_filter: dict[str, Any] | None = None if include_superseded else {"superseded": False}

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        try:
            results = collection.query(**query_kwargs)
        except Exception as exc:
            log.warning("Chroma query failed with filter %s: %s", where_filter, exc)
            # If query failed (e.g., no documents matched where filter), try without where or return []
            return []

        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        passages: list[Passage] = []
        for i in range(len(ids)):
            doc_id = ids[i]
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            dist = distances[i] if i < len(distances) else 0.0

            # Cosine similarity: score = 1 - distance
            score = 1.0 - float(dist)

            source = str(meta.get("source", ""))
            section = str(meta.get("section", ""))
            raw_page = meta.get("page")
            page_val: int | None = None
            if raw_page is not None:
                try:
                    page_val = int(raw_page)
                except (ValueError, TypeError):
                    page_val = None

            passages.append(
                Passage(
                    chunk_id=doc_id,
                    text=text,
                    source=source,
                    section=section,
                    page=page_val,
                    score=score,
                )
            )

        return passages
