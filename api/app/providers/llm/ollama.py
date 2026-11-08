"""Ollama provider: ollama-host or ollama-incluster (section 3.5).

Calls the Ollama /api/chat endpoint with format=json and the Answer schema,
using OLLAMA_URL and LLM_MODEL with temperature from settings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from ...services.prompt import build_prompt

log = logging.getLogger("rahbar.providers.ollama")

_TIMEOUT = 100.0


def _clean_json_content(content: str) -> str:
    """Strip markdown code fence wrapper if present."""
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            last_fence = text.rfind("```")
            text = text[:last_fence]
    return text.strip()


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.name = settings.llm_provider
        self._client = client

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Call Ollama /api/chat with format=json and the Answer schema."""
        prompt = build_prompt(question, passages)
        schema = Answer.model_json_schema()
        url = f"{self.settings.ollama_url.rstrip('/')}/api/chat"

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are RAHBAR, an academic guidance assistant. "
                        "You MUST respond with a valid JSON object strictly matching this schema:\n"
                        f"{json.dumps(schema)}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": 400,
            },
            "stream": False,
        }

        if self._client is not None:
            response = self._client.post(url, json=payload, timeout=_TIMEOUT)
        else:
            response = httpx.post(url, json=payload, timeout=_TIMEOUT)

        response.raise_for_status()
        data = response.json()
        raw_content = data.get("message", {}).get("content", "")
        cleaned = _clean_json_content(raw_content)

        return Answer.model_validate_json(cleaned)
