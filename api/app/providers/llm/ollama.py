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


def _clean_json_content(content: str) -> str:
    """Strip think blocks, markdown fences, and extract JSON object."""
    text = content.strip()
    if "<think>" in text and "</think>" in text:
        text = text[text.find("</think>") + len("</think>") :].strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            last_fence = text.rfind("```")
            text = text[:last_fence]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
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
        url = f"{self.settings.ollama_url.rstrip('/')}/api/chat"

        example_structure = {
            "reasoning": "brief explanation from the passages",
            "answer": "clear direct answer",
            "citations": [
                {
                    "source": passages[0].source if passages else "doc.pdf",
                    "section": passages[0].section if passages else "Section",
                    "page": passages[0].page if passages else 1,
                }
            ],
            "answered_from_context": True,
            "confidence": "high",
        }

        citations_hint = "\n".join(
            f'  * {{"source": "{p.source}", "section": "{p.section}", "page": {p.page if p.page is not None else 1}}}'
            for p in passages[:3]
        ) if passages else ""

        content_prompt = (
            "You are RAHBAR, an academic guidance assistant for FAST-NUCES students.\n"
            "Answer the question using ONLY the provided passages.\n"
            "You MUST respond with a single valid JSON object strictly matching this schema structure:\n"
            f"{json.dumps(example_structure, indent=2)}\n"
            "CRITICAL RULES:\n"
            "1. Do NOT echo schema types or definitions. Provide real answer values in the JSON fields.\n"
            "2. When the answer is found in the passages, set 'answered_from_context': true and 'confidence': 'high'.\n"
            "3. For citations, use the exact source and section from the passages, such as:\n"
            f"{citations_hint}\n"
        )

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": content_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "options": {
                "temperature": self.settings.temperature,
            },
            "stream": False,
        }

        timeout = 180.0
        if self._client is not None:
            response = self._client.post(url, json=payload, timeout=timeout)
        else:
            response = httpx.post(url, json=payload, timeout=timeout)

        response.raise_for_status()
        data = response.json()
        raw_content = data.get("message", {}).get("content", "")
        cleaned = _clean_json_content(raw_content)

        return Answer.model_validate_json(cleaned)
