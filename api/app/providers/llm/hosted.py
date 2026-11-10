"""Hosted free-tier provider (Groq, Google AI Studio, OpenRouter, OpenAI-compatible).

The key arrives as LLM_API_KEY from a Kubernetes Secret created on the command
line. It never appears in a file you commit, and never in a log line.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from ...services.prompt import build_prompt


class HostedProvider:
    name = "hosted"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=100.0)

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Call OpenAI-compatible chat completions endpoint."""
        prompt = build_prompt(question, passages)
        url = (self.settings.hosted_base_url.rstrip("/") or "https://api.groq.com/openai/v1") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are RAHBAR. You answer university questions strictly from retrieved passages. Output valid JSON matching the Answer schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
        }
        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return Answer.model_validate_json(content)
