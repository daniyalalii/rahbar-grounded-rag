"""Turns a question into an AskResponse. This is where the query pipeline lives.

Implements relevance floor gating (honest abstention without calling LLM),
prompt building with untrusted data delimiters, model response validation
with one retry, safe fallback, and citation validation against retrieved chunks.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..schemas import Answer, AskResponse, Citation, RetrievedRef
from .prompt import build_prompt

log = logging.getLogger("rahbar.answer")

SAFE_CANNED_ANSWER = Answer(
    reasoning="Fallback: model response failed validation after retry",
    answer="I am unable to answer this question at the moment. Please consult the official university regulations.",
    citations=[],
    answered_from_context=False,
    confidence="low",
)

ABSTENTION_MESSAGE = "The corpus does not cover this."


def answer_question(
    question: str, retriever: Retriever, provider: LLMProvider, settings: Settings
) -> AskResponse:
    """Execute the full RAG query pipeline for a question."""
    passages = retriever.retrieve(question, k=settings.top_k)

    # 1. Relevance floor & honest abstention without calling the LLM
    has_relevant = any(p.score >= settings.relevance_floor for p in passages)
    if not passages or not has_relevant:
        log.info(
            "abstention_below_relevance_floor",
            extra={
                "question": question,
                "count": len(passages),
                "max_score": max((p.score for p in passages), default=0.0),
                "relevance_floor": settings.relevance_floor,
            },
        )
        return AskResponse(
            request_id=request_id_var.get(),
            answer=ABSTENTION_MESSAGE,
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[
                RetrievedRef(
                    chunk_id=p.chunk_id,
                    source=p.source,
                    section=p.section,
                    page=p.page,
                    score=p.score,
                )
                for p in passages
            ],
            provider=provider.name,
        )

    # Filter to passages meeting or exceeding the relevance floor
    relevant_passages = [p for p in passages if p.score >= settings.relevance_floor]

    # 2. Build prompt (also done inside OllamaProvider if called)
    _ = build_prompt(question, relevant_passages)

    # 3. Call model with validation and one retry
    answer: Answer | None = None
    for attempt in range(2):
        try:
            raw = provider.generate(question, relevant_passages)
            if isinstance(raw, Answer):
                answer = raw
            else:
                answer = Answer.model_validate(raw)
            break
        except Exception as exc:
            log.warning("LLM generation/validation attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                answer = SAFE_CANNED_ANSWER

    if answer is None:
        answer = SAFE_CANNED_ANSWER

    # 4. Drop any citation that does not match a retrieved chunk
    retrieved_pairs = {(p.source.strip().lower(), p.section.strip().lower()) for p in passages}
    valid_citations: list[Citation] = []
    for c in answer.citations:
        if (c.source.strip().lower(), c.section.strip().lower()) in retrieved_pairs:
            valid_citations.append(c)

    # If all citations were dropped, decline the answer
    answered_from_context = answer.answered_from_context
    if answered_from_context and not valid_citations:
        answered_from_context = False

    log.info(
        "answered",
        extra={
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "answered_from_context": answered_from_context,
            "citations_kept": len(valid_citations),
            "citations_dropped": len(answer.citations) - len(valid_citations),
        },
    )

    return AskResponse(
        request_id=request_id_var.get(),
        answer=answer.answer,
        citations=valid_citations,
        answered_from_context=answered_from_context,
        confidence=answer.confidence,
        retrieved=[
            RetrievedRef(
                chunk_id=p.chunk_id,
                source=p.source,
                section=p.section,
                page=p.page,
                score=p.score,
            )
            for p in passages
        ],
        provider=provider.name,
    )
