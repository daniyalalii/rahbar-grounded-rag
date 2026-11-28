"""Turns a question into an AskResponse. This is where the query pipeline lives.

Implements semantic caching via Redis, relevance floor gating (honest abstention),
prompt building with untrusted data delimiters, model response validation
with one retry, safe fallback, and citation validation against retrieved chunks.

RAGOps Part D: every query emits a full trace record via tracer.emit_trace().
"""

from __future__ import annotations

import logging
import re
import time

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..schemas import Answer, AskResponse, Citation, RetrievedRef
from .cache import SemanticCache
from .prompt import build_prompt
from .tracer import emit_trace, new_trace_id

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
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache: SemanticCache | None = None,
    is_followup: bool = False,
) -> AskResponse:
    """Execute the full RAG query pipeline for a question. Emits a trace record (Part D)."""
    trace_id = new_trace_id()
    t_start = time.perf_counter()
    latency_stages: dict[str, float] = {}
    guardrail_actions: list[dict] = []

    # 0. Semantic cache check
    cache_instance = cache or SemanticCache(settings)
    initial_gen = cache_instance.get_generation() if hasattr(cache_instance, "get_generation") else None
    question_vec: list[float] | None = None

    t_embed_start = time.perf_counter()
    if hasattr(retriever, "_embed"):
        try:
            question_vec = retriever._embed(question)  # type: ignore
        except Exception:
            question_vec = None
    latency_stages["embed_ms"] = round((time.perf_counter() - t_embed_start) * 1000, 1)

    if question_vec:
        cached_hit = cache_instance.check_hit(question_vec, is_followup=is_followup)
        if cached_hit is not None:
            resp = AskResponse(
                request_id=request_id_var.get(),
                answer=cached_hit.answer,
                citations=cached_hit.citations,
                answered_from_context=cached_hit.answered_from_context,
                confidence=cached_hit.confidence,
                retrieved=cached_hit.retrieved,
                provider="semantic-cache",
            )
            latency_stages["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            emit_trace(
                trace_id=trace_id,
                user_query=question,
                rewritten_query=None,
                embed_model=settings.embed_model,
                embed_version="v1",
                collection_name=settings.collection,
                retrieved_chunks=[],
                filters_applied={},
                reranked_order=[],
                prompt_text="(cache hit)",
                llm_model="semantic-cache",
                llm_params={},
                response_text=str(cached_hit.answer),
                citations=[c.dict() for c in (cached_hit.citations or [])],
                guardrail_actions=[],
                latency_stages=latency_stages,
                cache_hit=True,
            )
            return resp

    # 1. Retrieve
    t_retrieve_start = time.perf_counter()
    passages = retriever.retrieve(question, k=settings.top_k)
    latency_stages["retrieve_ms"] = round((time.perf_counter() - t_retrieve_start) * 1000, 1)

    # Guardrail: injection detection on query
    _query_injection_patterns = [
        "ignore all previous", "system override", "developer mode",
        "disregard all instructions", "forget everything",
    ]
    if any(p in question.lower() for p in _query_injection_patterns):
        guardrail_actions.append({"type": "injection", "action": "flagged", "source": "user_query"})

    # Guardrail: retrieval-time injection defence on retrieved chunks (Part D4)
    _chunk_injection_regexes = [
        re.compile(r"(?im)^.*\bignore\s+all\s+previous\s+instructions\b.*$"),
        re.compile(r"(?im)^.*\bsystem\s*:\s*ignore\b.*$"),
        re.compile(r"(?im)^.*\bdeveloper\s+mode\b.*$"),
        re.compile(r"(?im)^.*\bdisregard\s+all\s+(?:previous\s+|university\s+)?(?:instructions|policies|rules)\b.*$"),
        re.compile(r"(?im)^.*\bforget\s+(?:everything|all\s+previous)\b.*$"),
    ]
    for p in passages:
        sanitized_text = p.text
        matched_patterns: list[str] = []
        for rx in _chunk_injection_regexes:
            if rx.search(sanitized_text):
                matched_patterns.append(rx.pattern)
                sanitized_text = rx.sub("[REDACTED_INJECTION_ATTEMPT]", sanitized_text)
        if matched_patterns:
            p.text = sanitized_text
            guardrail_actions.append({
                "type": "injection",
                "action": "sanitized_and_flagged",
                "source": p.source,
                "chunk_id": p.chunk_id,
                "patterns_matched": len(matched_patterns),
            })
            log.warning(
                "retrieval_injection_detected",
                extra={"source": p.source, "chunk_id": p.chunk_id, "patterns": len(matched_patterns)},
            )

    # Relevance floor & honest abstention
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
        resp = AskResponse(
            request_id=request_id_var.get(),
            answer=ABSTENTION_MESSAGE,
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[
                RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                             page=p.page, score=p.score)
                for p in passages
            ],
            provider=provider.name,
        )
        latency_stages["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        emit_trace(
            trace_id=trace_id,
            user_query=question,
            rewritten_query=None,
            embed_model=settings.embed_model,
            embed_version="v1",
            collection_name=settings.collection,
            retrieved_chunks=[{"chunk_id": p.chunk_id, "doc_id": p.source,
                               "doc_version": getattr(p, "doc_version", "unknown"),
                               "score": p.score} for p in passages],
            filters_applied={"relevance_floor": settings.relevance_floor},
            reranked_order=[],
            prompt_text="(abstained: below relevance floor)",
            llm_model=provider.name,
            llm_params={"temperature": settings.temperature},
            response_text=ABSTENTION_MESSAGE,
            citations=[],
            guardrail_actions=guardrail_actions,
            latency_stages=latency_stages,
            cache_hit=False,
        )
        return resp

    relevant_passages = [p for p in passages if p.score >= settings.relevance_floor]

    # 2. Build prompt
    prompt_text = build_prompt(question, relevant_passages)

    # 3. Call model
    t_gen_start = time.perf_counter()
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
    latency_stages["generate_ms"] = round((time.perf_counter() - t_gen_start) * 1000, 1)

    if answer is None:
        answer = SAFE_CANNED_ANSWER

    # 4. Citation validation
    retrieved_pairs = {(p.source.strip().lower(), p.section.strip().lower()) for p in passages}
    valid_citations: list[Citation] = []
    for c in answer.citations:
        if (c.source.strip().lower(), c.section.strip().lower()) in retrieved_pairs:
            valid_citations.append(c)
        else:
            # Flexible match: if the source matches a retrieved passage and section or page aligns
            for p in passages:
                if c.source.strip().lower() == p.source.strip().lower():
                    c_sec = c.section.strip().lower()
                    p_sec = p.section.strip().lower()
                    if c_sec in p_sec or p_sec in c_sec or (c.page is not None and c.page == p.page):
                        valid_citations.append(Citation(source=p.source, section=p.section, page=c.page or p.page))
                        break

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

    response = AskResponse(
        request_id=request_id_var.get(),
        answer=answer.answer,
        citations=valid_citations,
        answered_from_context=answered_from_context,
        confidence=answer.confidence,
        retrieved=[
            RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                         page=p.page, score=p.score)
            for p in passages
        ],
        provider=provider.name,
    )

    # 5. Semantic cache store
    if question_vec:
        cache_instance.store(
            question, question_vec, response, is_followup=is_followup, generation=initial_gen
        )

    # 6. Emit trace record (Part D1)
    latency_stages["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    emit_trace(
        trace_id=trace_id,
        user_query=question,
        rewritten_query=None,
        embed_model=settings.embed_model,
        embed_version="v1",
        collection_name=settings.collection,
        retrieved_chunks=[
            {
                "chunk_id": p.chunk_id,
                "doc_id": p.source,
                "doc_version": getattr(p, "doc_version", "unknown"),
                "section": p.section,
                "page": p.page,
                "score": round(p.score, 4),
                "text": p.text,
            }
            for p in passages
        ],
        filters_applied={"relevance_floor": settings.relevance_floor, "top_k": settings.top_k},
        reranked_order=[p.chunk_id for p in passages],
        prompt_text=str(prompt_text) if prompt_text else "",
        llm_model=provider.name,
        llm_params={"temperature": settings.temperature},
        response_text=str(response.answer),
        citations=[c.model_dump() if hasattr(c, "model_dump") else vars(c) for c in valid_citations],
        guardrail_actions=guardrail_actions,
        latency_stages=latency_stages,
        cache_hit=False,
    )

    return response
