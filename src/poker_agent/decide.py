"""The decision pipeline: analyze -> prompt -> LLM -> validate -> recommendation.

If the model is unreachable or keeps producing invalid moves, the
deterministic baseline's decision is returned and marked as a fallback.
"""

from __future__ import annotations

import time
from typing import Literal, Protocol

from pydantic import BaseModel

from poker_agent.analysis import Analysis, analyze
from poker_agent.baseline import baseline_decision
from poker_agent.config import Settings
from poker_agent.decision import Decision
from poker_agent.llm.client import ChatResult, LLMError, OllamaClient
from poker_agent.llm.prompt import build_messages, legal_lines
from poker_agent.llm.schema import decision_schema
from poker_agent.state import GameState
from poker_agent.validate import validate_decision


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[dict], schema: dict | None = ..., temperature: float = ...,
             think: bool = ..., seed: int | None = ..., max_tokens: int | None = ...) -> ChatResult: ...


class Recommendation(BaseModel):
    decision: Decision
    source: Literal["llm", "baseline", "fallback"]
    baseline: Decision
    analysis: Analysis
    model: str | None = None
    prompt_version: str | None = None
    attempts: int = 0
    valid_first_try: bool | None = None
    raw_outputs: list[str] = []
    adjustments: list[str] = []
    errors: list[str] = []
    llm_latency_s: float = 0.0
    total_latency_s: float = 0.0


def make_client(settings: Settings) -> OllamaClient:
    c = settings.llm
    return OllamaClient(c.base_url, c.model, c.timeout_seconds)


def decide(
    state: GameState,
    settings: Settings,
    client: ChatClient | None = None,
    use_llm: bool = True,
    seed: int | None = None,
) -> Recommendation:
    start = time.perf_counter()
    analysis = analyze(state, settings.equity)
    base = baseline_decision(state, analysis)
    rec = Recommendation(decision=base, source="baseline", baseline=base, analysis=analysis)
    if not use_llm:
        rec.total_latency_s = time.perf_counter() - start
        return rec

    cfg = settings.llm
    client = client or make_client(settings)
    rec.model = client.model
    rec.prompt_version = cfg.prompt_version
    messages = build_messages(state, analysis, cfg.prompt_version, settings.game.buy_in)
    schema = decision_schema(analysis.legal if cfg.constrain_actions else None)

    for attempt in range(1 + cfg.max_retries):
        rec.attempts = attempt + 1
        try:
            reply = client.chat(messages, schema=schema, temperature=cfg.temperature,
                                think=cfg.think, seed=seed,
                                max_tokens=cfg.max_tokens * (6 if cfg.think else 1))
        except LLMError as e:
            rec.errors.append(str(e))
            break
        rec.llm_latency_s += reply.latency_s
        rec.raw_outputs.append(reply.content)
        result = validate_decision(reply.content, analysis.legal, settings.game.chip_increment)
        if rec.valid_first_try is None:
            rec.valid_first_try = result.ok
        if result.ok:
            rec.decision = result.decision
            rec.source = "llm"
            rec.adjustments = result.adjustments
            rec.total_latency_s = time.perf_counter() - start
            return rec
        rec.errors.extend(result.errors)
        messages = messages + [
            {"role": "assistant", "content": reply.content},
            {"role": "user", "content": "That answer is invalid: " + "; ".join(result.errors)
             + ".\nLEGAL ACTIONS:\n" + "\n".join(legal_lines(analysis)) + "\nAnswer again."},
        ]

    rec.decision = base
    rec.source = "fallback"
    rec.total_latency_s = time.perf_counter() - start
    return rec
