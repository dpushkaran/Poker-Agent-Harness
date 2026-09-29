"""The decision format shared by the LLM, the baseline policy and the UI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Move(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(BaseModel):
    action: Move
    amount: float | None = Field(
        default=None,
        description="For bet/raise: total chips committed on this street after the action "
        "('raise to'), in dollars. Null for fold/check/call/all_in.",
    )
    sizing_rationale: str | None = Field(default=None, description="Why this size, if betting")
    reasoning: str = Field(description="2-4 sentences explaining the decision")
    key_factors: list[str] = Field(default_factory=list, max_length=5)
    confidence: Confidence = Confidence.MEDIUM
