"""JSON schema for the model's reply, optionally narrowed to the legal moves."""

from __future__ import annotations

from poker_agent.decision import Decision
from poker_agent.rules import LegalActions


def decision_schema(legal: LegalActions | None = None) -> dict:
    """Schema passed to Ollama's ``format`` so decoding is constrained to it.

    With ``legal`` given, the action enum only lists moves that are legal in
    this spot (Ollama cannot express the bet-size bounds reliably, so sizes
    are still checked by the validator).
    """
    schema = Decision.model_json_schema()
    props = schema["properties"]
    # Inline the enums so the schema has no $refs, which some runtimes ignore.
    defs = schema.pop("$defs", {})
    props["action"] = {"type": "string", "enum": [m for m in defs["Move"]["enum"]]}
    props["confidence"] = {"type": "string", "enum": defs["Confidence"]["enum"]}
    if legal is not None:
        props["action"]["enum"] = legal.names()
    # Generation follows property order: reason first, then commit to an action.
    order = ["reasoning", "key_factors", "action", "amount", "sizing_rationale", "confidence"]
    schema["properties"] = {k: props[k] for k in order}
    schema["required"] = order
    return schema
