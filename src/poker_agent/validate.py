"""Check a model's decision against the legal actions before trusting it.

Small, unambiguous slips are repaired and recorded as adjustments (e.g.
"bet" when the move is technically a raise, a size a few cents off the
chip increment). Anything else is an error, which triggers a retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from poker_agent.decision import Decision, Move
from poker_agent.rules import LegalActions

CLAMP_TOLERANCE = 0.10  # clamp sizes within 10% outside the legal range; reject beyond


@dataclass
class ValidationResult:
    decision: Decision | None
    errors: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.decision is not None and not self.errors


def parse_decision(raw: str | dict) -> tuple[Decision | None, str | None]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Decision.model_validate(data), None
    except json.JSONDecodeError as e:
        return None, f"reply was not valid JSON ({e.msg})"
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        return None, f"reply did not match the schema: {loc}: {first['msg']}"


def validate_decision(
    raw: str | dict, legal: LegalActions, increment: float | None = None
) -> ValidationResult:
    d, err = parse_decision(raw)
    if d is None:
        return ValidationResult(None, errors=[err])
    res = ValidationResult(d)
    names = legal.names()
    action = d.action

    # Unambiguous vocabulary slips.
    if action is Move.BET and not legal.can_bet and legal.can_raise:
        action = Move.RAISE
        res.adjustments.append("model said 'bet' facing a bet; treated as a raise")
    elif action is Move.RAISE and not legal.can_raise and legal.can_bet:
        action = Move.BET
        res.adjustments.append("model said 'raise' with no bet to raise; treated as a bet")
    elif action is Move.CALL and legal.can_check:
        action = Move.CHECK
        res.adjustments.append("model said 'call' with nothing to call; treated as a check")

    if action.value not in names:
        res.errors.append(f"'{action.value}' is not legal here; legal actions are {', '.join(names)}")
        return res

    amount = None
    if action in (Move.BET, Move.RAISE):
        if d.amount is None:
            res.errors.append(f"{action.value} needs an amount between {legal.min_to:.2f} and "
                              f"{legal.max_to:.2f}")
            return res
        lo, hi = legal.min_to, legal.max_to
        if d.amount < lo * (1 - CLAMP_TOLERANCE) or d.amount > hi * (1 + CLAMP_TOLERANCE):
            res.errors.append(f"{action.value} amount {d.amount:.2f} is outside the legal range "
                              f"{lo:.2f}-{hi:.2f} (amount is the total 'to' figure)")
            return res
        amount = legal.snap(d.amount, increment)
        if abs(amount - d.amount) > 1e-9:
            res.adjustments.append(f"amount {d.amount:.2f} adjusted to legal {amount:.2f}")
        if amount >= hi:
            action, amount = Move.ALL_IN, None
            res.adjustments.append("size equals the whole stack; recorded as all-in")

    res.decision = d.model_copy(update={"action": action, "amount": amount})
    return res
