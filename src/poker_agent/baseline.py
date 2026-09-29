"""A deterministic rule-based policy.

It is intentionally simple: preflop charts by position, and postflop
decisions driven by equity against the estimated ranges versus pot odds.
It is the fallback when the LLM fails and the benchmark the LLM is
compared with in evals.
"""

from __future__ import annotations

from functools import lru_cache

from poker_agent.analysis import Analysis
from poker_agent.decision import Confidence, Decision, Move
from poker_agent.math.ranges import (
    BB_DEFEND_RANGE, CALL_RANGE, OPEN_RANGES, combo, parse_range,
)
from poker_agent.state import ActionType, GameState, Street

PREMIUM = "QQ+, AKs, AKo"
# Isolation-raise ranges shrink as more players limp in (index = limpers, capped).
ISO_RANGES = {
    1: "66+, A9s+, KTs+, QJs, AJo+, KQo",
    2: "88+, AJs+, KQs, AQo+",
}
STRONG = "TT+, AQs+, AKo"
LATE = ("CO", "BTN", "SB")
ALL_IN_THRESHOLD = 0.7  # shove instead of betting more than this share of the stack


@lru_cache(maxsize=64)
def _range(text: str) -> frozenset:
    return frozenset(parse_range(text))


def _in(state: GameState, text: str) -> bool:
    a, b = state.hole_card_ids
    return combo(a, b) in _range(text)


def _aggressive(a: Analysis, to: float, why: str, factors: list[str], conf=Confidence.MEDIUM) -> Decision:
    legal = a.legal
    to = legal.snap(to)
    if to >= legal.max_to * ALL_IN_THRESHOLD:
        return Decision(action=Move.ALL_IN, reasoning=why + " Sizing would commit most of the stack, so shove.",
                        key_factors=factors, confidence=conf)
    move = Move.BET if legal.can_bet else Move.RAISE
    return Decision(action=move, amount=to, reasoning=why, key_factors=factors, confidence=conf,
                    sizing_rationale=f"{move.value} to {to:.2f}")


def _passive(a: Analysis, why: str, factors: list[str], conf=Confidence.MEDIUM) -> Decision:
    if a.legal.can_check:
        return Decision(action=Move.CHECK, reasoning=why, key_factors=factors, confidence=conf)
    return Decision(action=Move.CALL, reasoning=why, key_factors=factors, confidence=conf)


def _give_up(a: Analysis, why: str, factors: list[str], conf=Confidence.MEDIUM) -> Decision:
    if a.legal.can_check:
        return Decision(action=Move.CHECK, reasoning=why, key_factors=factors, confidence=conf)
    return Decision(action=Move.FOLD, reasoning=why, key_factors=factors, confidence=conf)


def _preflop(state: GameState, a: Analysis) -> Decision:
    pos = a.hero_position
    bb = a.big_blind
    pre = [x for x in state.actions if x.street is Street.PREFLOP]
    raises = sum(1 for x in pre if x.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN))
    limpers = sum(1 for x in pre if x.type is ActionType.CALL) if raises == 0 else 0
    can_raise = a.legal.can_raise
    facts = [f"{pos}", f"{a.strength.description}", f"{a.hero_stack_bb:.0f}bb behind"]

    if raises == 0:
        open_range = OPEN_RANGES.get(pos, OPEN_RANGES["BTN"])
        if limpers:
            open_range = ISO_RANGES[min(limpers, 2)]
        if can_raise and _in(state, open_range):
            size = (3 + limpers) * bb
            why = f"Hand is in the {pos} opening range" + (f"; isolate {limpers} limper(s)." if limpers else ".")
            return _aggressive(a, size, why, facts)
        if a.legal.can_check:
            return _passive(a, "Take the free flop from the big blind.", facts)
        if limpers and pos in LATE and _in(state, "22+, A2s+, K9s+, QTs+, J9s+, T8s+, 97s+, 86s+, 76s, 65s"):
            return _passive(a, "Speculative hand with good implied odds in a limped pot.", facts, Confidence.LOW)
        return _give_up(a, f"Hand is outside the {pos} range.", facts)

    if raises == 1:
        if can_raise and _in(state, PREMIUM):
            to = a.legal.call_amount + a.legal.hero_committed_street
            return _aggressive(a, to * (3 if pos in LATE else 3.5), "Premium hand: 3-bet for value.", facts, Confidence.HIGH)
        defend = BB_DEFEND_RANGE if pos == "BB" else CALL_RANGE
        if _in(state, defend) and a.equity.equity >= (a.odds.pot_odds or 0):
            return _passive(a, "Hand is strong enough to continue and the price is right.", facts)
        return _give_up(a, "Not strong enough to continue against a raise.", facts)

    if _in(state, PREMIUM):
        if can_raise:
            return _aggressive(a, a.legal.max_to, "Premium hand facing a re-raise: get it in.", facts, Confidence.HIGH)
        return _passive(a, "Premium hand facing a re-raise: call.", facts, Confidence.HIGH)
    if _in(state, STRONG) and a.equity.equity >= (a.odds.pot_odds or 0):
        return _passive(a, "Strong hand with enough equity to call the re-raise.", facts, Confidence.LOW)
    return _give_up(a, "Facing multiple raises without a premium hand.", facts)


def _postflop(a: Analysis) -> Decision:
    eq = a.equity.equity
    po = a.odds.pot_odds or 0.0
    draw = bool([d for d in a.strength.draws if "backdoor" not in d])
    heads_up = a.players_in_hand == 2
    facts = [a.strength.description, f"equity {eq:.0%}", f"pot odds {po:.0%}" if po else "no bet to face"]
    if draw:
        facts.append(", ".join(a.strength.draws))

    if a.legal.can_check:
        if a.legal.can_bet and eq >= 0.65:
            return _aggressive(a, a.odds.pot * 0.66, "Strong hand against the estimated ranges: bet for value.", facts)
        if a.legal.can_bet and draw and heads_up:
            return _aggressive(a, a.odds.pot * 0.5, "Semi-bluff a strong draw heads-up.", facts, Confidence.LOW)
        return _passive(a, "Not strong enough to bet for value; check.", facts)

    implied = 0.05 if draw and a.odds.spr > 3 and a.street is not Street.RIVER else 0.0
    if a.legal.can_raise and eq >= 0.72:
        current = a.legal.call_amount + a.legal.hero_committed_street
        return _aggressive(a, current * 3, "Well ahead of the betting range: raise for value.", facts, Confidence.HIGH)
    if eq + implied >= po:
        why = "Equity beats the price" + (" once implied odds are included." if implied and eq < po else ".")
        return _passive(a, why, facts)
    return _give_up(a, f"Equity {eq:.0%} is below the {po:.0%} needed to call.", facts)


def baseline_decision(state: GameState, analysis: Analysis) -> Decision:
    if analysis.street is Street.PREFLOP:
        return _preflop(state, analysis)
    return _postflop(analysis)
