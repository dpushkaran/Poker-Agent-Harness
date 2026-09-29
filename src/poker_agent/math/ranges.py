"""Hand-range notation and default opponent ranges.

A range is a mapping of two-card combos to weights in (0, 1]. The parser
understands the usual shorthand::

    QQ+  99-66  AKs  AKo  AK  ATs+  A5s-A2s  T9s-65s  KQo:0.5

Default ranges are deliberately looser than standard online ranges since
this is a home game; any of them can be overridden per opponent.
"""

from __future__ import annotations

import re
from itertools import combinations

from poker_agent.cards import RANKS, Card, rank_of, suit_of
from poker_agent.state import ActionType, GameState, Street

Combo = tuple[Card, Card]  # (higher id, lower id)
Range = dict[Combo, float]


class RangeError(ValueError):
    pass


def combo(a: Card, b: Card) -> Combo:
    return (a, b) if a > b else (b, a)


def _class_combos(hi: int, lo: int, kind: str | None) -> list[Combo]:
    """Combos for one hand class, e.g. hi=12, lo=11, kind='s' -> AKs."""
    if hi == lo:
        cards = [hi * 4 + s for s in range(4)]
        return [combo(a, b) for a, b in combinations(cards, 2)]
    out = []
    for s1 in range(4):
        for s2 in range(4):
            suited = s1 == s2
            if (kind == "s" and not suited) or (kind == "o" and suited):
                continue
            out.append(combo(hi * 4 + s1, lo * 4 + s2))
    return out


_TOKEN = re.compile(
    r"^(?P<a>[2-9TJQKA])(?P<b>[2-9TJQKA])(?P<k>[so])?"
    r"(?:(?P<plus>\+)|-(?P<c>[2-9TJQKA])(?P<d>[2-9TJQKA])(?P<k2>[so])?)?$"
)


def _expand_token(tok: str) -> list[tuple[int, int, str | None]]:
    m = _TOKEN.match(tok)
    if not m:
        raise RangeError(f"cannot parse range token {tok!r}")
    a, b = RANKS.index(m["a"]), RANKS.index(m["b"])
    kind = m["k"]
    hi, lo = max(a, b), min(a, b)
    pair = hi == lo
    if pair and kind:
        raise RangeError(f"pairs cannot be suited/offsuit: {tok!r}")

    if m["plus"]:
        if pair:
            return [(r, r, None) for r in range(hi, 13)]
        return [(hi, k, kind) for k in range(lo, hi)]

    if m["c"]:
        c, d = RANKS.index(m["c"]), RANKS.index(m["d"])
        hi2, lo2 = max(c, d), min(c, d)
        if (m["k2"] or None) != kind:
            raise RangeError(f"mismatched suitedness in {tok!r}")
        if pair:
            if hi2 != lo2:
                raise RangeError(f"pair range must end in a pair: {tok!r}")
            top, bot = max(hi, hi2), min(hi, hi2)
            return [(r, r, None) for r in range(bot, top + 1)]
        if hi == hi2:  # A5s-A2s: fixed top card, kicker range
            top, bot = max(lo, lo2), min(lo, lo2)
            return [(hi, k, kind) for k in range(bot, top + 1)]
        if hi - lo == hi2 - lo2:  # T9s-65s: slide both cards down together
            gap = hi - lo
            top, bot = max(hi, hi2), min(hi, hi2)
            return [(h, h - gap, kind) for h in range(bot, top + 1)]
        raise RangeError(f"unsupported range span {tok!r}")

    return [(hi, lo, kind)]


def parse_range(text: str) -> Range:
    """Parse range notation into combo weights. ``"any"`` means all 1326 combos."""
    if text.strip().lower() in ("any", "random", "100%"):
        return {combo(a, b): 1.0 for a, b in combinations(range(52), 2)}
    out: Range = {}
    for raw in text.split(","):
        tok = raw.strip()
        if not tok:
            continue
        weight = 1.0
        if ":" in tok:
            tok, w = tok.split(":", 1)
            weight = float(w)
            if not 0 < weight <= 1:
                raise RangeError(f"weight must be in (0, 1]: {raw!r}")
        for hi, lo, kind in _expand_token(tok.strip()):
            for c in _class_combos(hi, lo, kind):
                out[c] = weight
    if not out:
        raise RangeError("empty range")
    return out


def hand_class(a: Card, b: Card) -> str:
    """Canonical 169-class label for two cards, e.g. 'AKs', 'T9o', '77'."""
    ra, rb = rank_of(a), rank_of(b)
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if suit_of(a) == suit_of(b) else "o")


def remove_dead(rng: Range, dead: set[Card]) -> Range:
    """Drop combos that use a known card (hero's hole cards or the board)."""
    return {c: w for c, w in rng.items() if c[0] not in dead and c[1] not in dead}


def range_size(rng: Range) -> float:
    return sum(rng.values())


# --- Default ranges ---------------------------------------------------------

OPEN_RANGES: dict[str, str] = {
    "UTG": "55+, A8s+, A5s-A4s, K9s+, Q9s+, J9s+, T9s, 98s, ATo+, KJo+",
    "UTG+1": "55+, A7s+, A5s-A3s, K9s+, Q9s+, J9s+, T9s, 98s, ATo+, KJo+",
    "MP": "44+, A7s+, A5s-A2s, K8s+, Q9s+, J9s+, T8s+, 98s, 87s, ATo+, KJo+, QJo",
    "LJ": "33+, A2s+, K8s+, Q9s+, J8s+, T8s+, 97s+, 87s, 76s, ATo+, KTo+, QJo",
    "HJ": "33+, A2s+, K7s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, A9o+, KTo+, QTo+, JTo",
    "CO": "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, "
    "A7o+, K9o+, Q9o+, J9o+, T9o",
    "BTN": "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 95s+, 85s+, 74s+, 64s+, 53s+, 43s, "
    "A2o+, K7o+, Q8o+, J8o+, T8o+, 98o, 87o",
    "SB": "22+, A2s+, K4s+, Q6s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, "
    "A5o+, K9o+, Q9o+, J9o+, T9o",
}
LIMP_RANGE = (
    "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 85s+, 75s+, 64s+, 54s, "
    "A2o+, K8o+, Q9o+, J9o+, T9o, 98o, QQ+:0.3, AKs:0.3, AKo:0.3"
)
CALL_RANGE = (
    "22-JJ, QQ+:0.4, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, "
    "ATo+, KTo+, QTo+, JTo, AKo:0.5"
)
BB_DEFEND_RANGE = (
    "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 85s+, 74s+, 63s+, 53s+, 43s, "
    "A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o"
)
THREE_BET_RANGE = "TT+, AQs+, AKo, AJs:0.5, KQs:0.5, A5s:0.5, AQo:0.5"
CALL_THREE_BET_RANGE = "77-QQ, AJs+, KQs, QJs:0.5, JTs:0.5, AQo+, KK+:0.3"
FOUR_BET_RANGE = "QQ+, AKs, AKo"
ANY_RANGE = "any"


def infer_preflop_range(state: GameState, seat: int) -> tuple[str, str]:
    """Estimate a seat's preflop range from their actions.

    Returns (label, range_notation). Seats that have not voluntarily acted
    (e.g. a big blind who checked) get an unrestricted range.
    """
    positions = state.positions
    pos = positions[seat]
    raises = 0
    label, rng = "no voluntary action", ANY_RANGE
    for a in state.actions:
        if a.street is not Street.PREFLOP:
            break
        aggressive = a.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)
        if a.seat == seat:
            if aggressive:
                if raises == 0:
                    label, rng = f"{pos} open-raise", OPEN_RANGES.get(pos, OPEN_RANGES["BTN"])
                elif raises == 1:
                    label, rng = f"{pos} 3-bet", THREE_BET_RANGE
                else:
                    label, rng = f"{pos} 4-bet+", FOUR_BET_RANGE
            elif a.type is ActionType.CALL:
                if raises == 0:
                    label, rng = f"{pos} limp", LIMP_RANGE
                elif raises == 1:
                    if pos == "BB":
                        label, rng = "BB defend vs raise", BB_DEFEND_RANGE
                    else:
                        label, rng = f"{pos} call vs raise", CALL_RANGE
                else:
                    label, rng = f"{pos} call vs 3-bet+", CALL_THREE_BET_RANGE
            elif a.type is ActionType.CHECK:
                label, rng = f"{pos} checked option", ANY_RANGE
        if aggressive:
            raises += 1
    return label, rng
