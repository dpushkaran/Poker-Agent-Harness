"""Hero equity against one or more opponent ranges.

Heads-up on the turn or river the result is exact (full enumeration over
the opponent's range and remaining board cards). Otherwise it is a Monte
Carlo estimate; ``stderr`` reports its sampling error.
"""

from __future__ import annotations

import math
import random
from itertools import accumulate

from phevaluator import evaluate_cards
from pydantic import BaseModel

from poker_agent.cards import Card, remaining_deck
from poker_agent.math.ranges import Combo, Range, remove_dead


class EquityError(ValueError):
    pass


class EquityResult(BaseModel):
    equity: float  # expected share of the pot at showdown (0..1)
    win: float  # outright win probability
    tie: float  # probability of a split
    samples: int
    method: str  # "exact" or "monte_carlo"
    stderr: float


def _showdown(hero: list[Card], opps: list[Combo], board: list[Card]) -> tuple[float, bool, bool]:
    """Return (hero's pot share, won outright, tied) for one runout. Lower rank wins."""
    h = evaluate_cards(*hero, *board)
    best_opp = min(evaluate_cards(*o, *board) for o in opps)
    if h < best_opp:
        return 1.0, True, False
    if h > best_opp:
        return 0.0, False, False
    winners = 1 + sum(1 for o in opps if evaluate_cards(*o, *board) == h)
    return 1.0 / winners, False, True


def equity(
    hero: list[Card],
    board: list[Card],
    ranges: list[Range],
    iterations: int = 20000,
    seed: int | None = None,
) -> EquityResult:
    if len(hero) != 2:
        raise EquityError("hero needs exactly two hole cards")
    if not ranges:
        raise EquityError("need at least one opponent range")
    dead = set(hero) | set(board)
    live = [remove_dead(r, dead) for r in ranges]
    if any(not r for r in live):
        raise EquityError("an opponent range is empty after removing known cards")

    if len(live) == 1 and len(board) >= 4:
        return _exact_heads_up(hero, board, live[0])
    return _monte_carlo(hero, board, live, iterations, random.Random(seed or None))


def _exact_heads_up(hero: list[Card], board: list[Card], rng: Range) -> EquityResult:
    total_w = share_sum = win_sum = tie_sum = 0.0
    n = 0
    for combo, w in rng.items():
        deck = remaining_deck([*hero, *board, *combo])
        runouts = [board + [c] for c in deck] if len(board) == 4 else [board]
        for full in runouts:
            share, won, tied = _showdown(hero, [combo], full)
            share_sum += share * w
            win_sum += won * w
            tie_sum += tied * w
            total_w += w
            n += 1
    return EquityResult(
        equity=share_sum / total_w,
        win=win_sum / total_w,
        tie=tie_sum / total_w,
        samples=n,
        method="exact",
        stderr=0.0,
    )


def _monte_carlo(
    hero: list[Card], board: list[Card], ranges: list[Range], iterations: int, rnd: random.Random
) -> EquityResult:
    tables = []
    for r in ranges:
        combos = list(r.keys())
        cum = list(accumulate(r.values()))
        tables.append((combos, cum))
    base_dead = set(hero) | set(board)
    deck_all = remaining_deck(base_dead)
    need = 5 - len(board)

    share_sum = share_sq = 0.0
    wins = ties = 0
    done = 0
    for _ in range(iterations):
        used = set(base_dead)
        opps: list[Combo] = []
        for combos, cum in tables:
            for _attempt in range(100):
                c = rnd.choices(combos, cum_weights=cum)[0]
                if c[0] not in used and c[1] not in used:
                    break
            else:
                break  # could not place this opponent; skip the iteration
            opps.append(c)
            used.update(c)
        if len(opps) != len(tables):
            continue
        deck = [c for c in deck_all if c not in used]
        runout = board + rnd.sample(deck, need)
        share, won, tied = _showdown(hero, opps, runout)
        share_sum += share
        share_sq += share * share
        wins += won
        ties += tied
        done += 1

    if done == 0:
        raise EquityError("could not deal any hands from the given ranges")
    mean = share_sum / done
    var = max(share_sq / done - mean * mean, 0.0)
    return EquityResult(
        equity=mean,
        win=wins / done,
        tie=ties / done,
        samples=done,
        method="monte_carlo",
        stderr=math.sqrt(var / done),
    )
