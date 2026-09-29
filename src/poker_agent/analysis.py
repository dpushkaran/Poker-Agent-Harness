"""Bundle every deterministic number about a decision point.

``analyze`` replays the hand, works out the hero's legal actions, pot
odds, hand strength, opponents' estimated ranges and hero's equity
against them. The result is what the LLM reasons over and what the UI
shows next to the recommendation.
"""

from __future__ import annotations

from pydantic import BaseModel

from poker_agent.config import EquityConfig
from poker_agent.math.equity import EquityResult, equity
from poker_agent.math.odds import OddsInfo, odds_info
from poker_agent.math.ranges import (
    CALL_RANGE, Range, infer_preflop_range, parse_range, range_size, remove_dead,
)
from poker_agent.math.strength import HandStrength, continues_vs_aggression, hand_strength
from poker_agent.rules import LegalActions, TableState, legal_actions, replay, to_dollars
from poker_agent.state import ActionType, GameState, Street

# How much weight a combo keeps when it would not continue vs aggression.
BLUFF_WEIGHT_AGGRESSIVE = 0.3  # opponent bet or raised
BLUFF_WEIGHT_PASSIVE = 0.5  # opponent called a bet
AGGRESSIVE = (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)


class OpponentInfo(BaseModel):
    seat: int
    position: str
    stack: float  # remaining behind
    committed: float  # total put in this hand
    range_label: str
    range_notation: str
    range_combos: float  # weighted combo count after narrowing
    narrowed_by: list[str]
    note: str | None = None
    in_equity_calc: bool


class Analysis(BaseModel):
    street: Street
    hero_position: str
    hero_cards: list[str]
    board: list[str]
    big_blind: float
    hero_stack: float
    hero_stack_bb: float
    players_in_hand: int
    players_to_act_behind: int
    legal: LegalActions
    odds: OddsInfo
    strength: HandStrength
    opponents: list[OpponentInfo]
    equity: EquityResult
    equity_basis: str  # what the equity was computed against
    equity_vs_random: float
    equity_margin: float | None  # equity - pot odds, when facing a bet
    warnings: list[str]


def _narrow(rng: Range, seat: int, state: GameState, board_ids: list[int]) -> tuple[Range, list[str]]:
    """Down-weight combos that wouldn't continue given the seat's postflop actions."""
    notes = []
    for street, n_cards in ((Street.FLOP, 3), (Street.TURN, 4), (Street.RIVER, 5)):
        if len(board_ids) < n_cards:
            break
        acts = [a.type for a in state.actions if a.street is street and a.seat == seat]
        if any(t in AGGRESSIVE for t in acts):
            factor, what = BLUFF_WEIGHT_AGGRESSIVE, "bet/raised"
        elif ActionType.CALL in acts:
            factor, what = BLUFF_WEIGHT_PASSIVE, "called"
        else:
            continue
        board = board_ids[:n_cards]
        rng = {
            c: (w if continues_vs_aggression(c, board) else w * factor) for c, w in rng.items()
        }
        notes.append(f"{what} on {street.value}")
    return rng, notes


def analyze(state: GameState, eq_cfg: EquityConfig | None = None) -> Analysis:
    eq_cfg = eq_cfg or EquityConfig()
    table: TableState = replay(state)
    legal = legal_actions(state, table)
    odds = odds_info(state, table)
    hole, board = state.hole_card_ids, state.board_ids
    dead = set(hole) | set(board)
    strength = hand_strength(hole, board)
    positions = state.positions
    warnings: list[str] = []

    opponents: list[OpponentInfo] = []
    eq_ranges: list[Range] = []
    for seat in table.in_hand:
        if seat == state.hero_seat:
            continue
        label, notation = infer_preflop_range(state, seat)
        voluntary = label != "no voluntary action"
        if seat in state.opponent_ranges:
            label, notation = f"{label} (user override)", state.opponent_ranges[seat]
        rng = remove_dead(parse_range(notation), dead)
        rng, narrowed = _narrow(rng, seat, state, board)
        # Postflop everyone still in the hand is in the pot; preflop only
        # players who have voluntarily entered (or checked the BB) count.
        include = state.street is not Street.PREFLOP or voluntary or seat in state.opponent_ranges
        if include:
            eq_ranges.append(rng)
        s = table.seats[seat]
        opponents.append(OpponentInfo(
            seat=seat,
            position=positions[seat],
            stack=to_dollars(s.stack),
            committed=to_dollars(s.committed_total),
            range_label=label,
            range_notation=notation,
            range_combos=round(range_size(rng), 1),
            narrowed_by=narrowed,
            note=state.opponent_notes.get(seat),
            in_equity_calc=include,
        ))

    seed = eq_cfg.seed or None
    if eq_ranges:
        basis = f"vs {len(eq_ranges)} opponent range{'s' if len(eq_ranges) > 1 else ''} in the pot"
    else:
        eq_ranges = [remove_dead(parse_range(CALL_RANGE), dead)]
        basis = "unopened pot: heads-up vs a typical calling range"
    eq = equity(hole, board, eq_ranges, iterations=eq_cfg.iterations, seed=seed)
    eq_rand = equity(
        hole, board, [remove_dead(parse_range("any"), dead)],
        iterations=min(eq_cfg.iterations, 5000), seed=seed,
    ).equity

    if state.reported_pot is not None and abs(state.reported_pot - odds.pot) >= 0.05:
        warnings.append(
            f"reported pot {state.reported_pot:.2f} differs from derived pot {odds.pot:.2f}; "
            "check the action history"
        )
    if eq.method == "monte_carlo" and eq.stderr > 0.01:
        warnings.append(f"equity is a noisy estimate (±{eq.stderr * 2:.1%})")

    hero = table.seats[state.hero_seat]
    bb = table.big_blind
    return Analysis(
        street=state.street,
        hero_position=state.hero_position,
        hero_cards=state.hole_cards,
        board=state.board,
        big_blind=to_dollars(bb),
        hero_stack=to_dollars(hero.stack),
        hero_stack_bb=round(hero.stack / bb, 1),
        players_in_hand=len(table.in_hand),
        players_to_act_behind=len([s for s in table.needs_action() if s != state.hero_seat]),
        legal=legal,
        odds=odds,
        strength=strength,
        opponents=opponents,
        equity=eq,
        equity_basis=basis,
        equity_vs_random=round(eq_rand, 4),
        equity_margin=round(eq.equity - odds.pot_odds, 4) if odds.pot_odds else None,
        warnings=warnings,
    )
