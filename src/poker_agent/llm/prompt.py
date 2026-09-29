"""Turn a game state plus computed analysis into chat messages.

Prompts are versioned so the eval harness can compare them side by side.
"""

from __future__ import annotations

from functools import lru_cache

from poker_agent.analysis import Analysis
from poker_agent.math.ranges import (
    BB_DEFEND_RANGE, CALL_RANGE, OPEN_RANGES, THREE_BET_RANGE, combo, parse_range,
)
from poker_agent.rules import replay, to_dollars
from poker_agent.state import ActionType, GameState, Street

SYSTEM_V1 = """\
You are a poker strategy assistant for a friendly home no-limit Texas Hold'em cash game \
({sb:.2f}/{bb:.2f} blinds, {buy_in:.0f}-dollar buy-ins, up to 7 players).

Typical home-game tendencies: players are loose and passive. They limp and call too often, \
rarely bluff with big bets or raises, and pay off value bets. So value bet thinner and larger, \
bluff less (especially multiway), and give big turn/river raises a lot of respect.

You receive the hand state plus numbers computed exactly by software. Rules:
- Treat every provided number (pot, pot odds, equity, outs, stack sizes) as correct. \
Do not recompute them.
- Choose exactly one action from the LEGAL ACTIONS list.
- For bet or raise, "amount" is the TOTAL you will have committed on this street after the \
action ("raise to"), in dollars, within the stated range. For fold, check, call and all_in \
use null.
- Compare equity with pot odds when facing a bet, and consider position, stack depth, \
the opponents' estimated ranges and any notes about them.
- Keep "reasoning" to 2-4 sentences that cite the key numbers. List up to 5 short \
"key_factors".
Respond only with JSON matching the schema."""

SYSTEM_V2 = SYSTEM_V1.replace(
    "Respond only with JSON matching the schema.",
    """Preflop:
- If the pot is unopened or only limped, raise or fold. Do not just call (limp), except to \
complete the small blind or overlimp in late position with small pairs or suited connectors \
behind several limpers. Pot odds are not the main consideration in these spots.
- The PREFLOP line says whether the hand is in a standard range for this spot. Follow it \
unless you have a specific reason not to.
- Facing a raise, 3-bet premium hands (QQ+, AK) for value rather than calling.
Sizing:
- Pick a size from SIZE OPTIONS unless you have a clear reason. Never pick the minimum just \
because it is the lower bound. Value bets are usually 50-75% of the pot, larger against \
players who call too much.
Respond only with JSON matching the schema.""",
)

PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}
DEFAULT_PROMPT = "v2"

ISO_RANGES = {1: "66+, A9s+, KTs+, QJs, AJo+, KQo", 2: "88+, AJs+, KQs, AQo+"}
PREMIUM = "QQ+, AKs, AKo"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def action_history(state: GameState) -> list[str]:
    """One line per street, e.g. 'Preflop: UTG(3) folds, MP(4) raises to 0.60'."""
    table = replay(state)
    pos = state.positions
    by_street: dict[Street, list[str]] = {}
    for a, total in table.log:
        who = f"{pos[a.seat]}({a.seat}{', hero' if a.seat == state.hero_seat else ''})"
        amt = f"{to_dollars(total):.2f}"
        verb = {
            ActionType.FOLD: "folds",
            ActionType.CHECK: "checks",
            ActionType.CALL: f"calls {amt}",
            ActionType.BET: f"bets {amt}",
            ActionType.RAISE: f"raises to {amt}",
            ActionType.ALL_IN: f"is all-in for {amt}",
        }[a.type]
        if a.type is ActionType.BET and a.street is Street.PREFLOP:
            verb = f"raises to {amt}"
        by_street.setdefault(a.street, []).append(f"{who} {verb}")
    blinds = (f"SB({state.small_blind_seat}) posts {state.blinds.small:.2f}, "
              f"BB({state.big_blind_seat}) posts {state.blinds.big:.2f}")
    lines = [f"Blinds: {blinds}"]
    for street in Street:
        if street in by_street:
            lines.append(f"{street.value.capitalize()}: " + ", ".join(by_street[street]))
    return lines


def legal_lines(a: Analysis) -> list[str]:
    la = a.legal
    out = []
    if la.can_fold:
        out.append("fold")
    if la.can_check:
        out.append("check")
    if la.can_call:
        all_in = " (puts you all-in)" if la.call_amount >= la.hero_stack else ""
        out.append(f"call {la.call_amount:.2f}{all_in}")
    if la.can_bet:
        out.append(f"bet: amount between {la.min_to:.2f} and {la.max_to:.2f}")
    if la.can_raise:
        out.append(f"raise: amount (raise-to total) between {la.min_to:.2f} and {la.max_to:.2f}")
    if "all_in" in la.names():
        out.append(f"all_in: commits your whole stack ({la.hero_stack + la.hero_committed_street:.2f} "
                   "total on this street)")
    return [f"- {x}" for x in out]


@lru_cache(maxsize=64)
def _combos(text: str) -> frozenset:
    return frozenset(parse_range(text))


def _in_range(state: GameState, text: str) -> bool:
    a, b = state.hole_card_ids
    return combo(a, b) in _combos(text)


def preflop_situation(state: GameState, a: Analysis) -> tuple[str, str, bool]:
    """(description, chart guidance, is_unopened_or_limped) for the hero's preflop spot."""
    pos = a.hero_position
    raises = limpers = 0
    last_raise_to = None
    for x in state.actions:
        if x.street is not Street.PREFLOP:
            break
        if x.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            raises += 1
            last_raise_to = x.amount
        elif x.type is ActionType.CALL and raises == 0:
            limpers += 1
    if raises == 0:
        if limpers == 0:
            desc = "Unopened pot: everyone before hero folded."
            rng, what = OPEN_RANGES.get(pos, OPEN_RANGES["BTN"]), f"{pos} opening range"
        else:
            desc = f"{limpers} limper(s), nobody has raised."
            rng, what = ISO_RANGES[min(limpers, 2)], "isolation-raise range"
        if pos == "BB":
            chart = ("hand is in" if _in_range(state, rng) else "hand is not in") + f" the {what}; check otherwise"
        else:
            chart = ("hand is in" if _in_range(state, rng) else "hand is not in") + f" the {what}"
        return desc, chart, True
    kind = "raise" if raises == 1 else "3-bet" if raises == 2 else f"{raises - 1}-bet"
    amt = f" to {last_raise_to:.2f}" if last_raise_to else " (all-in)"
    desc = f"Facing a {kind}{amt}."
    if _in_range(state, PREMIUM):
        chart = "premium hand: re-raise for value"
    elif raises == 1 and _in_range(state, THREE_BET_RANGE):
        chart = "strong hand: 3-bet or call"
    elif raises == 1 and _in_range(state, BB_DEFEND_RANGE if pos == "BB" else CALL_RANGE):
        chart = "in the standard calling range vs a raise"
    else:
        chart = "outside the standard continuing range; usually fold"
    return desc, chart, False


def size_options(state: GameState, a: Analysis) -> list[str]:
    """Concrete legal sizes, snapped to the chip increment."""
    la, o = a.legal, a.odds
    if la.min_to is None:
        return []
    bb = a.big_blind
    cur = la.call_amount + la.hero_committed_street  # bet the hero is facing ('to' level)
    opts: list[tuple[str, float]] = []
    if a.street is Street.PREFLOP:
        desc, _, unopened = preflop_situation(state, a)
        if unopened:
            limpers = round((o.pot - bb * 1.5) / bb) if o.pot > bb * 1.5 else 0
            opts.append(("standard raise (3bb + 1bb per limper)", (3 + max(limpers, 0)) * bb))
            opts.append(("larger raise (4bb + 1bb per limper)", (4 + max(limpers, 0)) * bb))
        else:
            opts.append(("3x the raise (in position)", cur * 3))
            opts.append(("4x the raise (out of position)", cur * 4))
    elif la.can_bet:
        for frac, name in ((0.33, "1/3 pot"), (0.5, "1/2 pot"), (0.66, "2/3 pot"), (0.75, "3/4 pot"),
                           (1.0, "pot")):
            opts.append((name, o.pot * frac))
    else:
        opts.append(("2.5x the bet", cur * 2.5))
        opts.append(("3x the bet", cur * 3))
        opts.append(("pot-sized raise", cur + (o.pot + la.call_amount)))
    out, seen = [], set()
    for name, to in opts:
        v = la.snap(to)
        if v in seen:
            continue
        seen.add(v)
        tag = " (all-in)" if v >= la.max_to else ""
        out.append(f"- {name}: {'bet' if la.can_bet else 'raise'} to {v:.2f}{tag}")
    return out


def render_state(state: GameState, a: Analysis, version: str = DEFAULT_PROMPT) -> str:
    o, s, e = a.odds, a.strength, a.equity
    lines = [
        f"TABLE: {len(state.players)} players dealt. Hero is seat {state.hero_seat} "
        f"({a.hero_position}).",
        f"HERO CARDS: {' '.join(a.hero_cards)}",
        f"BOARD: {' '.join(a.board) if a.board else '(none)'} [{a.street.value}]",
        "",
        "ACTION SO FAR:",
        *[f"  {x}" for x in action_history(state)],
        "",
        "OPPONENTS STILL IN THE HAND:",
    ]
    for opp in a.opponents:
        narrowed = f", narrowed because they {' and '.join(opp.narrowed_by)}" if opp.narrowed_by else ""
        note = f" Read: {opp.note}." if opp.note else ""
        lines.append(
            f"  {opp.position} (seat {opp.seat}): {opp.stack:.2f} behind. Estimated range: "
            f"{opp.range_label}{narrowed} (~{opp.range_combos:.0f} combos).{note}"
        )
    lines += [
        "",
        "COMPUTED NUMBERS (from software; equity is vs the estimated ranges above):",
        f"  Pot: {o.pot:.2f} ({o.pot_bb:g}bb). To call: {o.to_call:.2f}.",
    ]
    preflop = None
    if version != "v1" and a.street is Street.PREFLOP:
        preflop = preflop_situation(state, a)
    if o.pot_odds and preflop and preflop[2]:
        lines.append(f"  To call {o.to_call:.2f} would be a limp; pot odds ({_pct(o.pot_odds)}) are "
                     "not the main consideration preflop.")
    elif o.pot_odds:
        lines.append(f"  Pot odds: you need {_pct(o.pot_odds)} equity to call. "
                     f"Bet faced is {o.bet_to_pot:.0%} of the pot; MDF {_pct(o.mdf)}.")
    lines += [
        f"  Hero hand: {s.description}."
        + (f" Draws: {', '.join(s.draws)}." if s.draws else "")
        + (f" Outs to improve: {s.outs}." if s.outs else ""),
    ]
    if s.board_texture:
        lines.append(f"  Board texture: {', '.join(s.board_texture)}.")
    lines.append(f"  Hero equity: {_pct(e.equity)} ({a.equity_basis}). "
                 f"Versus a random hand: {_pct(a.equity_vs_random)}.")
    if a.equity_margin is not None and not (preflop and preflop[2]):
        lines.append(f"  Equity minus pot odds: {a.equity_margin:+.1%}.")
    lines += [
        f"  Hero stack: {a.hero_stack:.2f} ({a.hero_stack_bb:g}bb). Effective stack: "
        f"{o.effective_stack:.2f} ({o.effective_stack_bb:g}bb). SPR: {o.spr:g}.",
        f"  Players still to act after hero on this street: {a.players_to_act_behind}.",
    ]
    lines += [f"  Warning: {w}" for w in a.warnings]
    if preflop:
        lines += ["", f"PREFLOP: {preflop[0]} Chart guidance: {preflop[1]}."]
    lines += ["", "LEGAL ACTIONS:", *legal_lines(a)]
    if version != "v1":
        sizes = size_options(state, a)
        if sizes:
            lines += ["", "SIZE OPTIONS (amount = raise-to total):", *sizes]
    lines += ["", "What should hero do?"]
    return "\n".join(lines)


def build_messages(state: GameState, a: Analysis, version: str = DEFAULT_PROMPT,
                   buy_in: float = 10.0) -> list[dict]:
    system = PROMPTS[version].format(sb=state.blinds.small, bb=state.blinds.big, buy_in=buy_in)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_state(state, a, version)},
    ]
