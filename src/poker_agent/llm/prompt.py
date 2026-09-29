"""Turn a game state plus computed analysis into chat messages.

Prompts are versioned so the eval harness can compare them side by side.
"""

from __future__ import annotations

from poker_agent.analysis import Analysis
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

PROMPTS = {"v1": SYSTEM_V1}
DEFAULT_PROMPT = "v1"


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


def render_state(state: GameState, a: Analysis) -> str:
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
    if o.pot_odds:
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
    if a.equity_margin is not None:
        lines.append(f"  Equity minus pot odds: {a.equity_margin:+.1%}.")
    lines += [
        f"  Hero stack: {a.hero_stack:.2f} ({a.hero_stack_bb:g}bb). Effective stack: "
        f"{o.effective_stack:.2f} ({o.effective_stack_bb:g}bb). SPR: {o.spr:g}.",
        f"  Players still to act after hero on this street: {a.players_to_act_behind}.",
    ]
    lines += [f"  Warning: {w}" for w in a.warnings]
    lines += ["", "LEGAL ACTIONS:", *legal_lines(a), "", "What should hero do?"]
    return "\n".join(lines)


def build_messages(state: GameState, a: Analysis, version: str = DEFAULT_PROMPT,
                   buy_in: float = 10.0) -> list[dict]:
    system = PROMPTS[version].format(sb=state.blinds.small, bb=state.blinds.big, buy_in=buy_in)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_state(state, a)},
    ]
