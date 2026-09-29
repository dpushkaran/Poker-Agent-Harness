"""Pot odds, stack-to-pot ratio and related bet-facing numbers."""

from __future__ import annotations

from pydantic import BaseModel

from poker_agent.rules import TableState, to_dollars
from poker_agent.state import GameState


class OddsInfo(BaseModel):
    pot: float
    to_call: float
    pot_odds: float | None  # equity needed to break even on a call (0..1)
    bet_to_pot: float | None  # size of the bet faced, as a fraction of the pot before it
    mdf: float | None  # minimum defense frequency vs that bet (0..1)
    effective_stack: float
    effective_stack_bb: float
    spr: float  # effective stack / pot
    pot_bb: float


def pot_odds(to_call: float, pot: float) -> float:
    """Fraction of the final pot the hero contributes by calling."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def odds_info(state: GameState, table: TableState) -> OddsInfo:
    hero = table.seats[state.hero_seat]
    to_call = table.to_call(state.hero_seat)
    pot = table.pot
    opp_stacks = [
        table.seats[s].stack for s in table.in_hand if s != state.hero_seat
    ]
    effective = min(hero.stack, max(opp_stacks, default=0))
    bb = table.big_blind

    if to_call > 0:
        pot_before_bet = pot - to_call
        po = pot_odds(to_call, pot)
        btp = to_call / pot_before_bet if pot_before_bet > 0 else None
        mdf = pot_before_bet / pot
    else:
        po = btp = mdf = None

    return OddsInfo(
        pot=to_dollars(pot),
        to_call=to_dollars(to_call),
        pot_odds=round(po, 4) if po is not None else None,
        bet_to_pot=round(btp, 3) if btp is not None else None,
        mdf=round(mdf, 4) if mdf is not None else None,
        effective_stack=to_dollars(effective),
        effective_stack_bb=round(effective / bb, 1),
        spr=round(effective / pot, 2) if pot else 0.0,
        pot_bb=round(pot / bb, 1),
    )
