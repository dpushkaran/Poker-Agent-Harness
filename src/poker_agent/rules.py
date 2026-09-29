"""No-limit hold'em betting rules.

``replay`` walks the action history of a ``GameState`` (posting blinds
automatically), enforcing turn order and bet sizing, and returns the
resulting ``TableState``. ``legal_actions`` derives what the hero may do
from that table state; it is the single source of truth used by both the
LLM output validator and the UI.

All arithmetic is done in integer cents to avoid float drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from poker_agent.state import ActionType, GameState, Street


class IllegalAction(ValueError):
    """The action history is inconsistent with the rules."""


class NotHerosTurn(ValueError):
    """A decision was requested but someone else is due to act."""


def to_cents(dollars: float) -> int:
    return round(dollars * 100)


def to_dollars(cents: int) -> float:
    return round(cents / 100, 2)


@dataclass
class SeatStatus:
    seat: int
    start_stack: int
    committed_total: int = 0
    committed_street: int = 0
    folded: bool = False

    @property
    def stack(self) -> int:
        return self.start_stack - self.committed_total

    @property
    def all_in(self) -> bool:
        return not self.folded and self.stack == 0

    @property
    def active(self) -> bool:
        """Still in the hand and able to act."""
        return not self.folded and self.stack > 0

    def commit(self, cents: int) -> None:
        self.committed_total += cents
        self.committed_street += cents


@dataclass
class TableState:
    street: Street
    order: list[int]  # clockwise from the button
    seats: dict[int, SeatStatus]
    big_blind: int
    current_bet: int = 0
    last_raise_size: int = 0
    acted_since_full_raise: set[int] = field(default_factory=set)
    acted_this_round: set[int] = field(default_factory=set)
    pointer: int = 0  # seat that acted last (or the seat before the first actor)

    @property
    def pot(self) -> int:
        return sum(s.committed_total for s in self.seats.values())

    @property
    def in_hand(self) -> list[int]:
        return [s for s in self.order if not self.seats[s].folded]

    @property
    def hand_over(self) -> bool:
        return len(self.in_hand) <= 1

    def needs_action(self) -> list[int]:
        active = [s for s in self.order if self.seats[s].active]
        if len(active) <= 1 and all(
            self.seats[s].committed_street >= self.current_bet for s in active
        ):
            return []
        return [
            s
            for s in active
            if s not in self.acted_this_round or self.seats[s].committed_street < self.current_bet
        ]

    @property
    def round_closed(self) -> bool:
        return self.hand_over or not self.needs_action()

    @property
    def to_act(self) -> int | None:
        """Next seat due to act on this street, or None if the round is closed."""
        if self.round_closed:
            return None
        waiting = set(self.needs_action())
        n = len(self.order)
        start = self.order.index(self.pointer)
        for i in range(1, n + 1):
            seat = self.order[(start + i) % n]
            if seat in waiting:
                return seat
        return None

    def to_call(self, seat: int) -> int:
        s = self.seats[seat]
        return min(self.current_bet - s.committed_street, s.stack)

    def can_raise(self, seat: int) -> bool:
        s = self.seats[seat]
        others_active = any(
            self.seats[o].active for o in self.order if o != seat
        )
        return (
            s.active
            and others_active
            and seat not in self.acted_since_full_raise
            and s.stack > self.current_bet - s.committed_street
        )

    def start_street(self, street: Street, first_pointer: int) -> None:
        self.street = street
        for s in self.seats.values():
            s.committed_street = 0
        self.current_bet = 0
        self.last_raise_size = self.big_blind
        self.acted_since_full_raise = set()
        self.acted_this_round = set()
        self.pointer = first_pointer


def replay(state: GameState) -> TableState:
    """Replay blinds and actions up to the current street."""
    order = state.seat_order
    sb, bb = to_cents(state.blinds.small), to_cents(state.blinds.big)
    table = TableState(
        street=Street.PREFLOP,
        order=order,
        seats={p.seat: SeatStatus(p.seat, to_cents(p.stack)) for p in state.players},
        big_blind=bb,
    )

    # Blinds. Posting is not "acting", so the big blind keeps its option.
    sb_seat, bb_seat = state.small_blind_seat, state.big_blind_seat
    table.start_street(Street.PREFLOP, first_pointer=bb_seat)
    for seat, amount in ((sb_seat, sb), (bb_seat, bb)):
        s = table.seats[seat]
        s.commit(min(amount, s.stack))
    table.current_bet = max(table.seats[sb_seat].committed_street, table.seats[bb_seat].committed_street)
    table.last_raise_size = bb

    by_street: dict[Street, list] = {st: [] for st in Street}
    for a in state.actions:
        by_street[a.street].append(a)

    for street in Street:
        if street.index > state.street.index:
            break
        if street is not Street.PREFLOP:
            prev = list(Street)[street.index - 1]
            if not table.round_closed:
                raise IllegalAction(
                    f"{prev.value} betting is incomplete: seat {table.to_act} still to act"
                )
            table.start_street(street, first_pointer=state.button_seat)
        for a in by_street[street]:
            _apply(table, a)

    return table


def _apply(table: TableState, a) -> None:
    where = f"{a.street.value}, seat {a.seat}"
    if table.hand_over:
        raise IllegalAction(f"{where}: hand is already over")
    expected = table.to_act
    if expected is None:
        raise IllegalAction(f"{where}: betting round is already closed")
    if a.seat != expected:
        raise IllegalAction(f"{where}: expected seat {expected} to act")

    s = table.seats[a.seat]
    to_call = table.current_bet - s.committed_street
    max_to = s.committed_street + s.stack

    if a.type is ActionType.FOLD:
        s.folded = True
    elif a.type is ActionType.CHECK:
        if to_call > 0:
            raise IllegalAction(f"{where}: cannot check facing a bet of {to_dollars(to_call)}")
    elif a.type is ActionType.CALL:
        if to_call <= 0:
            raise IllegalAction(f"{where}: nothing to call")
        s.commit(min(to_call, s.stack))
    else:
        target = max_to if a.type is ActionType.ALL_IN else to_cents(a.amount)
        if target > max_to:
            raise IllegalAction(
                f"{where}: {to_dollars(target)} exceeds stack (max {to_dollars(max_to)})"
            )
        if target <= table.current_bet:
            if a.type is ActionType.ALL_IN:  # all-in for less than (or equal to) a call
                s.commit(target - s.committed_street)
                table.acted_this_round.add(a.seat)
                table.pointer = a.seat
                return
            raise IllegalAction(
                f"{where}: raise to {to_dollars(target)} is not above the current bet "
                f"{to_dollars(table.current_bet)}"
            )
        if not table.can_raise(a.seat):
            raise IllegalAction(f"{where}: betting was not reopened, cannot raise")
        increment = target - table.current_bet
        is_all_in = target == max_to
        if increment < table.last_raise_size and not is_all_in:
            min_to = table.current_bet + table.last_raise_size
            raise IllegalAction(f"{where}: minimum is {to_dollars(min_to)}, got {to_dollars(target)}")
        s.commit(target - s.committed_street)
        table.current_bet = target
        if increment >= table.last_raise_size:
            table.last_raise_size = increment
            table.acted_since_full_raise = set()

    table.acted_since_full_raise.add(a.seat)
    table.acted_this_round.add(a.seat)
    table.pointer = a.seat


class LegalActions(BaseModel):
    """What the hero may do right now. Amounts in dollars; bet/raise are 'to' totals."""

    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: float
    can_bet: bool
    can_raise: bool
    min_to: float | None
    max_to: float | None
    hero_stack: float
    hero_committed_street: float

    def snap(self, to: float, increment: float = 0.10) -> float:
        """Round a bet/raise 'to' amount to the chip increment and clamp it to legal bounds."""
        if self.min_to is None or self.max_to is None:
            raise ValueError("no bet or raise is available")
        snapped = round(round(to / increment) * increment, 2)
        return min(max(snapped, self.min_to), self.max_to)

    def names(self) -> list[str]:
        out = []
        if self.can_fold:
            out.append("fold")
        if self.can_check:
            out.append("check")
        if self.can_call:
            out.append("call")
        if self.can_bet:
            out.append("bet")
        if self.can_raise:
            out.append("raise")
        if self.can_bet or self.can_raise or self.can_call:
            out.append("all_in")
        return out


def legal_actions(state: GameState, table: TableState | None = None) -> LegalActions:
    table = table or replay(state)
    hero = state.hero_seat
    if table.hand_over:
        raise NotHerosTurn("the hand is over")
    if table.seats[hero].folded:
        raise NotHerosTurn("hero has folded")
    if table.to_act != hero:
        who = "nobody (betting round closed)" if table.to_act is None else f"seat {table.to_act}"
        raise NotHerosTurn(f"it is not hero's turn: next to act is {who}")

    s = table.seats[hero]
    to_call = table.to_call(hero)
    max_to = s.committed_street + s.stack
    can_bet = table.current_bet == 0 and s.stack > 0 and any(
        table.seats[o].active for o in table.order if o != hero
    )
    can_raise = table.current_bet > 0 and table.can_raise(hero)
    if can_bet:
        min_to = min(table.big_blind, max_to)
    elif can_raise:
        min_to = min(table.current_bet + table.last_raise_size, max_to)
    else:
        min_to = None
    return LegalActions(
        can_fold=to_call > 0,
        can_check=to_call == 0,
        can_call=to_call > 0,
        call_amount=to_dollars(to_call),
        can_bet=can_bet,
        can_raise=can_raise,
        min_to=to_dollars(min_to) if min_to is not None else None,
        max_to=to_dollars(max_to) if (can_bet or can_raise) else None,
        hero_stack=to_dollars(s.stack),
        hero_committed_street=to_dollars(s.committed_street),
    )
