"""Input models describing a hand in progress.

Amounts are in dollars. Player stacks are the stacks at the *start* of
the hand (before blinds); the pot, per-street commitments and remaining
stacks are derived by replaying the action history (see ``rules.py``),
so the user never has to keep a running pot in sync by hand.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from poker_agent.cards import Card, card_str, ensure_unique, parse_cards

POSITION_NAMES: dict[int, list[str]] = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "UTG"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
    7: ["BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"],
    8: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
    9: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"],
}


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

    @property
    def index(self) -> int:
        return list(Street).index(self)

    @classmethod
    def from_board_size(cls, n: int) -> Street:
        try:
            return {0: cls.PREFLOP, 3: cls.FLOP, 4: cls.TURN, 5: cls.RIVER}[n]
        except KeyError:
            raise ValueError(f"board must have 0, 3, 4 or 5 cards, got {n}") from None


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class Player(BaseModel):
    seat: int = Field(ge=1, le=10)
    stack: float = Field(gt=0, description="Stack at the start of the hand, in dollars")
    name: str | None = None


class Action(BaseModel):
    """One betting action.

    For ``bet`` and ``raise``, ``amount`` is the player's *total* commitment
    on this street after the action ("raise to"). It is ignored for other
    action types, whose sizes are implied by the state.
    """

    street: Street
    seat: int
    type: ActionType
    amount: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _amount_required(self) -> Action:
        if self.type in (ActionType.BET, ActionType.RAISE) and self.amount is None:
            raise ValueError(f"{self.type.value} requires an amount (the 'to' total)")
        return self


class Blinds(BaseModel):
    small: float = 0.10
    big: float = 0.20


class GameState(BaseModel):
    players: list[Player] = Field(min_length=2, max_length=9)
    button_seat: int
    hero_seat: int
    hole_cards: list[str] = Field(min_length=2, max_length=2)
    board: list[str] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    blinds: Blinds = Blinds()
    reported_pot: float | None = Field(
        default=None, description="Optional pot size as seen at the table, for cross-checking"
    )
    opponent_notes: dict[int, str] = Field(
        default_factory=dict, description="Free-text reads per seat, e.g. 'calls too much'"
    )
    opponent_ranges: dict[int, str] = Field(
        default_factory=dict,
        description="Per-seat preflop range overrides in range notation, e.g. {5: '22+, A2s+'}",
    )

    @field_validator("hole_cards", "board", mode="before")
    @classmethod
    def _split_card_string(cls, v):
        if isinstance(v, str):
            return [card_str(c) for c in parse_cards(v)] if v.strip() else []
        return v

    @model_validator(mode="after")
    def _validate(self) -> GameState:
        seats = [p.seat for p in self.players]
        if len(set(seats)) != len(seats):
            raise ValueError("duplicate seat numbers")
        if self.button_seat not in seats:
            raise ValueError(f"button seat {self.button_seat} has no player")
        if self.hero_seat not in seats:
            raise ValueError(f"hero seat {self.hero_seat} has no player")
        Street.from_board_size(len(self.board))
        ensure_unique(self.hole_card_ids + self.board_ids)
        for a in self.actions:
            if a.seat not in seats:
                raise ValueError(f"action by empty seat {a.seat}")
            if a.street.index > self.street.index:
                raise ValueError(f"{a.street.value} action but board is still {self.street.value}")
        return self

    # --- derived, static properties -------------------------------------

    @property
    def street(self) -> Street:
        return Street.from_board_size(len(self.board))

    @property
    def hole_card_ids(self) -> list[Card]:
        return parse_cards(self.hole_cards)

    @property
    def board_ids(self) -> list[Card]:
        return parse_cards(self.board)

    @property
    def seat_order(self) -> list[int]:
        """Seats in clockwise order starting with the button."""
        seats = sorted(p.seat for p in self.players)
        i = seats.index(self.button_seat)
        return seats[i:] + seats[:i]

    @property
    def positions(self) -> dict[int, str]:
        names = POSITION_NAMES[len(self.players)]
        return dict(zip(self.seat_order, names, strict=True))

    @property
    def small_blind_seat(self) -> int:
        order = self.seat_order
        return order[0] if len(order) == 2 else order[1]

    @property
    def big_blind_seat(self) -> int:
        order = self.seat_order
        return order[1] if len(order) == 2 else order[2]

    def player(self, seat: int) -> Player:
        return next(p for p in self.players if p.seat == seat)

    @property
    def hero(self) -> Player:
        return self.player(self.hero_seat)

    @property
    def hero_position(self) -> str:
        return self.positions[self.hero_seat]

