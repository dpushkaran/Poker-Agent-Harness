"""Card parsing and deck utilities.

Cards are represented as ints 0..51 so they can be fed straight into the
hand evaluator: ``id = rank_index * 4 + suit_index`` where ranks run
2..A (0..12) and suits are c, d, h, s (0..3).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYMBOLS = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}

Card = int
FULL_DECK: tuple[Card, ...] = tuple(range(52))

_CARD_RE = re.compile(r"(10|[2-9TJQKA])([cdhs])", re.IGNORECASE)


class CardError(ValueError):
    """Raised for malformed or duplicate cards."""


def parse_card(text: str) -> Card:
    """Parse one card such as ``"Ah"``, ``"td"``, ``"10s"`` or ``"K♥"``."""
    s = text.strip()
    for sym, letter in SUIT_SYMBOLS.items():
        s = s.replace(sym, letter)
    m = _CARD_RE.fullmatch(s)
    if not m:
        raise CardError(f"invalid card: {text!r}")
    rank = "T" if m.group(1) == "10" else m.group(1).upper()
    return RANKS.index(rank) * 4 + SUITS.index(m.group(2).lower())


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    """Parse several cards from ``"AhKd"``, ``"Ah Kd"``, ``"Ah,Kd"`` or a list."""
    if isinstance(text, str):
        s = text
        for sym, letter in SUIT_SYMBOLS.items():
            s = s.replace(sym, letter)
        s = re.sub(r"[\s,]+", "", s)
        tokens = _CARD_RE.findall(s)
        if "".join(r + su for r, su in tokens).lower() != s.lower():
            raise CardError(f"invalid card list: {text!r}")
        cards = [parse_card(r + su) for r, su in tokens]
    else:
        cards = [parse_card(t) for t in text]
    ensure_unique(cards)
    return cards


def card_str(card: Card) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def cards_str(cards: Iterable[Card]) -> str:
    return " ".join(card_str(c) for c in cards)


def rank_of(card: Card) -> int:
    """Rank index 0 (deuce) .. 12 (ace)."""
    return card // 4


def suit_of(card: Card) -> int:
    return card % 4


def ensure_unique(cards: Iterable[Card]) -> None:
    seen: set[Card] = set()
    for c in cards:
        if c in seen:
            raise CardError(f"duplicate card: {card_str(c)}")
        seen.add(c)


def remaining_deck(dead: Iterable[Card]) -> list[Card]:
    dead_set = set(dead)
    return [c for c in FULL_DECK if c not in dead_set]
