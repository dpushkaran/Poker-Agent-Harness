"""Made-hand classification, draws, outs and board texture.

Descriptions are relative to the board ("top pair, K kicker", "set",
"nut flush draw") because that is what matters for decisions and what
the LLM needs to reason about.
"""

from __future__ import annotations

from collections import Counter

from phevaluator import evaluate_cards
from pydantic import BaseModel

from poker_agent.cards import RANKS, Card, rank_of, remaining_deck, suit_of

CATEGORIES = [
    "high card", "one pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
]


def category(rank: int) -> int:
    """Map a phevaluator rank (1 = royal flush .. 7462) to an index into CATEGORIES."""
    if rank <= 10:
        return 8
    if rank <= 166:
        return 7
    if rank <= 322:
        return 6
    if rank <= 1599:
        return 5
    if rank <= 1609:
        return 4
    if rank <= 2467:
        return 3
    if rank <= 3325:
        return 2
    if rank <= 6185:
        return 1
    return 0


def best_category(cards: list[Card]) -> int:
    return category(evaluate_cards(*cards)) if len(cards) >= 5 else _partial_category(cards)


def _partial_category(cards: list[Card]) -> int:
    """Category for fewer than five cards (preflop / hole cards only)."""
    counts = sorted(Counter(rank_of(c) for c in cards).values(), reverse=True)
    if counts and counts[0] >= 3:
        return 3
    if counts.count(2) >= 2:
        return 2
    if counts and counts[0] == 2:
        return 1
    return 0


class HandStrength(BaseModel):
    category: str
    description: str
    draws: list[str]
    outs: int  # unseen cards that improve hero to two pair or better (or complete a draw)
    outs_equity_estimate: float | None  # rule of 2/4, as a fraction
    board_texture: list[str]


def _rank_name(r: int) -> str:
    return RANKS[r]


def _straight_ranks(ranks: set[int]) -> bool:
    """True if the rank set contains five consecutive ranks (ace plays low too)."""
    rs = set(ranks)
    if 12 in rs:
        rs.add(-1)
    return any(all(r + i in rs for i in range(5)) for r in range(-1, 9))


def straight_draw_ranks(hole: list[Card], board: list[Card]) -> set[int]:
    """Ranks that would give hero a straight using at least one hole card."""
    have = {rank_of(c) for c in hole + board}
    board_ranks = {rank_of(c) for c in board}
    if _straight_ranks(have):
        return set()
    out = set()
    for r in range(13):
        if r in have:
            continue
        if _straight_ranks(have | {r}) and not _straight_ranks(board_ranks | {r}):
            out.add(r)
    return out


def flush_draw(hole: list[Card], board: list[Card]) -> tuple[int, int] | None:
    """(suit, cards of that suit) if hero has 3-4 of a suit using a hole card, else None."""
    counts = Counter(suit_of(c) for c in hole + board)
    for suit, n in counts.most_common():
        if n >= 5:
            return None
        if n >= 3 and any(suit_of(c) == suit for c in hole):
            return suit, n
    return None


def _is_nut_suit_card(hole: list[Card], board: list[Card], suit: int) -> bool:
    """Hero holds the highest card of `suit` not on the board."""
    on_board = {rank_of(c) for c in board if suit_of(c) == suit}
    top_missing = max(r for r in range(13) if r not in on_board)
    return any(suit_of(c) == suit and rank_of(c) == top_missing for c in hole)


def board_texture(board: list[Card]) -> list[str]:
    if not board:
        return []
    out = []
    ranks = [rank_of(c) for c in board]
    suits = Counter(suit_of(c) for c in board)
    rc = Counter(ranks)
    if max(rc.values()) >= 3:
        out.append("trips on board")
    elif list(rc.values()).count(2) >= 2:
        out.append("double-paired board")
    elif 2 in rc.values():
        out.append("paired board")
    top_suit = max(suits.values())
    if top_suit >= 3:
        out.append("monotone" if top_suit == len(board) == 3 else "flush possible")
    elif top_suit == 2 and len(board) <= 4:
        out.append("two-tone (flush draw possible)")
    else:
        out.append("rainbow" if len(board) <= 4 else "no flush possible")
    uniq = set(ranks)
    if 12 in uniq:
        uniq.add(-1)
    window = max(sum(1 for r in uniq if lo <= r <= lo + 4) for lo in range(-1, 9))
    if window >= 3:
        out.append("straight possible")
    elif window == 2 and len(board) <= 4:
        out.append("some straight draws")
    out.append("high card " + _rank_name(max(ranks)))
    return out


def _pair_description(hole: list[Card], board: list[Card]) -> str:
    hr = sorted((rank_of(c) for c in hole), reverse=True)
    br = sorted({rank_of(c) for c in board}, reverse=True)
    if hr[0] == hr[1]:
        p = hr[0]
        if p > br[0]:
            return f"overpair ({_rank_name(p)}{_rank_name(p)})"
        if p < br[-1]:
            return f"underpair ({_rank_name(p)}{_rank_name(p)})"
        return f"pocket pair below top card ({_rank_name(p)}{_rank_name(p)})"
    for r in hr:
        if r in br:
            idx = br.index(r)
            if idx == 0:
                which = "top pair"
            elif idx == len(br) - 1:
                which = "bottom pair"
            else:
                which = "second pair" if idx == 1 else "middle pair"
            kicker = hr[1] if r == hr[0] else hr[0]
            return f"{which} ({_rank_name(r)}s), {_rank_name(kicker)} kicker"
    return "pair on board only (playing the board)"


def describe(hole: list[Card], board: list[Card]) -> tuple[int, str]:
    """(category index, human description) of hero's current made hand."""
    cards = hole + board
    if not board:
        hr = sorted((rank_of(c) for c in hole), reverse=True)
        suited = suit_of(hole[0]) == suit_of(hole[1])
        if hr[0] == hr[1]:
            return 1, f"pocket pair ({_rank_name(hr[0])}{_rank_name(hr[0])})"
        return 0, f"{_rank_name(hr[0])}{_rank_name(hr[1])} {'suited' if suited else 'offsuit'}"

    cat = best_category(cards)
    board_cat = best_category(board) if len(board) >= 5 else _partial_category(board)
    name = CATEGORIES[cat]
    if len(board) == 5 and evaluate_cards(*cards) == evaluate_cards(*board):
        return cat, f"{name} (playing the board)"
    if cat == 0:
        hr = sorted((rank_of(c) for c in hole), reverse=True)
        overs = [r for r in hr if r > max(rank_of(c) for c in board)]
        extra = f", {len(overs)} overcard{'s' if len(overs) != 1 else ''}" if overs else ""
        return cat, f"{_rank_name(hr[0])}-high{extra}"
    if cat == 1:
        return cat, _pair_description(hole, board)
    if cat == 2:
        board_pairs = [r for r, n in Counter(rank_of(c) for c in board).items() if n >= 2]
        hr = [rank_of(c) for c in hole]
        if hr[0] == hr[1] and board_pairs:
            return cat, "two pair (pocket pair + board pair)"
        if board_pairs:
            return cat, f"two pair (one hole card + board pair: {_pair_description(hole, board)})"
        return cat, "two pair (both hole cards)"
    if cat == 3:
        hr = [rank_of(c) for c in hole]
        return cat, "set" if hr[0] == hr[1] else "trips (one hole card + board pair)"
    if cat == 5:
        suit = Counter(suit_of(c) for c in cards).most_common(1)[0][0]
        nut = _is_nut_suit_card(hole, board, suit)
        return cat, "nut flush" if nut else "flush"
    if cat == board_cat:
        return cat, f"{name} (mostly on board)"
    return cat, name


def outs(hole: list[Card], board: list[Card]) -> int:
    """Unseen cards that improve hero to two pair or better, or to a better category.

    An out must widen hero's lead over what the board alone makes, so a card
    that merely pairs the board (improving everyone equally) is not counted.
    """
    if len(board) not in (3, 4):
        return 0
    cur_margin = best_category(hole + board) - best_category(board)
    n = 0
    for c in remaining_deck(hole + board):
        new = best_category(hole + board + [c])
        if new < 2:
            continue
        if new - best_category(board + [c]) > cur_margin:
            n += 1
    return n


def hand_strength(hole: list[Card], board: list[Card]) -> HandStrength:
    cat, desc = describe(hole, board)
    draws: list[str] = []
    if len(board) in (3, 4) and cat < 5:
        fd = flush_draw(hole, board)
        if fd:
            suit, n = fd
            nut = _is_nut_suit_card(hole, board, suit)
            if n == 4:
                draws.append("nut flush draw" if nut else "flush draw")
            elif len(board) == 3:
                draws.append("backdoor flush draw")
        if cat < 4:
            sd = straight_draw_ranks(hole, board)
            if len(sd) >= 2:
                draws.append("open-ended straight draw" if len(sd) == 2 else "double straight draw")
            elif len(sd) == 1:
                draws.append("gutshot straight draw")
    n_outs = outs(hole, board)
    if len(board) == 3:
        est = min(n_outs * 4, 100) / 100
    elif len(board) == 4:
        est = min(n_outs * 2, 100) / 100
    else:
        est = None
    return HandStrength(
        category=CATEGORIES[cat],
        description=desc,
        draws=draws,
        outs=n_outs,
        outs_equity_estimate=est,
        board_texture=board_texture(board),
    )


def continues_vs_aggression(hole: tuple[Card, Card], board: list[Card]) -> bool:
    """Cheap test used to narrow ranges: pair+ using a hole card, or a real draw."""
    cards = list(hole) + board
    cat = best_category(cards)
    board_cat = best_category(board) if len(board) >= 5 else _partial_category(board)
    if cat > board_cat:
        return True
    if len(board) < 5:
        fd = flush_draw(list(hole), board)
        if fd and fd[1] == 4:
            return True
        if len(straight_draw_ranks(list(hole), board)) >= 2:
            return True
    return False
