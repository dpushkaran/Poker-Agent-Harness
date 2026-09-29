import pytest

from poker_agent.cards import parse_cards as pc
from poker_agent.math.strength import (
    board_texture, continues_vs_aggression, hand_strength, outs, straight_draw_ranks,
)


@pytest.mark.parametrize("hole,board,category,desc", [
    ("AhKd", "", "high card", "AK offsuit"),
    ("7h7d", "", "one pair", "pocket pair (77)"),
    ("AhKd", "Kc7s2d", "one pair", "top pair (Ks), A kicker"),
    ("Ah7d", "Kc7s2d", "one pair", "second pair (7s), A kicker"),
    ("Ah2h", "Kc7s2d", "one pair", "bottom pair (2s), A kicker"),
    ("QhQd", "Jc7s2d", "one pair", "overpair (QQ)"),
    ("3h3d", "Jc7s5d", "one pair", "underpair (33)"),
    ("7h7d", "Kc7s2d", "three of a kind", "set"),
    ("Ah7d", "Kc7s7c", "three of a kind", "trips (one hole card + board pair)"),
    ("Kh7d", "Kc7s2d", "two pair", "two pair (both hole cards)"),
    ("AhJh", "Kh7h2h", "flush", "nut flush"),
    ("QhJh", "Kh7h2h", "flush", "flush"),
    ("AhKd", "QsJc7d", "high card", "A-high, 2 overcards"),
    ("2c3d", "AhKhQhJhTh", "straight flush", "straight flush (playing the board)"),
])
def test_made_hands(hole, board, category, desc):
    hs = hand_strength(pc(hole), pc(board))
    assert hs.category == category
    assert hs.description == desc


def test_flush_draw_and_outs():
    hs = hand_strength(pc("Ah5h"), pc("Kh7h2c"))
    assert "nut flush draw" in hs.draws
    assert hs.outs >= 9
    assert hs.outs_equity_estimate == pytest.approx(hs.outs * 0.04)


def test_open_ended_and_gutshot():
    assert "open-ended straight draw" in hand_strength(pc("9h8d"), pc("7c6s2d")).draws
    assert "gutshot straight draw" in hand_strength(pc("9h8d"), pc("6c5s2d")).draws
    assert straight_draw_ranks(pc("9h8d"), pc("7c6s2d")) == {3, 8}  # 5 and T


def test_outs_ignore_board_only_improvements():
    # A 7 or 2 pairs the board for everyone, so only aces and kings count.
    assert outs(pc("AhKd"), pc("Kc7s2d")) == 5  # 3 aces + 2 kings -> two pair / trips


def test_board_texture():
    assert board_texture(pc("Kh7h2h")) == ["monotone", "high card K"]
    assert "paired board" in board_texture(pc("KhKd2c"))
    assert "straight possible" in board_texture(pc("9h8d7c"))
    assert "rainbow" in board_texture(pc("Kh7d2c"))


def test_continues_vs_aggression():
    board = pc("Kh7h2c")
    assert continues_vs_aggression(tuple(pc("AsKs")), board)  # top pair
    assert continues_vs_aggression(tuple(pc("Ah5h")), board)  # flush draw
    assert not continues_vs_aggression(tuple(pc("QsJd")), board)  # air
