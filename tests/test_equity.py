import time

import pytest

from poker_agent.cards import parse_cards as pc
from poker_agent.math.equity import EquityError, equity
from poker_agent.math.ranges import parse_range as pr


def mc(hero, board, *ranges, n=20000):
    return equity(pc(hero), pc(board), [pr(r) for r in ranges], iterations=n, seed=7)


def test_aces_vs_kings():
    assert mc("AhAs", "", "KK").equity == pytest.approx(0.82, abs=0.02)


def test_aks_vs_queens():
    assert mc("AhKh", "", "QQ").equity == pytest.approx(0.46, abs=0.02)


def test_flush_draw_vs_top_pair_on_flop():
    r = mc("9h8h", "Kh7h2c", "KsQd")
    assert r.method == "monte_carlo"
    # Exact enumeration gives 0.3980 (flush outs plus runner-runner straights/two pair)
    assert r.equity == pytest.approx(0.398, abs=0.015)


def test_turn_heads_up_is_exact():
    r = equity(pc("9h8h"), pc("Kh7h2c3d"), [pr("KsQd")])
    assert r.method == "exact"
    assert r.equity == pytest.approx(9 / 44, abs=1e-9)


def test_river_split_pot():
    r = equity(pc("2c3d"), pc("AhKhQhJhTh"), [pr("any")])
    assert r.tie == 1.0 and r.equity == 1.0 / 2


def test_multiway_lowers_equity():
    hu = mc("JhJd", "", "any").equity
    three = mc("JhJd", "", "any", "any").equity
    assert three < hu - 0.1


def test_dead_card_removal_empties_range():
    with pytest.raises(EquityError, match="empty"):
        equity(pc("AhAs"), [], [pr("AhAs")])


def test_speed_budget():
    t = time.perf_counter()
    mc("AhKd", "Qs7c2d", "any", "any", n=20000)
    assert time.perf_counter() - t < 2.0
