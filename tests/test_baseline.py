from pathlib import Path

import pytest

from poker_agent.analysis import analyze
from poker_agent.baseline import baseline_decision
from poker_agent.cli import load_state, main
from poker_agent.config import EquityConfig
from poker_agent.decision import Move
from poker_agent.rules import legal_actions
from poker_agent.state import GameState

CFG = EquityConfig(iterations=3000, seed=5)
FIX = Path(__file__).parent / "fixtures"


def spot(hero_cards, actions=(), hero=4, board="", button=1):
    return GameState.model_validate(dict(
        players=[{"seat": s, "stack": 10} for s in range(1, 8)],
        button_seat=button, hero_seat=hero, hole_cards=hero_cards, board=board,
        actions=[dict(zip(("street", "seat", "type", "amount"), a)) for a in actions],
    ))


def decide(st):
    return baseline_decision(st, analyze(st, CFG))


def is_legal(st, d):
    la = legal_actions(st)
    assert d.action.value in la.names()
    if d.action in (Move.BET, Move.RAISE):
        assert la.min_to <= d.amount <= la.max_to


def test_utg_opens_aces_folds_trash():
    st = spot("AhAd")
    d = decide(st)
    assert d.action is Move.RAISE and d.amount == 0.60
    is_legal(st, d)
    assert decide(spot("7h2d")).action is Move.FOLD


def test_isolation_raise_sizes_up_for_limpers():
    st = spot("AhQd", [("preflop", 4, "call"), ("preflop", 5, "call")], hero=6)
    d = decide(st)
    assert d.action is Move.RAISE and d.amount == 1.00  # 3bb + 2 limpers


def test_three_bets_premium():
    st = spot("KhKd", [("preflop", 4, "raise", 0.60)], hero=5)
    d = decide(st)
    assert d.action is Move.RAISE and d.amount == pytest.approx(2.10)
    is_legal(st, d)


def test_big_blind_checks_option():
    acts = [("preflop", s, "fold") for s in (4, 5, 6, 7)] + [("preflop", 1, "call"), ("preflop", 2, "call")]
    d = decide(spot("7h2d", acts, hero=3))
    assert d.action is Move.CHECK


def test_flop_draw_fixture_is_legal():
    st = load_state(FIX / "flop_draw.yaml")
    d = decide(st)
    is_legal(st, d)
    assert d.action in (Move.CALL, Move.RAISE, Move.ALL_IN)


def test_cli_decide_runs(capsys):
    assert main(["decide", "--baseline", str(FIX / "flop_draw.yaml")]) == 0
    out = capsys.readouterr().out
    assert "nut flush draw" in out and "=>" in out


def test_cli_reports_bad_state(tmp_path, capsys):
    f = tmp_path / "bad.yaml"
    f.write_text("players: [{seat: 1, stack: 10}, {seat: 2, stack: 10}]\n"
                 "button_seat: 1\nhero_seat: 2\nhole_cards: AhKd\n")
    assert main(["decide", "--baseline", str(f)]) == 2
    assert "not hero's turn" in capsys.readouterr().err
