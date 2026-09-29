from pathlib import Path

import pytest
import yaml

from poker_agent.analysis import analyze
from poker_agent.config import EquityConfig
from poker_agent.state import GameState

FIX = Path(__file__).parent / "fixtures"
CFG = EquityConfig(iterations=5000, seed=3)


def load(name):
    return GameState.model_validate(yaml.safe_load((FIX / name).read_text()))


def test_flop_draw_analysis():
    a = analyze(load("flop_draw.yaml"), CFG)
    assert a.hero_position == "BTN"
    assert a.odds.pot == 2.30  # 0.10 + 0.20 + 0.60*2 + 0.80
    assert a.odds.to_call == 0.80
    assert a.legal.can_raise and a.legal.min_to == 1.60
    assert "nut flush draw" in a.strength.draws
    [villain] = [o for o in a.opponents if o.in_equity_calc]
    assert villain.position == "MP" and villain.range_label == "MP open-raise"
    assert villain.narrowed_by == ["bet/raised on flop"]
    assert villain.note == "c-bets almost every flop"
    assert 0.3 < a.equity.equity < 0.6
    assert a.equity_margin == pytest.approx(a.equity.equity - a.odds.pot_odds, abs=1e-3)


def test_unopened_pot_uses_calling_range():
    st = GameState.model_validate(dict(
        players=[{"seat": s, "stack": 10} for s in range(1, 8)],
        button_seat=1, hero_seat=4, hole_cards="AhKd",
    ))
    a = analyze(st, CFG)
    assert a.equity_basis.startswith("unopened pot")
    assert a.players_to_act_behind == 6
    assert not any(o.in_equity_calc for o in a.opponents)


def test_range_override_and_pot_warning():
    st = load("flop_draw.yaml").model_copy(update={
        "opponent_ranges": {4: "KK+, AK"}, "reported_pot": 3.00,
    })
    a = analyze(st, CFG)
    villain = next(o for o in a.opponents if o.seat == 4)
    assert "user override" in villain.range_label
    assert any("differs from derived pot" in w for w in a.warnings)
