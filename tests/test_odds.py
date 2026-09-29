import pytest

from poker_agent.math.odds import odds_info, pot_odds
from poker_agent.rules import replay
from poker_agent.state import GameState


def test_pot_odds_formula():
    # Facing a pot-sized bet: call 10 into 20 -> need 1/3
    assert pot_odds(10, 20) == pytest.approx(1 / 3)
    assert pot_odds(0, 20) == 0


def test_odds_info_facing_flop_bet():
    st = GameState.model_validate(dict(
        players=[{"seat": 1, "stack": 10}, {"seat": 2, "stack": 6}],
        button_seat=1, hero_seat=1, hole_cards="AhKd", board="Qs7c2d",
        actions=[
            {"street": "preflop", "seat": 1, "type": "raise", "amount": 0.60},
            {"street": "preflop", "seat": 2, "type": "call"},
            {"street": "flop", "seat": 2, "type": "bet", "amount": 1.20},
        ],
    ))
    o = odds_info(st, replay(st))
    assert o.pot == 2.40
    assert o.to_call == 1.20
    assert o.pot_odds == pytest.approx(1.2 / 3.6, abs=1e-4)
    assert o.bet_to_pot == 1.0
    assert o.mdf == 0.5
    assert o.effective_stack == 4.20  # villain's 6.00 - 0.60 - 1.20
    assert o.effective_stack_bb == 21.0
    assert o.pot_bb == 12.0
