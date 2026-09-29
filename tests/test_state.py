import pytest
from pydantic import ValidationError

from poker_agent.state import Action, ActionType, GameState, Street


def make_state(**kw):
    base = dict(
        players=[{"seat": s, "stack": 10.0} for s in range(1, 8)],
        button_seat=1,
        hero_seat=4,
        hole_cards="AhKd",
    )
    base.update(kw)
    return GameState.model_validate(base)


def test_card_strings_normalised():
    st = make_state(hole_cards="ah kd", board="Qs 7c 2d")
    assert st.hole_cards == ["Ah", "Kd"]
    assert st.board == ["Qs", "7c", "2d"]
    assert st.street is Street.FLOP


def test_positions_seven_handed():
    st = make_state(button_seat=6)
    assert st.positions == {
        6: "BTN", 7: "SB", 1: "BB", 2: "UTG", 3: "MP", 4: "HJ", 5: "CO"
    }
    assert st.small_blind_seat == 7
    assert st.big_blind_seat == 1
    assert st.hero_position == "HJ"


def test_heads_up_button_is_small_blind():
    st = make_state(players=[{"seat": 2, "stack": 10}, {"seat": 5, "stack": 10}],
                    button_seat=5, hero_seat=2)
    assert st.small_blind_seat == 5
    assert st.big_blind_seat == 2


def test_duplicate_card_between_hand_and_board():
    with pytest.raises(ValidationError, match="duplicate"):
        make_state(board="AhQs2d")


def test_bad_board_size():
    with pytest.raises(ValidationError, match="board"):
        make_state(board="Qs7c")


def test_action_on_future_street_rejected():
    with pytest.raises(ValidationError, match="board is still preflop"):
        make_state(actions=[{"street": "flop", "seat": 4, "type": "check"}])


def test_bet_requires_amount():
    with pytest.raises(ValidationError):
        Action(street=Street.FLOP, seat=1, type=ActionType.BET)


def test_unknown_seats_rejected():
    with pytest.raises(ValidationError, match="hero seat"):
        make_state(hero_seat=9)
    with pytest.raises(ValidationError, match="empty seat"):
        make_state(actions=[{"street": "preflop", "seat": 9, "type": "fold"}])


def test_compact_action_strings_and_player_mapping():
    st = GameState.model_validate(dict(
        players={1: 10, 2: 8.5, 3: 12},
        button_seat=1, hero_seat=1, hole_cards="AhKd", board="Qs7c2d",
        actions=["pf 1 raise 0.60", "pf 2 call", "preflop 3 call", "f 2 check", "flop 3 bet 0.90"],
    ))
    assert st.player(2).stack == 8.5
    assert st.actions[0] == Action(street=Street.PREFLOP, seat=1, type=ActionType.RAISE, amount=0.6)
    assert st.actions[4].street is Street.FLOP and st.actions[4].amount == 0.9


def test_bad_compact_action():
    with pytest.raises(ValidationError, match="expected"):
        make_state(actions=["preflop raise"])
