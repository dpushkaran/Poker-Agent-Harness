import pytest

from poker_agent.rules import IllegalAction, NotHerosTurn, legal_actions, replay
from poker_agent.state import GameState

SEVEN = [{"seat": s, "stack": 10.0} for s in range(1, 8)]
# Button 1 -> SB 2, BB 3, UTG 4, MP 5, HJ 6, CO 7


def gs(actions=(), hero=4, board="", players=SEVEN, button=1, cards="AhKd"):
    return GameState.model_validate(dict(
        players=players, button_seat=button, hero_seat=hero,
        hole_cards=cards, board=board,
        actions=[dict(zip(("street", "seat", "type", "amount"), a)) for a in actions],
    ))


def pf(seat, typ, amount=None):
    return ("preflop", seat, typ, amount)


def test_blinds_posted_and_utg_first():
    t = replay(gs())
    assert t.pot == 30
    assert t.to_act == 4
    la = legal_actions(gs())
    assert la.can_fold and la.can_call and la.can_raise and not la.can_check
    assert la.call_amount == 0.20
    assert la.min_to == 0.40  # min raise to 2x BB
    assert la.max_to == 10.00


def test_min_raise_tracks_last_increment():
    # UTG raises to 0.60 (increment 0.40) -> next min raise is to 1.00
    st = gs([pf(4, "raise", 0.60)], hero=5)
    la = legal_actions(st)
    assert la.call_amount == 0.60
    assert la.min_to == 1.00


def test_raise_below_minimum_rejected():
    with pytest.raises(IllegalAction, match="minimum is 1.0"):
        replay(gs([pf(4, "raise", 0.60), pf(5, "raise", 0.80)], hero=6))


def test_out_of_turn_rejected():
    with pytest.raises(IllegalAction, match="expected seat 4"):
        replay(gs([pf(5, "fold")]))


def test_check_facing_bet_rejected():
    with pytest.raises(IllegalAction, match="cannot check"):
        replay(gs([pf(4, "check")]))


def test_big_blind_option_after_limps():
    acts = [pf(4, "call"), pf(5, "fold"), pf(6, "fold"), pf(7, "fold"), pf(1, "fold"), pf(2, "call")]
    la = legal_actions(gs(acts, hero=3))
    assert la.can_check and not la.can_fold and not la.can_call
    assert la.can_raise and la.min_to == 0.40


def test_flop_action_starts_left_of_button():
    acts = [pf(4, "call"), pf(5, "fold"), pf(6, "fold"), pf(7, "fold"),
            pf(1, "fold"), pf(2, "call"), pf(3, "check")]
    st = gs(acts, hero=2, board="Qs7c2d")
    t = replay(st)
    assert t.pot == 60
    assert t.to_act == 2
    la = legal_actions(st, t)
    assert la.can_check and la.can_bet and la.min_to == 0.20


def test_incomplete_preflop_before_flop_rejected():
    with pytest.raises(IllegalAction, match="preflop betting is incomplete"):
        replay(gs([pf(4, "call")], board="Qs7c2d"))


def test_not_heros_turn():
    with pytest.raises(NotHerosTurn, match="seat 4"):
        legal_actions(gs(hero=6))


def heads_up(actions=(), hero=1, board="", stacks=(10.0, 10.0)):
    players = [{"seat": 1, "stack": stacks[0]}, {"seat": 2, "stack": stacks[1]}]
    return gs(actions, hero=hero, board=board, players=players, button=1)


def test_heads_up_button_acts_first_preflop_last_postflop():
    assert replay(heads_up()).to_act == 1
    st = heads_up([pf(1, "call"), pf(2, "check")], hero=2, board="Qs7c2d")
    assert replay(st).to_act == 2


def test_short_all_in_does_not_reopen_betting():
    players = [{"seat": 1, "stack": 10}, {"seat": 2, "stack": 10},
               {"seat": 3, "stack": 10}, {"seat": 4, "stack": 1.20}]
    # BTN 1, SB 2, BB 3, UTG 4. Flop: SB bets 1.00, BB calls, UTG all-in 1.00->... short
    acts = [pf(4, "call"), pf(1, "call"), pf(2, "call"), pf(3, "check"),
            ("flop", 2, "bet", 0.60), ("flop", 3, "call"), ("flop", 4, "all_in"),
            ("flop", 1, "call")]
    # UTG had 1.00 left; all-in to 1.00 is +0.40 over 0.60 (< full raise of 0.60)
    st = gs(acts, hero=2, players=players, board="Qs7c2d")
    la = legal_actions(st)
    assert la.call_amount == 0.40
    assert not la.can_raise


def test_full_all_in_raise_reopens():
    players = [{"seat": 1, "stack": 10}, {"seat": 2, "stack": 10},
               {"seat": 3, "stack": 10}, {"seat": 4, "stack": 2.00}]
    acts = [pf(4, "call"), pf(1, "call"), pf(2, "call"), pf(3, "check"),
            ("flop", 2, "bet", 0.60), ("flop", 3, "call"), ("flop", 4, "all_in"),
            ("flop", 1, "call")]
    # UTG all-in to 1.80: +1.20 over 0.60 is a full raise
    la = legal_actions(gs(acts, hero=2, players=players, board="Qs7c2d"))
    assert la.can_raise and la.min_to == 3.00


def test_short_stack_call_capped_at_stack():
    st = heads_up([pf(1, "raise", 5.00)], hero=2, stacks=(10.0, 3.0))
    la = legal_actions(st)
    assert la.call_amount == 2.80  # 3.00 stack minus 0.20 posted
    assert not la.can_raise  # can only call all-in


def test_cannot_raise_when_everyone_else_all_in():
    st = heads_up([pf(1, "all_in")], hero=2, stacks=(3.0, 10.0))
    la = legal_actions(st)
    assert la.can_call and not la.can_raise


def test_hand_over_after_folds():
    acts = [pf(s, "fold") for s in (4, 5, 6, 7, 1, 2)]
    with pytest.raises(NotHerosTurn, match="hand is over"):
        legal_actions(gs(acts, hero=3))


def test_bet_exceeding_stack_rejected():
    with pytest.raises(IllegalAction, match="exceeds stack"):
        replay(gs([pf(4, "raise", 12.00)]))


def test_snap_rounds_and_clamps():
    la = legal_actions(gs())
    assert la.snap(0.63) == 0.60
    assert la.snap(0.10) == 0.40  # below min raise
    assert la.snap(50) == 10.00  # above stack


def test_legal_actions_for_other_seat():
    st = gs([pf(4, "raise", 0.60)], hero=7)
    la = legal_actions(st, seat=5)
    assert la.call_amount == 0.60 and la.min_to == 1.00
    with pytest.raises(NotHerosTurn, match="not seat 6's turn"):
        legal_actions(st, seat=6)
