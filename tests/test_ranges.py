import pytest

from poker_agent.cards import parse_cards
from poker_agent.math.ranges import (
    BB_DEFEND_RANGE, CALL_RANGE, FOUR_BET_RANGE, LIMP_RANGE, OPEN_RANGES,
    THREE_BET_RANGE, RangeError, hand_class, infer_preflop_range, parse_range,
    remove_dead,
)
from poker_agent.state import GameState


def classes(text):
    return {hand_class(*c) for c in parse_range(text)}


@pytest.mark.parametrize("text,count", [
    ("AA", 6), ("AKs", 4), ("AKo", 12), ("AK", 16), ("QQ+", 18),
    ("99-66", 24), ("ATs+", 16), ("A5s-A2s", 16), ("T9s-65s", 20), ("any", 1326),
])
def test_combo_counts(text, count):
    assert len(parse_range(text)) == count


def test_class_expansion():
    assert classes("ATs+") == {"ATs", "AJs", "AQs", "AKs"}
    assert classes("T9s-76s") == {"T9s", "98s", "87s", "76s"}
    assert classes("KQo, 22") == {"KQo", "22"}


def test_weights():
    r = parse_range("AA, KK:0.5")
    assert r[max(parse_cards("KhKs")), min(parse_cards("KhKs"))] == 0.5


@pytest.mark.parametrize("bad", ["AKx", "AKs-QJo", "AAs", "Z9", "AA:2"])
def test_bad_ranges(bad):
    with pytest.raises(RangeError):
        parse_range(bad)


@pytest.mark.parametrize("text", [
    *OPEN_RANGES.values(), LIMP_RANGE, CALL_RANGE, BB_DEFEND_RANGE, THREE_BET_RANGE, FOUR_BET_RANGE,
])
def test_default_ranges_parse(text):
    assert parse_range(text)


def test_open_ranges_widen_by_position():
    sizes = [len(parse_range(OPEN_RANGES[p])) for p in ("UTG", "MP", "HJ", "CO", "BTN")]
    assert sizes == sorted(sizes)


def test_remove_dead():
    r = remove_dead(parse_range("AA"), set(parse_cards("Ah")))
    assert len(r) == 3


def test_infer_preflop_ranges():
    st = GameState.model_validate(dict(
        players=[{"seat": s, "stack": 10} for s in range(1, 8)],
        button_seat=1, hero_seat=7, hole_cards="AhKd",
        actions=[
            {"street": "preflop", "seat": 4, "type": "call"},
            {"street": "preflop", "seat": 5, "type": "raise", "amount": 0.80},
            {"street": "preflop", "seat": 6, "type": "raise", "amount": 2.40},
        ],
    ))
    assert infer_preflop_range(st, 4)[0] == "UTG limp"
    assert infer_preflop_range(st, 5) == ("MP open-raise", OPEN_RANGES["MP"])
    assert infer_preflop_range(st, 6) == ("HJ 3-bet", THREE_BET_RANGE)
    assert infer_preflop_range(st, 3)[1] == "any"
