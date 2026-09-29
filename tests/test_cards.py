import pytest

from poker_agent.cards import (
    CardError,
    card_str,
    parse_card,
    parse_cards,
    rank_of,
    remaining_deck,
    suit_of,
)


def test_parse_card_roundtrip():
    for text in ["2c", "Td", "Jh", "As"]:
        assert card_str(parse_card(text)) == text


def test_parse_card_variants():
    assert parse_card("10h") == parse_card("Th")
    assert parse_card("ah") == parse_card("Ah")
    assert parse_card("K♥") == parse_card("Kh")
    assert parse_card("AS") == parse_card("As")


def test_card_ids():
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    assert rank_of(parse_card("Kd")) == 11
    assert suit_of(parse_card("Kd")) == 1


@pytest.mark.parametrize("text", ["AhKd", "Ah Kd", "Ah, Kd", ["Ah", "Kd"]])
def test_parse_cards_formats(text):
    assert [card_str(c) for c in parse_cards(text)] == ["Ah", "Kd"]


@pytest.mark.parametrize("bad", ["Ax", "1h", "AhK", "Zz"])
def test_invalid_cards(bad):
    with pytest.raises(CardError):
        parse_cards(bad)


def test_duplicates_rejected():
    with pytest.raises(CardError, match="duplicate"):
        parse_cards("AhAh")


def test_remaining_deck():
    deck = remaining_deck(parse_cards("AhKd"))
    assert len(deck) == 50
    assert parse_card("Ah") not in deck
