import json

import pytest

from poker_agent.cli import load_state
from poker_agent.config import EquityConfig, Settings
from poker_agent.decide import decide
from poker_agent.decision import Move
from poker_agent.llm.client import ChatResult, LLMError
from poker_agent.rules import legal_actions
from poker_agent.validate import validate_decision

FIX = "tests/fixtures/flop_draw.yaml"
SETTINGS = Settings(equity=EquityConfig(iterations=2000, seed=1))


def reply(action, amount=None, **kw):
    return json.dumps({"reasoning": "because", "key_factors": [], "action": action,
                       "amount": amount, "confidence": "medium", **kw})


class FakeClient:
    model = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, schema=None, temperature=0.2, think=False, seed=None):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResult(content=r, latency_s=0.01, prompt_tokens=1, output_tokens=1)


@pytest.fixture
def legal():
    return legal_actions(load_state(FIX))  # fold, call 0.80, raise 1.60-8.50


def test_valid_raise_passes(legal):
    r = validate_decision(reply("raise", 2.40), legal)
    assert r.ok and r.decision.amount == 2.40 and not r.adjustments


def test_bet_facing_bet_becomes_raise(legal):
    r = validate_decision(reply("bet", 2.43), legal)
    assert r.ok and r.decision.action is Move.RAISE and r.decision.amount == 2.40
    assert len(r.adjustments) == 2


def test_illegal_check_rejected(legal):
    r = validate_decision(reply("check"), legal)
    assert not r.ok and "not legal" in r.errors[0]


def test_amount_bounds(legal):
    assert not validate_decision(reply("raise", 0.90), legal).ok  # way under 1.60
    assert not validate_decision(reply("raise"), legal).ok  # missing amount
    r = validate_decision(reply("raise", 1.50), legal)  # within tolerance -> clamped
    assert r.ok and r.decision.amount == 1.60
    r = validate_decision(reply("raise", 8.50), legal)
    assert r.decision.action is Move.ALL_IN and r.decision.amount is None


def test_garbage_rejected(legal):
    assert "not valid JSON" in validate_decision("nope", legal).errors[0]
    assert "schema" in validate_decision('{"action": "dance"}', legal).errors[0]


def test_decide_uses_llm_answer():
    client = FakeClient(reply("call"))
    rec = decide(load_state(FIX), SETTINGS, client=client)
    assert rec.source == "llm" and rec.decision.action is Move.CALL
    assert rec.valid_first_try is True and rec.attempts == 1


def test_decide_retries_then_succeeds():
    client = FakeClient(reply("check"), reply("raise", 2.40))
    rec = decide(load_state(FIX), SETTINGS, client=client)
    assert rec.source == "llm" and rec.attempts == 2
    assert rec.valid_first_try is False
    assert "invalid" in client.calls[1][-1]["content"]


def test_decide_falls_back_to_baseline():
    rec = decide(load_state(FIX), SETTINGS, client=FakeClient(reply("check"), reply("check")))
    assert rec.source == "fallback" and rec.decision == rec.baseline
    rec = decide(load_state(FIX), SETTINGS, client=FakeClient(LLMError("down")))
    assert rec.source == "fallback" and rec.errors == ["down"]
