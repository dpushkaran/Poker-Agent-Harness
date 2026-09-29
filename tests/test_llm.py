import json

import httpx
import pytest

from poker_agent.analysis import analyze
from poker_agent.cli import load_state
from poker_agent.config import EquityConfig
from poker_agent.llm.client import LLMError, OllamaClient
from poker_agent.llm.prompt import action_history, build_messages
from poker_agent.llm.schema import decision_schema
from poker_agent.rules import legal_actions

FIX = "tests/fixtures/flop_draw.yaml"


def client_with(handler):
    return OllamaClient(transport=httpx.MockTransport(handler))


def test_chat_sends_schema_and_parses_reply():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={
            "message": {"content": '{"action": "call"}'},
            "prompt_eval_count": 100, "eval_count": 20,
        })

    res = client_with(handler).chat([{"role": "user", "content": "hi"}], schema={"type": "object"},
                                    seed=1, max_tokens=50)
    assert res.content == '{"action": "call"}'
    assert res.output_tokens == 20
    assert seen["format"] == {"type": "object"}
    assert seen["stream"] is False and seen["think"] is False
    assert seen["options"] == {"temperature": 0.2, "seed": 1, "num_predict": 50}


def test_chat_errors_are_llm_errors():
    with pytest.raises(LLMError, match="500"):
        client_with(lambda r: httpx.Response(500, text="boom")).chat([])

    def down(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="cannot reach"):
        client_with(down).chat([])


def test_health_reports_missing_model():
    c = client_with(lambda r: httpx.Response(200, json={"models": [{"name": "other:7b"}]}))
    ok, msg = c.health()
    assert not ok and "ollama pull" in msg


def test_schema_limits_actions_to_legal_moves():
    st = load_state(FIX)
    schema = decision_schema(legal_actions(st))
    assert schema["properties"]["action"]["enum"] == ["fold", "call", "raise", "all_in"]
    assert list(schema["properties"])[0] == "reasoning"
    assert "$defs" not in schema


def test_prompt_contains_numbers_and_legal_actions():
    st = load_state(FIX)
    a = analyze(st, EquityConfig(iterations=2000, seed=1))
    system, user = build_messages(st, a)
    assert "0.10/0.20" in system["content"]
    text = user["content"]
    assert "MP(4) raises to 0.60" in text
    assert "BTN(7, hero) calls 0.60" in text
    assert "nut flush draw" in text
    assert "raise: amount (raise-to total) between 1.60 and 8.50" in text
    assert "c-bets almost every flop" in text


def test_action_history_all_streets():
    st = load_state(FIX)
    lines = action_history(st)
    assert lines[0].startswith("Blinds: SB(1) posts 0.10")
    assert lines[-1] == "Flop: MP(4) bets 0.80"
