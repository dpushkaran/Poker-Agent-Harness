import json

import pytest
from fastapi.testclient import TestClient

from poker_agent.config import EquityConfig, Settings, StorageConfig
from poker_agent.llm.client import ChatResult
from poker_agent.web.app import create_app


class FakeLLM:
    model = "fake"

    def chat(self, messages, schema=None, temperature=0.2, think=False, seed=None):
        return ChatResult(json.dumps({"reasoning": "fine", "key_factors": ["x"], "action": "call",
                                      "amount": None, "confidence": "high"}), 0.01, 1, 1)

    def health(self):
        return True, "fake ready"


@pytest.fixture
def client(tmp_path):
    s = Settings(equity=EquityConfig(iterations=500, seed=1),
                 storage=StorageConfig(db_path=str(tmp_path / "w.db")))
    return TestClient(create_app(s, client=FakeLLM()))


def state(**kw):
    base = {"players": [{"seat": i, "stack": 10} for i in range(1, 8)], "button_seat": 1,
            "hero_seat": 7, "hole_cards": ["Ah", "Kd"], "board": [], "actions": []}
    base.update(kw)
    return base


def test_index_and_config(client):
    assert "Poker Agent" in client.get("/").text
    assert client.get("/api/config").json()["big_blind"] == 0.2
    assert client.get("/api/health").json()["ok"] is True


def test_table_view_for_next_seat(client):
    v = client.post("/api/table", json=state()).json()
    assert v["to_act"] == 4 and not v["hero_to_act"]
    assert v["options"]["call_amount"] == 0.2
    assert v["pot"] == 0.3
    assert [s["position"] for s in v["seats"]][:3] == ["BTN", "SB", "BB"]


def test_table_needs_board_after_round_closes(client):
    acts = [{"street": "preflop", "seat": s, "type": "fold"} for s in (4, 5, 6)]
    acts += [{"street": "preflop", "seat": 7, "type": "raise", "amount": 0.6},
             {"street": "preflop", "seat": 1, "type": "fold"},
             {"street": "preflop", "seat": 2, "type": "fold"},
             {"street": "preflop", "seat": 3, "type": "call"}]
    v = client.post("/api/table", json=state(actions=acts)).json()
    assert v["round_closed"] and v["needs_board"] == 3 and v["to_act"] is None


def test_table_rejects_illegal_history(client):
    r = client.post("/api/table", json=state(actions=[{"street": "preflop", "seat": 5, "type": "fold"}]))
    assert r.status_code == 422 and "expected seat 4" in r.json()["detail"]
    r = client.post("/api/table", json=state(hole_cards=["Ah", "Ah"]))
    assert r.status_code == 422 and "duplicate" in r.json()["detail"]


def test_decide_logs_and_review(client):
    acts = [{"street": "preflop", "seat": 4, "type": "raise", "amount": 0.6},
            {"street": "preflop", "seat": 5, "type": "fold"},
            {"street": "preflop", "seat": 6, "type": "fold"}]
    r = client.post("/api/decide", json={"state": state(actions=acts), "hand_id": "h1"}).json()
    assert r["recommendation"]["source"] == "llm"
    assert r["recommendation"]["decision"]["action"] == "call"
    assert client.post(f"/api/review/{r['id']}", json={"followed": True}).json()["ok"]
    [row] = client.get("/api/log").json()
    assert row["followed"] == 1 and row["hand_id"] == "h1"
    assert client.post("/api/review/999", json={"note": "x"}).status_code == 404


def test_decide_when_not_heros_turn(client):
    r = client.post("/api/decide", json={"state": state()})
    assert r.status_code == 422 and "not hero's turn" in r.json()["detail"]
