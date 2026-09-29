import pytest

from poker_agent.cli import load_state, main
from poker_agent.config import EquityConfig, Settings
from poker_agent.decide import decide
from poker_agent.eval.runner import load_scenarios
from poker_agent.store import Store, export_scenario

FIX = "tests/fixtures/flop_draw.yaml"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "log.db")


def logged(store):
    st = load_state(FIX)
    rec = decide(st, Settings(equity=EquityConfig(iterations=500, seed=1)), use_llm=False)
    return store.log(st, rec, hand_id="h1"), st, rec


def test_log_and_get_roundtrip(store):
    rid, st, rec = logged(store)
    st2, rec2 = store.get(rid)
    assert st2 == st
    assert rec2.decision == rec.decision
    [row] = store.recent()
    assert row["hero_cards"] == "Ah5h" and row["source"] == "baseline"


def test_review(store):
    rid, *_ = logged(store)
    store.review(rid, followed=True, note="won a small pot")
    row = store.recent()[0]
    assert row["followed"] == 1 and row["note"] == "won a small pot"
    with pytest.raises(KeyError):
        store.review(999, note="x")


def test_export_scenario_joins_eval_set(store, tmp_path):
    rid, st, _ = logged(store)
    out = tmp_path / "scen" / "logged.yaml"
    out.parent.mkdir()
    export_scenario(store, rid, ["call", "raise"], out)
    [sc] = load_scenarios(out.parent)
    assert sc.id == f"logged-{rid}"
    assert sc.state.actions == st.actions and sc.state.button_seat == st.button_seat
    with pytest.raises(ValueError, match="already exists"):
        export_scenario(store, rid, ["call"], out)


def test_cli_log_commands(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[storage]\ndb_path = "{tmp_path / "x.db"}"\n[equity]\niterations = 500\n')
    assert main(["--config", str(cfg), "decide", "--baseline", "--log", FIX]) == 0
    assert main(["--config", str(cfg), "log", "--review", "1", "--followed", "no"]) == 0
    capsys.readouterr()
    assert main(["--config", str(cfg), "log"]) == 0
    assert "#1" in capsys.readouterr().out
