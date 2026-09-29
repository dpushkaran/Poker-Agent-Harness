import json
from pathlib import Path

from poker_agent.config import EquityConfig, Settings
from poker_agent.eval.runner import (
    load_scenarios, render_report, run_eval, summarize, write_results,
)
from poker_agent.llm.client import ChatResult
from poker_agent.rules import legal_actions

SCENARIOS = Path(__file__).parent.parent / "eval" / "scenarios"
SETTINGS = Settings(equity=EquityConfig(iterations=1000, seed=2))


def test_all_scenarios_are_valid_hero_decisions():
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) >= 30
    assert len({s.id for s in scenarios}) == len(scenarios)
    for sc in scenarios:
        names = legal_actions(sc.state).names()  # raises if it isn't hero's turn
        for move in sc.acceptable:
            assert move.value in names, f"{sc.id}: {move.value} not legal ({names})"


def test_filters():
    assert all("river" in s.tags for s in load_scenarios(SCENARIOS, tags=["river"]))
    assert [s.id for s in load_scenarios(SCENARIOS, ids=["pf-utg-aa-open"])] == ["pf-utg-aa-open"]


class AlwaysFold:
    model = "folder"

    def chat(self, messages, schema=None, temperature=0.2, think=False, seed=None, max_tokens=None):
        action = "fold" if "fold" in schema["properties"]["action"]["enum"] else "check"
        return ChatResult(json.dumps({"reasoning": "r", "key_factors": [], "action": action,
                                      "amount": None, "confidence": "low"}), 0.01, 1, 1)


def test_run_eval_scores_and_reports(tmp_path):
    scenarios = load_scenarios(SCENARIOS, ids=["pf-utg-aa-open", "pf-utg-72o-fold"])
    records = run_eval(scenarios, SETTINGS, ["folder"], repeats=2,
                       client_factory=lambda s: AlwaysFold(), progress=lambda *_: None)
    rows = {r["setting"]: r for r in summarize(records)}
    assert rows["baseline"]["action_agreement"] == 1.0
    model = rows["folder [v1]"]
    assert model["runs"] == 4 and model["action_agreement"] == 0.5
    assert model["consistency"] == 1.0 and model["fallback_rate"] == 0.0
    report = render_report(records, scenarios, SETTINGS, 2)
    assert "pf-utg-aa-open" in report and "fold×2" in report
    md, csv_path = write_results(records, report, tmp_path, "t")
    assert md.exists() and len(csv_path.read_text().splitlines()) == 1 + 2 + 4
