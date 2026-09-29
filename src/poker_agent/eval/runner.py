"""Run labeled scenarios through one or more models and score the answers.

A scenario is a hand state plus the set of acceptable actions (and
optionally a best action and a sizing range). Each model/prompt setting
is scored on agreement with the labels, output validity, consistency
across repeats and latency; the baseline policy is scored alongside as a
reference.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from poker_agent.config import Settings
from poker_agent.decide import ChatClient, Recommendation, decide
from poker_agent.decision import Move
from poker_agent.state import GameState

DEFAULT_SCENARIO_DIR = Path("eval/scenarios")
DEFAULT_RESULTS_DIR = Path("eval/results")


class Scenario(BaseModel):
    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    state: GameState
    acceptable: list[Move]
    best: Move | None = None
    amount: tuple[float, float] | None = None  # acceptable bet/raise 'to' range

    @model_validator(mode="before")
    @classmethod
    def _compact(cls, data: dict) -> dict:
        """Build `state` from the compact scenario fields."""
        if "state" in data:
            return data
        data = dict(data)
        data["state"] = {
            "players": data.pop("players", {s: 10.0 for s in range(1, 8)}),
            "button_seat": data.pop("button", 1),
            "hero_seat": data.pop("hero"),
            "hole_cards": data.pop("cards"),
            "board": data.pop("board", ""),
            "actions": data.pop("actions", []),
            "opponent_notes": data.pop("opponent_notes", {}),
        }
        return data


def load_scenarios(directory: Path = DEFAULT_SCENARIO_DIR, tags: list[str] | None = None,
                   ids: list[str] | None = None) -> list[Scenario]:
    out = []
    for path in sorted(directory.glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text()) or []:
            try:
                sc = Scenario.model_validate(raw)
            except ValueError as e:
                raise ValueError(f"{path.name}: scenario {raw.get('id')!r}: {e}") from e
            if tags and not set(tags) & set(sc.tags):
                continue
            if ids and sc.id not in ids:
                continue
            out.append(sc)
    return out


class RunRecord(BaseModel):
    scenario: str
    setting: str  # model / prompt label, or "baseline"
    repeat: int
    action: Move
    amount: float | None
    source: str
    action_ok: bool
    best_ok: bool | None
    sizing_ok: bool | None
    valid_first_try: bool | None
    latency_s: float
    reasoning: str


def score(sc: Scenario, rec: Recommendation, setting: str, repeat: int) -> RunRecord:
    d = rec.decision
    amount = d.amount
    if d.action is Move.ALL_IN:
        amount = rec.analysis.legal.hero_stack + rec.analysis.legal.hero_committed_street
    sizing_ok = None
    if sc.amount and d.action in (Move.BET, Move.RAISE, Move.ALL_IN) and d.action in sc.acceptable:
        lo, hi = sc.amount
        sizing_ok = lo - 1e-9 <= amount <= hi + 1e-9
    return RunRecord(
        scenario=sc.id,
        setting=setting,
        repeat=repeat,
        action=d.action,
        amount=amount,
        source=rec.source,
        action_ok=d.action in sc.acceptable,
        best_ok=(d.action == sc.best) if sc.best else None,
        sizing_ok=sizing_ok,
        valid_first_try=rec.valid_first_try,
        latency_s=round(rec.total_latency_s, 2),
        reasoning=d.reasoning,
    )


def run_eval(
    scenarios: list[Scenario],
    settings: Settings,
    models: list[str],
    repeats: int = 1,
    client_factory=None,
    progress=print,
) -> list[RunRecord]:
    """Score the baseline once and each model `repeats` times per scenario."""
    settings = settings.model_copy(deep=True)
    if not settings.equity.seed:
        settings.equity.seed = 1  # identical analysis numbers for every model
    records: list[RunRecord] = []
    for sc in scenarios:
        rec = decide(sc.state, settings, use_llm=False)
        records.append(score(sc, rec, "baseline", 0))

    for model in models:
        s = settings.model_copy(deep=True)
        s.llm.model = model
        label = setting_label(s)
        client: ChatClient | None = client_factory(s) if client_factory else None
        for sc in scenarios:
            for r in range(repeats):
                rec = decide(sc.state, s, client=client, seed=r + 1)
                rr = score(sc, rec, label, r)
                records.append(rr)
                mark = "ok " if rr.action_ok else "BAD"
                amt = f" {rr.amount:.2f}" if rr.amount is not None else ""
                progress(f"[{label}] {mark} {sc.id} #{r + 1}: {rr.action.value}{amt} "
                         f"({rr.source}, {rr.latency_s:.1f}s)")
    return records


def setting_label(s: Settings) -> str:
    flags = [s.llm.prompt_version]
    if s.llm.think:
        flags.append("think")
    if not s.llm.constrain_actions:
        flags.append("unconstrained")
    return f"{s.llm.model} [{', '.join(flags)}]"


def _rate(values: list[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def summarize(records: list[RunRecord]) -> list[dict]:
    rows = []
    for setting in dict.fromkeys(r.setting for r in records):
        rs = [r for r in records if r.setting == setting]
        by_sc: dict[str, list[RunRecord]] = {}
        for r in rs:
            by_sc.setdefault(r.scenario, []).append(r)
        consistency = [
            Counter(x.action for x in group).most_common(1)[0][1] / len(group)
            for group in by_sc.values()
        ]
        lat = sorted(r.latency_s for r in rs)
        is_llm = setting != "baseline"
        rows.append({
            "setting": setting,
            "runs": len(rs),
            "action_agreement": _rate([r.action_ok for r in rs]),
            "best_match": _rate([r.best_ok for r in rs]),
            "sizing_ok": _rate([r.sizing_ok for r in rs]),
            "valid_first_try": _rate([r.valid_first_try for r in rs]) if is_llm else None,
            "fallback_rate": _rate([r.source == "fallback" for r in rs]) if is_llm else None,
            "consistency": statistics.mean(consistency) if is_llm else None,
            "latency_p50": statistics.median(lat) if is_llm else None,
            "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if is_llm else None,
        })
    return rows


def render_report(records: list[RunRecord], scenarios: list[Scenario], settings: Settings,
                  repeats: int) -> str:
    rows = summarize(records)
    lines = [
        f"# Eval report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{len(scenarios)} scenarios, {repeats} repeat(s) per model. Temperature "
        f"{settings.llm.temperature}, equity iterations {settings.equity.iterations}.",
        "",
        "| setting | runs | action agreement | best match | sizing ok | valid 1st try "
        "| fallback | consistency | p50 s | p95 s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lat50 = "-" if r["latency_p50"] is None else f"{r['latency_p50']:.1f}"
        lat95 = "-" if r["latency_p95"] is None else f"{r['latency_p95']:.1f}"
        lines.append(
            f"| {r['setting']} | {r['runs']} | {_pct(r['action_agreement'])} | "
            f"{_pct(r['best_match'])} | {_pct(r['sizing_ok'])} | {_pct(r['valid_first_try'])} | "
            f"{_pct(r['fallback_rate'])} | {_pct(r['consistency'])} | {lat50} | {lat95} |"
        )
    settings_order = [r["setting"] for r in rows]
    lines += ["", "## Per scenario", "",
              "| scenario | acceptable | " + " | ".join(settings_order) + " |",
              "|---|---|" + "---|" * len(settings_order)]
    for sc in scenarios:
        cells = []
        for setting in settings_order:
            rs = [r for r in records if r.setting == setting and r.scenario == sc.id]
            counts = Counter(
                r.action.value + (f" {r.amount:.2f}" if r.amount is not None and r.action is not Move.ALL_IN else "")
                for r in rs
            )
            text = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.most_common())
            bad = any(not r.action_ok for r in rs)
            cells.append(f"**{text}** ✗" if bad else text)
        lines.append(f"| {sc.id} | {', '.join(m.value for m in sc.acceptable)} | "
                     + " | ".join(cells) + " |")
    misses = [r for r in records if not r.action_ok and r.setting != "baseline"]
    if misses:
        lines += ["", "## Disagreements (model reasoning)", ""]
        for r in misses:
            lines.append(f"- **{r.scenario}** ({r.setting}) → {r.action.value}: {r.reasoning}")
    return "\n".join(lines) + "\n"


def write_results(records: list[RunRecord], report: str, out_dir: Path = DEFAULT_RESULTS_DIR,
                  name: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    md, csv_path = out_dir / f"{stem}.md", out_dir / f"{stem}.csv"
    md.write_text(report)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(RunRecord.model_fields))
        w.writeheader()
        for r in records:
            w.writerow(r.model_dump(mode="json"))
    return md, csv_path
