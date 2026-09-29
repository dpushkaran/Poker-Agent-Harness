"""Command-line entry point: ``poker-agent decide|serve|eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from poker_agent.analysis import Analysis
from poker_agent.config import Settings, load_settings
from poker_agent.decide import Recommendation, decide
from poker_agent.decision import Decision
from poker_agent.rules import IllegalAction, NotHerosTurn
from poker_agent.state import GameState


def load_state(path: str | Path, settings: Settings | None = None) -> GameState:
    """Load a hand file; hands without a `blinds` entry use the configured blinds."""
    data = yaml.safe_load(Path(path).read_text())
    if settings is not None and "blinds" not in data:
        g = settings.game
        data["blinds"] = {"small": g.small_blind, "big": g.big_blind, "increment": g.chip}
    return GameState.model_validate(data)


def format_analysis(a: Analysis) -> str:
    o, s, e = a.odds, a.strength, a.equity
    lines = [
        f"{a.street.value.upper()}  hero {a.hero_position} {' '.join(a.hero_cards)}"
        + (f"  board {' '.join(a.board)}" if a.board else ""),
        f"  pot {o.pot:.2f} ({o.pot_bb:g}bb)  to call {o.to_call:.2f}  stack {a.hero_stack:.2f} "
        f"({a.hero_stack_bb:g}bb)  SPR {o.spr:g}",
        f"  hand: {s.description}" + (f"  draws: {', '.join(s.draws)}" if s.draws else "")
        + (f"  outs: {s.outs}" if s.outs else ""),
        f"  equity {e.equity:.1%} ({a.equity_basis}; vs random {a.equity_vs_random:.1%})"
        + (f"  pot odds {o.pot_odds:.1%}  margin {a.equity_margin:+.1%}" if o.pot_odds else ""),
        f"  legal: {', '.join(a.legal.names())}"
        + (f"  (bet/raise to {a.legal.min_to:.2f}-{a.legal.max_to:.2f})" if a.legal.min_to else ""),
    ]
    for opp in a.opponents:
        if opp.in_equity_calc:
            lines.append(f"  vs seat {opp.seat} {opp.position}: {opp.range_label} "
                         f"(~{opp.range_combos:g} combos)")
    lines += [f"  ! {w}" for w in a.warnings]
    return "\n".join(lines)


def format_decision(d: Decision, source: str) -> str:
    amt = f" to {d.amount:.2f}" if d.amount is not None else ""
    lines = [f"\n=> {d.action.value.upper()}{amt}   [{source}, confidence {d.confidence.value}]",
             f"   {d.reasoning}"]
    if d.key_factors:
        lines.append(f"   factors: {'; '.join(d.key_factors)}")
    return "\n".join(lines)


def format_recommendation(rec: Recommendation) -> str:
    source = f"{rec.source}: {rec.model}" if rec.model and rec.source == "llm" else rec.source
    out = [format_analysis(rec.analysis), format_decision(rec.decision, source)]
    if rec.source == "llm" and rec.baseline.action != rec.decision.action:
        b = rec.baseline
        out.append(f"   (baseline would {b.action.value}"
                   + (f" to {b.amount:.2f}" if b.amount is not None else "") + ")")
    out += [f"   note: {x}" for x in rec.adjustments]
    if rec.source == "fallback":
        out += [f"   llm problem: {x}" for x in rec.errors]
    if rec.model:
        out.append(f"   [{rec.total_latency_s:.1f}s total, {rec.attempts} attempt(s)]")
    return "\n".join(out)


def cmd_decide(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    if args.model:
        settings.llm.model = args.model
    if args.think:
        settings.llm.think = True
    try:
        state = load_state(args.file, settings)
        rec = decide(state, settings, use_llm=not args.baseline)
    except (IllegalAction, NotHerosTurn, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.log:
        from poker_agent.store import Store

        rec_id = Store(settings.storage.db_path).log(state, rec)
        print(f"logged as decision #{rec_id}", file=sys.stderr)
    if args.json:
        print(json.dumps(rec.model_dump(mode="json"), indent=2))
    else:
        print(format_recommendation(rec))
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    from poker_agent.store import Store

    store = Store(load_settings(args.config).storage.db_path)
    if args.review is not None:
        followed = {"yes": True, "no": False}.get(args.followed) if args.followed else None
        store.review(args.review, followed=followed, note=args.note)
        return 0
    for r in store.recent(args.limit):
        amt = f" {r['amount']:.2f}" if r["amount"] is not None else ""
        followed = {1: " followed", 0: " ignored"}.get(r["followed"], "")
        note = f"  -- {r['note']}" if r["note"] else ""
        print(f"#{r['id']:<4} {r['created_at']}  {r['street']:<7} {r['hero_cards']:<5} "
              f"{r['board']:<10} {r['action']}{amt} [{r['source']}]{followed}{note}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from poker_agent.store import Store, export_scenario

    store = Store(load_settings(args.config).storage.db_path)
    sc = export_scenario(store, args.id, args.acceptable, Path(args.out), best=args.best,
                         scenario_id=args.name)
    print(f"added scenario {sc['id']} to {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from poker_agent.eval.runner import (
        load_scenarios, render_report, run_eval, summarize, write_results,
    )

    settings = load_settings(args.config)
    if args.prompt:
        settings.llm.prompt_version = args.prompt
    settings.llm.think = args.think
    if args.unconstrained:
        settings.llm.constrain_actions = False
    scenarios = load_scenarios(Path(args.scenarios), tags=args.tags, ids=args.ids)
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 2
    models = [] if args.baseline_only else (args.models or [settings.llm.model])
    records = run_eval(scenarios, settings, models, repeats=args.repeats)
    report = render_report(records, scenarios, settings, args.repeats)
    md, csv_path = write_results(records, report)
    for row in summarize(records):
        agree = row["action_agreement"]
        print(f"{row['setting']}: {agree:.0%} action agreement over {row['runs']} runs")
    print(f"report: {md}\nruns:   {csv_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import socket

    import uvicorn

    from poker_agent.web.app import create_app

    settings = load_settings(args.config)
    app = create_app(settings)
    if args.host == "0.0.0.0":
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                print(f"Open http://{s.getsockname()[0]}:{args.port} on your phone "
                      "(same Wi-Fi network)")
        except OSError:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poker-agent", description=__doc__)
    p.add_argument("--config", default=None, help="path to config.toml")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("decide", help="recommend an action for a hand described in YAML")
    d.add_argument("file")
    d.add_argument("--json", action="store_true", help="print machine-readable JSON")
    d.add_argument("--baseline", action="store_true", help="skip the LLM, use the rule policy")
    d.add_argument("--model", help="override the Ollama model from config")
    d.add_argument("--think", action="store_true", help="enable model thinking (slower)")
    d.add_argument("--log", action="store_true", help="save the decision to the SQLite log")
    d.set_defaults(func=cmd_decide)

    e = sub.add_parser("eval", help="score models on labeled scenarios")
    e.add_argument("--models", type=lambda v: v.split(","), help="comma-separated Ollama models")
    e.add_argument("--prompt", help="prompt version, e.g. v1")
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--think", action="store_true", help="enable model thinking")
    e.add_argument("--unconstrained", action="store_true",
                   help="don't restrict the schema to legal actions (measures raw legality)")
    e.add_argument("--tags", type=lambda v: v.split(","), help="only scenarios with these tags")
    e.add_argument("--ids", type=lambda v: v.split(","), help="only these scenario ids")
    e.add_argument("--baseline-only", action="store_true")
    e.add_argument("--scenarios", default="eval/scenarios")
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="run the web UI")
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow phones on the LAN")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    lg = sub.add_parser("log", help="list logged decisions, or review one")
    lg.add_argument("--limit", type=int, default=20)
    lg.add_argument("--review", type=int, metavar="ID", help="decision id to annotate")
    lg.add_argument("--followed", choices=["yes", "no"])
    lg.add_argument("--note", help="review note, e.g. the outcome")
    lg.set_defaults(func=cmd_log)

    ex = sub.add_parser("export", help="turn a logged decision into an eval scenario")
    ex.add_argument("id", type=int)
    ex.add_argument("--acceptable", required=True, type=lambda v: v.split(","))
    ex.add_argument("--best")
    ex.add_argument("--name", help="scenario id (default logged-<id>)")
    ex.add_argument("--out", default="eval/scenarios/logged.yaml")
    ex.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
