"""Command-line entry point: ``poker-agent decide|serve|eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from poker_agent.analysis import Analysis
from poker_agent.config import load_settings
from poker_agent.decide import Recommendation, decide
from poker_agent.decision import Decision
from poker_agent.rules import IllegalAction, NotHerosTurn
from poker_agent.state import GameState


def load_state(path: str | Path) -> GameState:
    return GameState.model_validate(yaml.safe_load(Path(path).read_text()))


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
        state = load_state(args.file)
        rec = decide(state, settings, use_llm=not args.baseline)
    except (IllegalAction, NotHerosTurn, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rec.model_dump(mode="json"), indent=2))
    else:
        print(format_recommendation(rec))
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
    d.set_defaults(func=cmd_decide)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
