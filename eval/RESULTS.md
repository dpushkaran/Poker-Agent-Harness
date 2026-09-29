# Eval results log

Summaries of notable eval runs (full reports are written to `eval/results/`, which is not
tracked). 34 scenarios, 2 repeats per model, temperature 0.2, Apple M5 Pro 48GB.

## 2026-09-29 — prompt v1 vs v2

| setting | action agreement | sizing ok | valid 1st try | fallback | consistency | p50 s | p95 s |
|---|---|---|---|---|---|---|---|
| baseline (rules) | 100% | 100% | - | - | - | - | - |
| qwen3:30b-a3b, v1 | 88% | 29% | 100% | 4% | 90% | 4.0 | 15.8 |
| qwen3:30b-a3b, v2 | **94%** | **100%** | 99% | 1% | 96% | **3.3** | 6.3 |
| gpt-oss:20b, v2 | 96%* | 95% | 37% | 34% | 94% | 18.3 | 24.7 |

\* inflated: a third of gpt-oss answers failed validation and the baseline answered instead.
Its forced reasoning likely overran the 600-token output cap.

Findings:
- v1: preflop the model reasoned "equity > pot odds, so call", limping unopened pots and
  flat-calling KK; it also sized every bet at the $0.20 minimum. Postflop it was fine.
- v2 adds a PREFLOP situation line with chart guidance, suppresses pot-odds framing in
  unopened/limped pots, and lists concrete SIZE OPTIONS. This fixed the limping and sizing.
- Remaining qwen3 misses: occasionally opens Q9o UTG despite "not in range", calls a
  UTG shove with KQo once, and iso-raises 65s over two limpers once.
- Decision: default to qwen3:30b-a3b with prompt v2.
