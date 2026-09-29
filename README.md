# Poker-Agent-Harness

A strategy aid for a friendly home no-limit Texas Hold'em game. You enter the hand as it plays out, and a **locally run LLM** recommends fold / check / call / bet / raise with its reasoning.

All the precise math is done by ordinary, tested code, not the model. That includes the legal actions, pot odds, SPR, hand strength, outs, and equity against estimated opponent ranges. The model reasons over those numbers. Its answer is then validated against the legal moves before it is shown. If the model is down or keeps answering illegally, a deterministic baseline policy answers instead and is labeled as such.

Built for $0.10/$0.20 blinds, $10 buy-ins and up to 7 players. These defaults live in `config.toml`.

## Setup (macOS)

```sh
brew install uv ollama
ollama serve                    # or: brew services start ollama
ollama pull qwen3:30b-a3b       # ~18 GB; runs on Apple Silicon GPU via Metal
uv sync
uv run pytest                   # optional
```

## Using it at the table

```sh
uv run poker-agent serve --host 0.0.0.0
```

The command prints a LAN address. Open it on your phone while it's on the same Wi-Fi network.

1. **Table setup** (once per session):
   - Tick the seats that are playing and set each stack.
   - Mark your seat and the button.
   - Tap ✎ to add a read ("calls too much") or a range override for a player.
2. **Your cards:** tap the two card slots.
3. **Actions:** the page always shows who is next to act and only the buttons that are legal for them. Enter what each player does. **Everyone folds to me** is a shortcut preflop.
4. When it's your turn, tap **Get recommendation**. Local models usually answer in about 4–8 s.
5. Record what you actually did. The page then asks for the flop, turn and river cards as each betting round closes.
6. **New hand** moves the button one seat. Update the stacks in setup when they change.

The pot is derived from the action history, so it can't drift. **Undo** steps back one action or card.

## CLI

```sh
uv run poker-agent decide tests/fixtures/flop_draw.yaml          # recommendation for a hand file
uv run poker-agent decide hand.yaml --baseline                  # rule-based policy only, no LLM
uv run poker-agent decide hand.yaml --model gpt-oss:20b --think --log
uv run poker-agent log                                           # recent logged decisions
uv run poker-agent log --review 12 --followed no --note "lost to a set"
```

A hand file is short YAML:

```yaml
players: {1: 10.00, 2: 12.40, 3: 8.00, 4: 10.00, 5: 15.20, 6: 10.00, 7: 9.10}
button_seat: 7
hero_seat: 7
hole_cards: Ah5h
board: Kh8h3c
actions: [pf 3 fold, pf 4 raise 0.60, pf 5 fold, pf 6 fold, pf 7 call, pf 1 fold, pf 2 fold,
          flop 4 bet 0.80]
opponent_notes: {4: "c-bets almost every flop"}
```

Notation:
- Stacks are the stacks at the start of the hand.
- `bet`/`raise` amounts are "raise to" totals for that street.
- Blinds are posted automatically.

## Evaluating decision quality

`eval/scenarios/*.yaml` holds labeled spots. Each one lists the acceptable actions, and some also give a best action and a sizing range.

```sh
uv run poker-agent eval                                   # default model vs the baseline
uv run poker-agent eval --models qwen3:30b-a3b,gpt-oss:20b --repeats 3
uv run poker-agent eval --think --tags river             # thinking mode, river spots only
uv run poker-agent eval --unconstrained                   # measure raw action legality
```

Each run writes a markdown report and a CSV to `eval/results/`. The report covers:
- agreement with the labels
- sizing
- valid-on-first-try rate
- fallbacks
- consistency across repeats
- latency
- the model's reasoning wherever it disagreed with a label

To grow the set from real play, export any logged decision as a scenario:

```sh
uv run poker-agent export 12 --acceptable call,raise --best call
```

## How it works

| Module | Role |
|---|---|
| `state.py` | Hand input models, seat positions (2–9 handed) |
| `rules.py` | Replays the betting history in integer cents; enforces turn order, min-raise and short all-in rules; computes legal actions |
| `math/odds.py` | Pot odds, MDF, effective stack, SPR |
| `math/ranges.py` | Range notation (`QQ+, ATs+, T9s-65s, KQo:0.5`), default home-game ranges, preflop range inference |
| `math/equity.py` | Equity vs weighted ranges: exact heads-up on turn/river, Monte Carlo otherwise (phevaluator) |
| `math/strength.py` | Hand description relative to the board, draws, outs, board texture |
| `analysis.py` | Bundles all of the above; narrows ranges after postflop bets/calls |
| `llm/` | Ollama client, versioned prompt, JSON schema (reasoning first, action enum limited to legal moves) |
| `validate.py`, `decide.py` | Validation, repair of trivial slips, retry, baseline fallback |
| `baseline.py` | Rule-based policy used as fallback and benchmark |
| `store.py` | SQLite decision log, reviews, scenario export |
| `web/` | FastAPI app and the single-page UI |

Opponent ranges are estimates. Preflop they come from position and action, and postflop they are down-weighted toward hands that would continue. Treat equity as "against a typical player who did this", and use the per-player range override when you know better.

## Settings

The main `config.toml` switches:
- `llm.model`
- `llm.think`: slower, lets thinking models reason first
- `llm.constrain_actions`
- `llm.prompt_version`
- `equity.iterations`

Everything runs locally, and nothing leaves your machine.
