"""SQLite log of every recommendation, for review and for growing the eval set."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import yaml

from poker_agent.decide import Recommendation
from poker_agent.state import GameState

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    hand_id TEXT,
    street TEXT NOT NULL,
    hero_cards TEXT NOT NULL,
    board TEXT NOT NULL,
    action TEXT NOT NULL,
    amount REAL,
    source TEXT NOT NULL,
    model TEXT,
    latency_s REAL,
    state_json TEXT NOT NULL,
    recommendation_json TEXT NOT NULL,
    followed INTEGER,          -- did hero take the advice? (NULL = not recorded)
    note TEXT                  -- free-text review note, e.g. the outcome
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, state: GameState, rec: Recommendation, hand_id: str | None = None) -> int:
        d = rec.decision
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO decisions (created_at, hand_id, street, hero_cards, board, action,"
                " amount, source, model, latency_s, state_json, recommendation_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"), hand_id, state.street.value,
                    "".join(state.hole_cards), "".join(state.board), d.action.value, d.amount,
                    rec.source, rec.model, round(rec.total_latency_s, 2),
                    state.model_dump_json(), rec.model_dump_json(),
                ),
            )
            return cur.lastrowid

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT id, created_at, hand_id, street, hero_cards, board, action, amount, source,"
                " model, latency_s, followed, note FROM decisions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def get(self, decision_id: int) -> tuple[GameState, Recommendation]:
        with self._conn() as c:
            row = c.execute("SELECT state_json, recommendation_json FROM decisions WHERE id = ?",
                            (decision_id,)).fetchone()
        if row is None:
            raise KeyError(f"no decision with id {decision_id}")
        return (GameState.model_validate_json(row["state_json"]),
                Recommendation.model_validate_json(row["recommendation_json"]))

    def review(self, decision_id: int, followed: bool | None = None, note: str | None = None) -> None:
        with self._conn() as c:
            n = c.execute("UPDATE decisions SET followed = COALESCE(?, followed),"
                          " note = COALESCE(?, note) WHERE id = ?",
                          (followed, note, decision_id)).rowcount
        if n == 0:
            raise KeyError(f"no decision with id {decision_id}")


def export_scenario(store: Store, decision_id: int, acceptable: list[str], out: Path,
                    best: str | None = None, scenario_id: str | None = None) -> dict:
    """Append a logged spot to a scenario YAML file so it joins the eval set."""
    state, _ = store.get(decision_id)
    sc = {
        "id": scenario_id or f"logged-{decision_id}",
        "description": f"Logged decision #{decision_id}",
        "players": {p.seat: p.stack for p in state.players},
        "button": state.button_seat,
        "hero": state.hero_seat,
        "cards": "".join(state.hole_cards),
        "board": "".join(state.board),
        "actions": [
            " ".join([a.street.value, str(a.seat), a.type.value]
                     + ([f"{a.amount:.2f}"] if a.amount is not None else []))
            for a in state.actions
        ],
        "acceptable": acceptable,
        "tags": ["logged", state.street.value],
    }
    if best:
        sc["best"] = best
    if state.opponent_notes:
        sc["opponent_notes"] = dict(state.opponent_notes)
    existing = yaml.safe_load(out.read_text()) if out.exists() else []
    existing = existing or []
    if any(e.get("id") == sc["id"] for e in existing):
        raise ValueError(f"scenario {sc['id']} already exists in {out}")
    existing.append(sc)
    out.write_text(yaml.safe_dump(existing, sort_keys=False, width=100))
    return sc

