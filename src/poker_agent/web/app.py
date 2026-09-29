"""FastAPI app: a JSON API plus a single static page for live hand entry.

Run with ``poker-agent serve --host 0.0.0.0`` and open it from a phone on
the same Wi-Fi network.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from poker_agent.config import Settings, load_settings
from poker_agent.decide import decide, make_client
from poker_agent.llm.prompt import action_history
from poker_agent.rules import IllegalAction, NotHerosTurn, legal_actions, replay, to_dollars
from poker_agent.state import GameState, Street
from poker_agent.store import Store

STATIC = Path(__file__).parent / "static"


class DecideRequest(BaseModel):
    state: dict
    use_llm: bool = True
    hand_id: str | None = None


class ReviewRequest(BaseModel):
    followed: bool | None = None
    note: str | None = None


def _parse_state(raw: dict) -> GameState:
    try:
        return GameState.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise HTTPException(422, f"{loc}: {first['msg']}" if loc else first["msg"]) from None


def table_view(state: GameState) -> dict:
    """Everything the page needs to render the table and the next action buttons."""
    table = replay(state)
    positions = state.positions
    seats = [
        {
            "seat": s,
            "position": positions[s],
            "stack": to_dollars(table.seats[s].stack),
            "committed_street": to_dollars(table.seats[s].committed_street),
            "folded": table.seats[s].folded,
            "all_in": table.seats[s].all_in,
            "hero": s == state.hero_seat,
        }
        for s in table.order
    ]
    to_act = table.to_act
    options = legal_actions(state, table, seat=to_act).model_dump() if to_act else None
    needs_board = None
    if table.round_closed and not table.hand_over and state.street is not Street.RIVER:
        needs_board = {Street.PREFLOP: 3, Street.FLOP: 1, Street.TURN: 1}[state.street]
    return {
        "street": state.street.value,
        "pot": to_dollars(table.pot),
        "seats": seats,
        "to_act": to_act,
        "options": options,
        "hero_to_act": to_act == state.hero_seat,
        "round_closed": table.round_closed,
        "hand_over": table.hand_over,
        "needs_board": needs_board,
        "history": action_history(state),
    }


def create_app(settings: Settings | None = None, client=None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Poker Agent")
    store = Store(settings.storage.db_path)
    llm = client or make_client(settings)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        g = settings.game
        return {"small_blind": g.small_blind, "big_blind": g.big_blind, "buy_in": g.buy_in,
                "max_players": g.max_players, "chip_increment": g.chip_increment,
                "model": settings.llm.model}

    @app.get("/api/health")
    def health():
        ok, msg = llm.health() if hasattr(llm, "health") else (True, "custom client")
        return {"ok": ok, "message": msg, "model": settings.llm.model}

    @app.post("/api/table")
    def table(raw: dict):
        state = _parse_state(raw)
        try:
            return table_view(state)
        except (IllegalAction, NotHerosTurn) as e:
            raise HTTPException(422, str(e)) from None

    @app.post("/api/decide")
    def decide_endpoint(req: DecideRequest):
        state = _parse_state(req.state)
        try:
            rec = decide(state, settings, client=llm, use_llm=req.use_llm)
        except (IllegalAction, NotHerosTurn) as e:
            raise HTTPException(422, str(e)) from None
        rec_id = store.log(state, rec, hand_id=req.hand_id)
        return {"id": rec_id, "recommendation": rec.model_dump(mode="json")}

    @app.post("/api/review/{decision_id}")
    def review(decision_id: int, req: ReviewRequest):
        try:
            store.review(decision_id, followed=req.followed, note=req.note)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        return {"ok": True}

    @app.get("/api/log")
    def log(limit: int = 20):
        return [dict(r) for r in store.recent(limit)]

    return app
