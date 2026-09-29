"""Load settings from config.toml, falling back to built-in defaults."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel


class GameConfig(BaseModel):
    small_blind: float = 0.10
    big_blind: float = 0.20
    buy_in: float | None = None  # only used as the default stack in the UI
    max_players: int = 7
    chip_increment: float | None = None  # defaults to the small blind

    @property
    def default_stack(self) -> float:
        return self.buy_in or 100 * self.big_blind

    @property
    def chip(self) -> float:
        return self.chip_increment or self.small_blind


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:30b-a3b"
    temperature: float = 0.2
    timeout_seconds: float = 60
    max_retries: int = 1
    think: bool = False
    max_tokens: int = 600
    constrain_actions: bool = True
    prompt_version: str = "v2"


class EquityConfig(BaseModel):
    iterations: int = 20000
    seed: int = 0


class StorageConfig(BaseModel):
    db_path: str = "poker_agent.db"


class Settings(BaseModel):
    game: GameConfig = GameConfig()
    llm: LLMConfig = LLMConfig()
    equity: EquityConfig = EquityConfig()
    storage: StorageConfig = StorageConfig()


def load_settings(path: str | Path | None = None) -> Settings:
    """Read settings from `path`, $POKER_AGENT_CONFIG, or ./config.toml."""
    path = Path(path or os.environ.get("POKER_AGENT_CONFIG", "config.toml"))
    if not path.exists():
        return Settings()
    with path.open("rb") as f:
        return Settings.model_validate(tomllib.load(f))
