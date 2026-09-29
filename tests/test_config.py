from poker_agent.config import Settings, load_settings


def test_defaults_when_missing(tmp_path):
    assert load_settings(tmp_path / "nope.toml") == Settings()


def test_loads_repo_config():
    s = load_settings("config.toml")
    assert (s.game.small_blind, s.game.big_blind) == (1.00, 2.00)
    assert s.game.default_stack == 1000.00
    assert s.game.chip == 1.00  # defaults to the small blind
    assert s.game.max_players == 7
