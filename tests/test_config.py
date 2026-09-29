from poker_agent.config import Settings, load_settings


def test_defaults_when_missing(tmp_path):
    assert load_settings(tmp_path / "nope.toml") == Settings()


def test_loads_repo_config():
    s = load_settings("config.toml")
    assert s.game.big_blind == 0.20
    assert s.game.max_players == 7
