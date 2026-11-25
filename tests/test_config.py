from nba_ml.config import GAME_LOG_SEASONS, SEASONS, SNAPSHOT_COUNT


def test_training_seasons_exclude_the_prior_and_the_current_season():
    assert SEASONS == ["2022-23", "2023-24", "2024-25", "2025-26"]
    assert GAME_LOG_SEASONS[0] == "2021-22"
    assert "2026-27" not in GAME_LOG_SEASONS


def test_regulation_snapshot_grid_has_one_row_per_minute():
    assert SNAPSHOT_COUNT == 48
