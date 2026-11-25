import pandas as pd

from nba_ml.config import SNAPSHOT_COUNT
from nba_ml.features.ratings import pregame_team_ratings
from nba_ml.features.snapshots import regulation_targets, snapshot_events


def _games():
    return pd.DataFrame(
        [
            {
                "game_id": "0000000001",
                "game_date": "2021-12-01",
                "season": "2021-22",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_off_rating": 100.0,
                "away_off_rating": 110.0,
                "home_def_rating": 110.0,
                "away_def_rating": 100.0,
                "home_pace": 98.0,
                "away_pace": 98.0,
                "final_total": 200,
            },
            {
                "game_id": "0000000002",
                "game_date": "2022-10-20",
                "season": "2022-23",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_off_rating": 120.0,
                "away_off_rating": 90.0,
                "home_def_rating": 90.0,
                "away_def_rating": 120.0,
                "home_pace": 102.0,
                "away_pace": 102.0,
                "final_total": 210,
            },
            {
                "game_id": "0000000003",
                "game_date": "2022-10-22",
                "season": "2022-23",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_off_rating": 200.0,
                "away_off_rating": 80.0,
                "home_def_rating": 80.0,
                "away_def_rating": 200.0,
                "home_pace": 110.0,
                "away_pace": 110.0,
                "final_total": 250,
            },
        ]
    )


def test_pregame_ratings_ignore_the_current_and_future_games():
    ratings = pregame_team_ratings(_games(), rolling_games=20, shrinkage_k=10).set_index("game_id")
    # Opening 2022-23 game has only the 2021-22 result in its window, and that result is also the prior.
    assert ratings.loc["0000000002", "home_off_rating"] == 100.0
    # The 200-point third game must not leak into the second game's features.
    assert ratings.loc["0000000002", "home_off_rating"] != 200.0
    # Third game sees the first two results, shrunk toward the 2021-22 mean of 100.
    recent = (100.0 + 120.0) / 2.0
    expected = (2 * recent + 10 * 100.0) / 12.0
    assert ratings.loc["0000000003", "home_off_rating"] == expected


def test_regulation_grid_matches_the_configured_snapshot_count():
    targets = regulation_targets()
    assert len(targets) == SNAPSHOT_COUNT
    assert targets[0] == (1, 12 * 60)
    assert targets[-1] == (4, 60)


def test_snapshot_uses_only_events_at_or_before_the_minute_and_resets_fouls():
    events = pd.DataFrame(
        [
            {
                "action_number": 1,
                "period": 1,
                "seconds_left": 720,
                "team_id": 1,
                "is_team_foul": False,
                "score_home": 0,
                "score_away": 0,
            },
            {
                "action_number": 2,
                "period": 1,
                "seconds_left": 690,
                "team_id": 1,
                "is_team_foul": False,
                "score_home": 2,
                "score_away": 0,
            },
            {
                "action_number": 3,
                "period": 1,
                "seconds_left": 630,
                "team_id": 1,
                "is_team_foul": True,
                "score_home": 2,
                "score_away": 0,
            },
            {
                "action_number": 4,
                "period": 2,
                "seconds_left": 720,
                "team_id": 2,
                "is_team_foul": False,
                "score_home": 10,
                "score_away": 8,
            },
        ]
    )
    states = snapshot_events(events, home_team_id=1, away_team_id=2).set_index(
        ["period", "seconds_left_in_period"]
    )
    tip = states.loc[(1, 720)]
    assert tip["current_total"] == 0
    at_eleven = states.loc[(1, 660)]
    assert at_eleven["home_score"] == 2
    assert at_eleven["home_fouls_period"] == 0
    at_ten = states.loc[(1, 600)]
    assert at_ten["home_fouls_period"] == 1
    next_quarter = states.loc[(2, 720)]
    assert next_quarter["current_total"] == 18
    assert next_quarter["home_fouls_period"] == 0
    assert next_quarter["score_margin"] == 2
