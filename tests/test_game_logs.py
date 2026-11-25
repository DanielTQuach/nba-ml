import pandas as pd

from nba_ml.ingest.game_logs import build_games, normalize_game_id, parse_minutes, possessions


def _team_row(game_id, date, matchup, team_id, abbr, pts, fga, fta, oreb, tov, minutes, season="2022-23"):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "MATCHUP": matchup,
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": abbr,
        "MIN": minutes,
        "PTS": pts,
        "FGM": 40,
        "FGA": fga,
        "FG3M": 10,
        "FG3A": 30,
        "FTM": 10,
        "FTA": fta,
        "OREB": oreb,
        "DREB": 30,
        "REB": 30 + oreb,
        "AST": 20,
        "STL": 7,
        "BLK": 4,
        "TOV": tov,
        "PF": 18,
        "season": season,
    }


def test_game_id_keeps_leading_zeros():
    assert normalize_game_id(22200001) == "0022200001"
    assert normalize_game_id("0022200001") == "0022200001"
    assert normalize_game_id("22200001.0") == "0022200001"


def test_build_games_pairs_home_and_away_and_estimates_pace():
    rows = [
        _team_row("0022200001", "2022-10-18", "BOS vs. NYK", 1610612738, "BOS", 110, 90, 20, 10, 12, 240),
        _team_row("0022200001", "2022-10-18", "NYK @ BOS", 1610612752, "NYK", 100, 88, 22, 8, 14, "240:00"),
        _team_row("0022200002", "2022-10-19", "LAL vs. GSW", 1610612747, "LAL", 120, 92, 18, 11, 13, 240),
        _team_row("0022200002", "2022-10-19", "GSW @ LAL", 1610612744, "GSW", 115, 91, 25, 9, 11, 240),
    ]
    games = build_games(pd.DataFrame(rows))
    assert list(games["game_id"]) == ["0022200001", "0022200002"]
    first = games.iloc[0]
    assert first["home_team_abbr"] == "BOS"
    assert first["away_team_abbr"] == "NYK"
    assert first["final_total"] == 210
    expected_home_poss = possessions(pd.Series([90]), pd.Series([10]), pd.Series([12]), pd.Series([20])).iloc[0]
    assert first["home_possessions"] == expected_home_poss
    assert first["home_off_rating"] == 100 * 110 / expected_home_poss
    assert first["home_pace"] > 90


def test_neutral_site_games_with_no_home_flag_still_pair():
    rows = [
        _team_row("0022400147", "2024-11-01", "WAS @ MIA", 1610612764, "WAS", 100, 80, 20, 8, 12, 240),
        _team_row("0022400147", "2024-11-01", "MIA @ WAS", 1610612748, "MIA", 90, 78, 18, 7, 11, 240),
    ]
    games = build_games(pd.DataFrame(rows))
    assert len(games) == 1
    assert games.loc[0, "final_total"] == 190
    assert {games.loc[0, "home_team_abbr"], games.loc[0, "away_team_abbr"]} == {"WAS", "MIA"}


def test_parse_minutes_accepts_clock_and_plain_totals():
    assert parse_minutes("240:00") == 240
    assert parse_minutes(48) == 48
    assert parse_minutes("32:30") == 32.5
