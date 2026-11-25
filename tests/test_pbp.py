import pandas as pd
import pytest

from nba_ml.ingest.pbp import counts_as_team_foul, normalize_pbp, parse_clock, parse_substitution


def test_parse_clock_accepts_mmss_and_iso():
    assert parse_clock("5:32") == 332
    assert parse_clock("PT05M32.00S") == 332
    assert parse_clock("PT12M00.00S") == 720
    assert parse_clock("0:00") == 0


def test_team_foul_skips_offensive_and_technical():
    assert counts_as_team_foul("foul", "personal")
    assert counts_as_team_foul("foul", "shooting")
    assert not counts_as_team_foul("foul", "offensive")
    assert not counts_as_team_foul("foul", "technical")
    assert not counts_as_team_foul("rebound", "personal")


def test_substitution_description_names_the_incoming_player_first():
    assert parse_substitution("SUB: Payton Pritchard FOR Derrick White") == (
        "Payton Pritchard",
        "Derrick White",
    )


def test_normalize_v3_treats_the_row_player_as_the_one_leaving():
    raw = pd.DataFrame(
        {
            "actionNumber": [1, 2, 3, 4],
            "clock": ["PT12M00.00S", "PT11M30.00S", "PT11M00.00S", "PT00M00.00S"],
            "period": [1, 1, 1, 4],
            "teamId": [10, 10, 10, 10],
            "personId": [1, 3, 1, 3],
            "playerName": ["A", "C", "A", "C"],
            "description": ["Jump ball", "C makes layup", "SUB: C FOR A", "end"],
            "actionType": ["jumpball", "2pt", "substitution", "period"],
            "subType": ["", "layup", "", ""],
            "scoreHome": ["", "2", "2", "110"],
            "scoreAway": ["", "0", "0", "100"],
        }
    )
    normalized = normalize_pbp(raw, home_pts=110, away_pts=100)
    assert list(normalized["score_home"]) == [0, 2, 2, 110]
    assert list(normalized["score_away"]) == [0, 0, 0, 100]
    substitution = normalized[normalized["action_type"] == "substitution"].iloc[0]
    assert substitution["player_out_name"] == "A"
    assert substitution["person_out"] == 1
    assert substitution["person_in"] == 3


def test_normalize_flips_score_when_the_feed_is_away_first():
    raw = pd.DataFrame(
        {
            "actionNumber": [1],
            "clock": ["0:00"],
            "period": [4],
            "teamId": [1],
            "personId": [1],
            "playerName": ["A"],
            "description": ["end"],
            "actionType": ["period"],
            "subType": [""],
            "scoreHome": ["108"],
            "scoreAway": ["99"],
        }
    )
    normalized = normalize_pbp(raw, home_pts=99, away_pts=108)
    assert normalized.loc[0, "score_home"] == 99
    assert normalized.loc[0, "score_away"] == 108


def test_shared_last_names_resolve_from_the_description_initial():
    raw = pd.DataFrame(
        {
            "personId": [10, 20, 10, 20, 30],
            "playerName": ["Green", "Green", "Green", "Green", "Thybulle"],
            "description": [
                "D. Green REBOUND (Off:0 Def:1)",
                "J. Green 3PT Jump Shot (3 PTS)",
                "SUB: J. Green FOR D. Green",
                "SUB: Thybulle FOR J. Green",
                "SUB: Green FOR Thybulle",
            ],
            "actionType": ["rebound", "made shot", "substitution", "substitution", "substitution"],
            "subType": ["", "", "", "", ""],
            "scoreHome": ["0", "3", "3", "3", "3"],
            "scoreAway": ["0", "0", "0", "0", "0"],
            "clock": ["12:00", "11:00", "10:00", "9:00", "8:00"],
            "actionNumber": [1, 2, 3, 4, 5],
            "period": [1, 1, 1, 1, 1],
            "teamId": [5, 5, 5, 5, 5],
        }
    )
    normalized = normalize_pbp(raw, home_pts=3, away_pts=0)
    first_sub = normalized[normalized["description"].str.contains("FOR D. Green")].iloc[0]
    assert first_sub["person_out"] == 10
    assert first_sub["person_in"] == 20
    thybulle_in = normalized[normalized["description"].str.contains("FOR J. Green")].iloc[0]
    assert thybulle_in["person_in"] == 30


def test_normalize_rejects_a_score_that_matches_neither_order():
    raw = pd.DataFrame(
        {
            "actionNumber": [1],
            "clock": ["0:00"],
            "period": [4],
            "teamId": [1],
            "personId": [1],
            "playerName": ["A"],
            "description": ["end"],
            "actionType": ["period"],
            "subType": [""],
            "scoreHome": ["10"],
            "scoreAway": ["10"],
        }
    )
    with pytest.raises(ValueError):
        normalize_pbp(raw, home_pts=80, away_pts=90)
