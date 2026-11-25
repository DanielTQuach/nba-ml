import pandas as pd

from nba_ml.features.lineups import lineup_snapshots, rotation_snapshots, starter_ids


def _event(number, period, seconds_left, action, team_id=1, person_id=None, person_out=None, person_in=None, sub_type=""):
    return {
        "action_number": number,
        "period": period,
        "seconds_left": seconds_left,
        "action_type": action,
        "sub_type": sub_type,
        "team_id": team_id,
        "person_id": person_id,
        "person_out": person_out,
        "person_in": person_in,
    }


def test_starters_are_the_players_with_a_position():
    box = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "team_id": [10, 10, 10, 10, 10, 10],
            "position": ["G", "G", "F", "F", "C", ""],
        }
    )
    assert starter_ids(box, 10) == [1, 2, 3, 4, 5]


def test_substitution_changes_the_court_and_keeps_minutes_on_the_departing_player():
    events = pd.DataFrame(
        [
            _event(1, 1, 720, "jumpball"),
            _event(2, 1, 660, "substitution", team_id=1, person_out=1, person_in=11),
            _event(3, 1, 600, "foul", team_id=1, person_id=11, sub_type="personal"),
        ]
    )
    snapshots = lineup_snapshots(events, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], home_team_id=1, away_team_id=2)
    assert snapshots is not None
    tip = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 720)].iloc[0]
    assert tip["home_m1"] == 0
    at_eleven = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 660)].iloc[0]
    home_ids = {at_eleven[f"home_p{slot}"] for slot in range(1, 6)}
    assert 11 in home_ids
    assert 1 not in home_ids
    # The departing player is no longer a column; the minute he played is not on the new player.
    new_slot = next(slot for slot in range(1, 6) if at_eleven[f"home_p{slot}"] == 11)
    assert at_eleven[f"home_m{new_slot}"] == 0
    at_ten = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 600)].iloc[0]
    foul_slot = next(slot for slot in range(1, 6) if at_ten[f"home_p{slot}"] == 11)
    assert at_ten[f"home_f{foul_slot}"] == 1


def test_rotation_stints_swap_players_and_keep_minutes_on_the_departing_player():
    stints = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1, 1, 1] + [2] * 5,
            "person_id": [1, 2, 3, 4, 5, 11, 6, 7, 8, 9, 10],
            "in_tenths": [0, 0, 0, 0, 0, 600] + [0] * 5,
            "out_tenths": [28800, 28800, 28800, 28800, 600, 28800] + [28800] * 5,
        }
    )
    events = pd.DataFrame(
        [
            {
                "action_number": 1,
                "period": 1,
                "seconds_left": 600,
                "action_type": "foul",
                "sub_type": "personal",
                "person_id": 11,
            }
        ]
    )
    snapshots = rotation_snapshots(stints, events, home_team_id=1, away_team_id=2)
    assert snapshots is not None
    tip = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 720)].iloc[0]
    assert {tip[f"home_p{slot}"] for slot in range(1, 6)} == {1, 2, 3, 4, 5}
    assert tip["home_m1"] == 0
    at_eleven = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 660)].iloc[0]
    home_ids = {at_eleven[f"home_p{slot}"] for slot in range(1, 6)}
    assert 11 in home_ids and 5 not in home_ids
    new_slot = next(slot for slot in range(1, 6) if at_eleven[f"home_p{slot}"] == 11)
    assert at_eleven[f"home_m{new_slot}"] == 0
    stayed = next(slot for slot in range(1, 6) if at_eleven[f"home_p{slot}"] == 1)
    assert at_eleven[f"home_m{stayed}"] == 1
    at_ten = snapshots[(snapshots["period"] == 1) & (snapshots["seconds_left_in_period"] == 600)].iloc[0]
    foul_slot = next(slot for slot in range(1, 6) if at_ten[f"home_p{slot}"] == 11)
    assert at_ten[f"home_f{foul_slot}"] == 1


def test_lineup_is_dropped_when_a_substitution_does_not_leave_five_players():
    events = pd.DataFrame([_event(1, 1, 700, "substitution", team_id=1, person_out=99, person_in=11)])
    assert lineup_snapshots(events, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], 1, 2) is None
