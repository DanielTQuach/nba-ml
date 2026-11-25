from nba_ml.predict import state_to_frame


def test_state_frame_derives_margin_total_and_regulation_clock():
    state = {
        "period": 4,
        "seconds_left_in_period": 125,
        "home_score": 98,
        "away_score": 101,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_fouls_period": 4,
        "away_fouls_period": 3,
        "home_pace": 100,
        "home_off_rating": 110,
        "home_def_rating": 108,
        "away_pace": 99,
        "away_off_rating": 112,
        "away_def_rating": 107,
    }
    frame = state_to_frame(state, use_lineups=False)
    assert frame.loc[0, "current_total"] == 199
    assert frame.loc[0, "score_margin"] == -3
    assert frame.loc[0, "seconds_left_in_regulation"] == 125
