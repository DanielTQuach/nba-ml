"""Minute-level game states and the remaining-points label."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from nba_ml.config import PERIOD_SECONDS, PROCESSED_DIR, REGULATION_PERIODS, SEASONS, SNAPSHOT_STRIDE_SECONDS
from nba_ml.features.ratings import pregame_team_ratings
from nba_ml.ingest.game_logs import load_games
from nba_ml.ingest.pbp import load_pbp, pbp_path

SNAPSHOT_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "period",
    "seconds_left_in_period",
    "seconds_left_in_regulation",
    "home_score",
    "away_score",
    "current_total",
    "score_margin",
    "home_fouls_period",
    "away_fouls_period",
    "home_team_id",
    "away_team_id",
    "home_pace",
    "home_off_rating",
    "home_def_rating",
    "away_pace",
    "away_off_rating",
    "away_def_rating",
    "final_total",
    "remaining_points",
]


def regulation_targets() -> list[tuple[int, int]]:
    """(period, seconds left) at each regulation minute, including the opening tip."""
    targets = []
    regulation_seconds = REGULATION_PERIODS * PERIOD_SECONDS
    for elapsed in range(0, regulation_seconds, SNAPSHOT_STRIDE_SECONDS):
        period = elapsed // PERIOD_SECONDS + 1
        seconds_left = PERIOD_SECONDS - (elapsed % PERIOD_SECONDS)
        targets.append((period, seconds_left))
    return targets


def _event_is_after(period: int, seconds_left: float, target_period: int, target_seconds_left: int) -> bool:
    if period > target_period:
        return True
    if period < target_period:
        return False
    return seconds_left < target_seconds_left


def snapshot_events(events: pd.DataFrame, home_team_id: int, away_team_id: int) -> pd.DataFrame:
    """Score and team-foul state at each regulation minute, using only events already on the clock."""
    if events.empty:
        raise ValueError("play-by-play has no events")
    ordered = events.sort_values(["period", "seconds_left", "action_number"], ascending=[True, False, True])
    periods = ordered["period"].to_numpy(dtype=int)
    seconds_left = ordered["seconds_left"].to_numpy(dtype=float)
    scores_home = ordered["score_home"].to_numpy(dtype=float)
    scores_away = ordered["score_away"].to_numpy(dtype=float)
    team_ids = ordered["team_id"].to_numpy(dtype=float)
    is_foul = ordered["is_team_foul"].to_numpy(dtype=bool)

    state_period = np.zeros(len(ordered), dtype=int)
    state_home = np.zeros(len(ordered), dtype=float)
    state_away = np.zeros(len(ordered), dtype=float)
    state_home_fouls = np.zeros(len(ordered), dtype=int)
    state_away_fouls = np.zeros(len(ordered), dtype=int)
    home_fouls = 0
    away_fouls = 0
    open_period = None
    for index in range(len(ordered)):
        if open_period != periods[index]:
            home_fouls = 0
            away_fouls = 0
            open_period = periods[index]
        if is_foul[index]:
            team_id = team_ids[index]
            if team_id == home_team_id:
                home_fouls += 1
            elif team_id == away_team_id:
                away_fouls += 1
        state_period[index] = periods[index]
        state_home[index] = scores_home[index]
        state_away[index] = scores_away[index]
        state_home_fouls[index] = home_fouls
        state_away_fouls[index] = away_fouls

    rows = []
    cursor = -1
    last_index = len(ordered) - 1
    for target_period, target_seconds in regulation_targets():
        while cursor < last_index and not _event_is_after(
            periods[cursor + 1], seconds_left[cursor + 1], target_period, target_seconds
        ):
            cursor += 1
        if cursor < 0:
            home_score = 0.0
            away_score = 0.0
            home_foul_count = 0
            away_foul_count = 0
        else:
            home_score = state_home[cursor]
            away_score = state_away[cursor]
            home_foul_count = int(state_home_fouls[cursor])
            away_foul_count = int(state_away_fouls[cursor])
        rows.append(
            {
                "period": target_period,
                "seconds_left_in_period": target_seconds,
                "seconds_left_in_regulation": (REGULATION_PERIODS - target_period) * PERIOD_SECONDS + target_seconds,
                "home_score": home_score,
                "away_score": away_score,
                "current_total": home_score + away_score,
                "score_margin": home_score - away_score,
                "home_fouls_period": home_foul_count,
                "away_fouls_period": away_foul_count,
            }
        )
    return pd.DataFrame(rows)


def _attach_game(states: pd.DataFrame, game: pd.Series, ratings: pd.Series) -> pd.DataFrame:
    frame = states.copy()
    frame.insert(0, "game_id", game["game_id"])
    frame.insert(1, "game_date", pd.Timestamp(game["game_date"]))
    frame.insert(2, "season", game["season"])
    frame["home_team_id"] = int(game["home_team_id"])
    frame["away_team_id"] = int(game["away_team_id"])
    for column in (
        "home_pace",
        "home_off_rating",
        "home_def_rating",
        "away_pace",
        "away_off_rating",
        "away_def_rating",
    ):
        frame[column] = float(ratings[column])
    final_total = float(game["final_total"])
    frame["final_total"] = final_total
    frame["remaining_points"] = (final_total - frame["current_total"]).round().astype(int)
    return frame[SNAPSHOT_COLUMNS]


def build_snapshots(games: pd.DataFrame | None = None) -> pd.DataFrame:
    games = load_games() if games is None else games
    training_games = games[games["season"].isin(SEASONS)].copy()
    ratings = pregame_team_ratings(games).set_index("game_id")
    by_id = training_games.set_index("game_id", drop=False)
    frames = []
    missing = 0
    for game_id, game_row in by_id.iterrows():
        if not pbp_path(game_id).exists():
            missing += 1
            continue
        events = load_pbp(str(game_id))
        states = snapshot_events(events, int(game_row["home_team_id"]), int(game_row["away_team_id"]))
        frames.append(_attach_game(states, game_row, ratings.loc[game_id]))
    if not frames:
        raise FileNotFoundError("no cached play-by-play files for training seasons")
    snapshots = pd.concat(frames, ignore_index=True)
    snapshots = snapshots[snapshots["remaining_points"] >= 0].reset_index(drop=True)
    print(
        f"snapshots rows={len(snapshots)} games={snapshots['game_id'].nunique()} missing_pbp={missing}",
        flush=True,
    )
    return snapshots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build minute snapshots with remaining-points labels.")
    parser.parse_args(argv)
    snapshots = build_snapshots()
    destination = PROCESSED_DIR / "snapshots.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(destination, index=False)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise
