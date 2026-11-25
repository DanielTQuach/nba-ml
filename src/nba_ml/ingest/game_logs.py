"""Season team logs and one-row-per-game tables."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ml.config import (
    GAME_LOG_DIR,
    GAME_LOG_SEASONS,
    MAX_RETRIES,
    PROCESSED_DIR,
    REQUEST_SLEEP_SECONDS,
    REQUEST_TIMEOUT,
)

COUNTING_STATS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf")


def normalize_game_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def parse_minutes(value: object) -> float:
    """Team logs store either player-minute totals ('240' or '240:00') or game minutes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(text)


def possessions(fga: pd.Series, oreb: pd.Series, tov: pd.Series, fta: pd.Series) -> pd.Series:
    return fga.astype(float) - oreb.astype(float) + tov.astype(float) + 0.44 * fta.astype(float)


def build_games(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Pair home and away team-game rows into one row per game.

    Possession estimates and per-game ratings are stored so pregame features can
    roll them forward. Callers must not use a game's own ratings as its features.
    """
    logs = team_logs.copy()
    logs.columns = [str(column).lower() for column in logs.columns]
    required = {"game_id", "game_date", "matchup", "team_id", "team_abbreviation", "min", "season", *COUNTING_STATS}
    missing = required.difference(logs.columns)
    if missing:
        raise ValueError(f"team logs are missing columns: {sorted(missing)}")

    logs["game_id"] = logs["game_id"].map(normalize_game_id)
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs["team_id"] = logs["team_id"].astype(int)
    logs["is_home"] = logs["matchup"].str.contains("vs.", regex=False)
    logs["player_minutes"] = logs["min"].map(parse_minutes)
    for column in COUNTING_STATS:
        logs[column] = pd.to_numeric(logs[column], errors="coerce")
    logs["possessions"] = possessions(logs["fga"], logs["oreb"], logs["tov"], logs["fta"])

    paired_rows: list[dict[str, object]] = []
    for game_id, group in logs.groupby("game_id", sort=False):
        group = group.drop_duplicates("team_id")
        if len(group) != 2:
            continue
        home_rows = group[group["is_home"]]
        if len(home_rows) == 1:
            home = home_rows.iloc[0]
            away = group[group["team_id"] != home["team_id"]].iloc[0]
        else:
            # Neutral-site games are sometimes logged as away for both teams.
            ordered = group.sort_values("team_id")
            home = ordered.iloc[0]
            away = ordered.iloc[1]
        paired_rows.append({"game_id": game_id, "home": home, "away": away})

    if not paired_rows:
        raise ValueError("no games could be paired from team logs")

    home = pd.DataFrame([row["home"] for row in paired_rows]).reset_index(drop=True)
    away = pd.DataFrame([row["away"] for row in paired_rows]).reset_index(drop=True)
    game_ids = pd.Series([row["game_id"] for row in paired_rows])

    game_minutes = home["player_minutes"] / 5.0
    # Some feeds already store a single game length near 48 instead of a five-man total.
    already_game_length = home["player_minutes"] < 80
    game_minutes = game_minutes.where(~already_game_length, home["player_minutes"])
    average_possessions = (home["possessions"] + away["possessions"]) / 2.0
    pace = 48.0 * average_possessions / game_minutes.replace(0, np.nan)

    games = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": home["season"].to_numpy(),
            "game_date": home["game_date"].to_numpy(),
            "home_team_id": home["team_id"].astype(int).to_numpy(),
            "away_team_id": away["team_id"].astype(int).to_numpy(),
            "home_team_abbr": home["team_abbreviation"].to_numpy(),
            "away_team_abbr": away["team_abbreviation"].to_numpy(),
            "home_pts": home["pts"].to_numpy(),
            "away_pts": away["pts"].to_numpy(),
            "home_possessions": home["possessions"].to_numpy(),
            "away_possessions": away["possessions"].to_numpy(),
            "home_off_rating": (100.0 * home["pts"] / home["possessions"].replace(0, np.nan)).to_numpy(),
            "away_off_rating": (100.0 * away["pts"] / away["possessions"].replace(0, np.nan)).to_numpy(),
            "home_def_rating": (100.0 * away["pts"] / away["possessions"].replace(0, np.nan)).to_numpy(),
            "away_def_rating": (100.0 * home["pts"] / home["possessions"].replace(0, np.nan)).to_numpy(),
            "home_pace": pace.to_numpy(),
            "away_pace": pace.to_numpy(),
            "final_total": (home["pts"] + away["pts"]).to_numpy(),
        }
    )
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _fetch_team_logs(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            endpoint = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",
                timeout=REQUEST_TIMEOUT,
            )
            frame = endpoint.league_game_log.get_data_frame()
            if frame.empty:
                raise RuntimeError(f"no team logs returned for {season}")
            frame["season"] = season
            return frame
        except Exception as exc:  # noqa: BLE001 - network and API errors retry together
            last_error = exc
            time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise RuntimeError(f"team logs for {season} failed after {MAX_RETRIES} tries") from last_error


def load_team_logs(directory: Path | None = None) -> pd.DataFrame:
    directory = directory or GAME_LOG_DIR
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no team logs in {directory}")
    return pd.concat(frames, ignore_index=True)


def load_games(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "games.parquet")
    games = pd.read_parquet(path)
    games["game_id"] = games["game_id"].map(normalize_game_id)
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def ingest_game_logs(seasons: list[str] | None = None) -> pd.DataFrame:
    seasons = seasons or list(GAME_LOG_SEASONS)
    GAME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        destination = GAME_LOG_DIR / f"{season}.parquet"
        if destination.exists() and destination.stat().st_size > 0:
            print(f"cached team logs {season}", flush=True)
            continue
        print(f"downloading team logs {season}", flush=True)
        frame = _fetch_team_logs(season)
        frame.to_parquet(destination, index=False)
        print(f"wrote {destination} rows={len(frame)}", flush=True)
    games = build_games(load_team_logs())
    games_path = PROCESSED_DIR / "games.parquet"
    games.to_parquet(games_path, index=False)
    counts = games.groupby("season").size()
    print(counts.to_string(), flush=True)
    print(f"wrote {games_path} games={len(games)}", flush=True)
    return games


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download regular-season team game logs.")
    parser.add_argument("--season", action="append", dest="seasons", help="Season to download, like 2022-23. Repeatable.")
    args = parser.parse_args(argv)
    ingest_game_logs(args.seasons)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise
