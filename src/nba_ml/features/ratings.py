"""Pregame pace and ratings from games strictly before the one being predicted."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from nba_ml.config import PROCESSED_DIR, ROLLING_GAMES, SHRINKAGE_K
from nba_ml.ingest.game_logs import load_games

RATING_COLUMNS = ("off_rating", "def_rating", "pace")


def previous_season(season: str) -> str:
    start = int(str(season).split("-")[0]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _team_results(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": pd.to_datetime(games["game_date"]),
            "season": games["season"],
            "team_id": games["home_team_id"].astype(int),
            "off_rating": games["home_off_rating"].astype(float),
            "def_rating": games["home_def_rating"].astype(float),
            "pace": games["home_pace"].astype(float),
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": pd.to_datetime(games["game_date"]),
            "season": games["season"],
            "team_id": games["away_team_id"].astype(int),
            "off_rating": games["away_off_rating"].astype(float),
            "def_rating": games["away_def_rating"].astype(float),
            "pace": games["away_pace"].astype(float),
        }
    )
    return pd.concat([home, away], ignore_index=True)


def pregame_team_ratings(
    games: pd.DataFrame,
    rolling_games: int = ROLLING_GAMES,
    shrinkage_k: float = SHRINKAGE_K,
) -> pd.DataFrame:
    """Return pregame pace and ratings for both teams in each game.

    The rolling window is the previous ``rolling_games`` games for that team,
    excluding the game itself. It is shrunk toward that team's previous-season mean.
    """
    history = _team_results(games).sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    grouped = history.groupby("team_id", sort=False)
    for column in RATING_COLUMNS:
        history[f"recent_{column}"] = grouped[column].transform(
            lambda series: series.shift(1).rolling(rolling_games, min_periods=1).mean()
        )
    history["recent_n"] = grouped["off_rating"].transform(
        lambda series: series.shift(1).rolling(rolling_games, min_periods=1).count()
    )

    season_means = history.groupby(["season", "team_id"], sort=False)[list(RATING_COLUMNS)].mean()
    season_means = season_means.rename(columns=lambda column: f"prior_{column}").reset_index()
    season_means["prior_season"] = season_means["season"]
    season_means = season_means.drop(columns=["season"])

    league_means = history.groupby("season", sort=False)[list(RATING_COLUMNS)].mean()
    league_means = league_means.rename(columns=lambda column: f"league_{column}").reset_index()
    league_means = league_means.rename(columns={"season": "prior_season"})

    history["prior_season"] = history["season"].map(previous_season)
    history = history.merge(season_means, on=["prior_season", "team_id"], how="left")
    history = history.merge(league_means, on="prior_season", how="left")
    recent_n = history["recent_n"].fillna(0.0)
    for column in RATING_COLUMNS:
        prior = history[f"prior_{column}"].fillna(history[f"league_{column}"])
        recent = history[f"recent_{column}"]
        shrunk = (recent_n * recent + shrinkage_k * prior) / (recent_n + shrinkage_k)
        history[f"pre_{column}"] = shrunk.where(recent_n > 0, prior)

    def _side(prefix: str, team_column: str) -> pd.DataFrame:
        side = history.rename(
            columns={
                "team_id": team_column,
                "pre_off_rating": f"{prefix}_off_rating",
                "pre_def_rating": f"{prefix}_def_rating",
                "pre_pace": f"{prefix}_pace",
            }
        )
        return side[["game_id", team_column, f"{prefix}_off_rating", f"{prefix}_def_rating", f"{prefix}_pace"]]

    home = _side("home", "home_team_id")
    away = _side("away", "away_team_id")
    base = games[["game_id", "home_team_id", "away_team_id"]].copy()
    base["home_team_id"] = base["home_team_id"].astype(int)
    base["away_team_id"] = base["away_team_id"].astype(int)
    rated = base.merge(home, on=["game_id", "home_team_id"], how="left").merge(
        away, on=["game_id", "away_team_id"], how="left"
    )
    return rated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write leakage-safe pregame team ratings.")
    parser.parse_args(argv)
    games = load_games()
    ratings = pregame_team_ratings(games)
    destination = PROCESSED_DIR / "pregame_ratings.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(destination, index=False)
    print(f"wrote {destination} rows={len(ratings)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise
