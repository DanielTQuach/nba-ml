"""Print P(final combined score = X) for one game state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ml.config import MODEL_DIR, PERIOD_SECONDS, REGULATION_PERIODS, REMAINING_PMF_MAX
from nba_ml.eval.report import load_model, network_pmf
from nba_ml.models.remaining import shift_to_final_total

REQUIRED = (
    "period",
    "seconds_left_in_period",
    "home_score",
    "away_score",
    "home_team_id",
    "away_team_id",
    "home_fouls_period",
    "away_fouls_period",
    "home_pace",
    "home_off_rating",
    "home_def_rating",
    "away_pace",
    "away_off_rating",
    "away_def_rating",
)


def _regulation_seconds(period: int, seconds_left: int) -> int:
    if period > REGULATION_PERIODS:
        return 0
    return (REGULATION_PERIODS - period) * PERIOD_SECONDS + seconds_left


def state_to_frame(state: dict, use_lineups: bool) -> pd.DataFrame:
    missing = [key for key in REQUIRED if key not in state]
    if missing:
        raise ValueError(f"game state is missing {missing}")
    period = int(state["period"])
    seconds_left = int(state["seconds_left_in_period"])
    home_score = float(state["home_score"])
    away_score = float(state["away_score"])
    row = {
        "period": period,
        "seconds_left_in_period": seconds_left,
        "seconds_left_in_regulation": _regulation_seconds(period, seconds_left),
        "current_total": home_score + away_score,
        "score_margin": home_score - away_score,
        "home_fouls_period": int(state["home_fouls_period"]),
        "away_fouls_period": int(state["away_fouls_period"]),
        "home_team_id": int(state["home_team_id"]),
        "away_team_id": int(state["away_team_id"]),
        "home_pace": float(state["home_pace"]),
        "home_off_rating": float(state["home_off_rating"]),
        "home_def_rating": float(state["home_def_rating"]),
        "away_pace": float(state["away_pace"]),
        "away_off_rating": float(state["away_off_rating"]),
        "away_def_rating": float(state["away_def_rating"]),
    }
    if use_lineups:
        home_ids = [int(player_id) for player_id in state["home_on_court"]]
        away_ids = [int(player_id) for player_id in state["away_on_court"]]
        if len(home_ids) != 5 or len(away_ids) != 5:
            raise ValueError("home_on_court and away_on_court must each list 5 player ids")
        minutes = {int(key): float(value) for key, value in state.get("player_minutes", {}).items()}
        fouls = {int(key): int(value) for key, value in state.get("player_fouls", {}).items()}
        for slot, player_id in enumerate(home_ids, start=1):
            row[f"home_p{slot}"] = player_id
            row[f"home_m{slot}"] = minutes.get(player_id, 0.0)
            row[f"home_f{slot}"] = fouls.get(player_id, 0)
        for slot, player_id in enumerate(away_ids, start=1):
            row[f"away_p{slot}"] = player_id
            row[f"away_m{slot}"] = minutes.get(player_id, 0.0)
            row[f"away_f{slot}"] = fouls.get(player_id, 0)
    return pd.DataFrame([row])


def default_checkpoint() -> Path:
    phase2 = MODEL_DIR / "phase2.pt"
    if phase2.exists():
        return phase2
    return MODEL_DIR / "phase1.pt"


def describe(state: dict, score: int, checkpoint: Path) -> dict[str, float]:
    model, payload = load_model(checkpoint)
    frame = state_to_frame(state, use_lineups=bool(payload["use_lineups"]))
    remaining = network_pmf(model, payload, frame)[0]
    current_total = int(round(float(frame.loc[0, "current_total"])))
    final = shift_to_final_total(remaining, current_total)
    if score < 0 or score >= len(final):
        probability = 0.0
    else:
        probability = float(final[score])
    grid = np.arange(len(final))
    cdf = np.cumsum(final)
    def quantile(level: float) -> int:
        return int(grid[np.searchsorted(cdf, level, side="left")])

    return {
        "probability": probability,
        "mean_final": float(np.dot(grid, final)),
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "support_max": current_total + REMAINING_PMF_MAX,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probability that the combined final score equals X.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True, dest="score")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or default_checkpoint()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    summary = describe(state, args.score, checkpoint)
    print(f"P(final total = {args.score}) = {summary['probability']:.6f}")
    print(f"mean final total = {summary['mean_final']:.1f}")
    print(f"p10 = {summary['p10']}  p50 = {summary['p50']}  p90 = {summary['p90']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise
