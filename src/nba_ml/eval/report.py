"""Held-out NLL, CRPS, and a reliability diagram."""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nba_ml.config import EVAL_DIR, MODEL_DIR, RELIABILITY_THRESHOLD
from nba_ml.eval.metrics import crps_from_pmf, nll_from_pmf, probability_final_over, reliability_curve
from nba_ml.models.baseline import EmpiricalRemaining
from nba_ml.models.remaining import RemainingPointsNet, remaining_pmf
from nba_ml.models.train import assign_splits, lineup_matrix, load_training_frame

CHUNK = 2048


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RemainingPointsNet(
        n_teams=int(payload["n_teams"]),
        numeric_dim=len(payload["numeric_features"]),
        n_players=int(payload["n_players"]),
        use_lineups=bool(payload["use_lineups"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def network_pmf(model: RemainingPointsNet, payload: dict, frame: pd.DataFrame) -> np.ndarray:
    mean = np.array(payload["numeric_mean"], dtype=float)
    std = np.array(payload["numeric_std"], dtype=float)
    features = payload["numeric_features"]
    numeric = torch.tensor((frame[features].to_numpy(dtype=float) - mean) / std, dtype=torch.float32)
    home = torch.tensor(
        [payload["team_to_idx"].get(int(value), 0) for value in frame["home_team_id"]],
        dtype=torch.long,
    )
    away = torch.tensor(
        [payload["team_to_idx"].get(int(value), 0) for value in frame["away_team_id"]],
        dtype=torch.long,
    )
    players = minutes = fouls = None
    if payload["use_lineups"]:
        encoded, minutes_np, fouls_np = lineup_matrix(frame, payload["player_to_idx"])
        players = torch.tensor(encoded, dtype=torch.long)
        minutes = torch.tensor(minutes_np, dtype=torch.float32)
        fouls = torch.tensor(fouls_np, dtype=torch.float32)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(frame), CHUNK):
            stop = start + CHUNK
            mu, alpha = model(
                numeric[start:stop],
                home[start:stop],
                away[start:stop],
                None if players is None else players[start:stop],
                None if minutes is None else minutes[start:stop],
                None if fouls is None else fouls[start:stop],
            )
            pieces.append(remaining_pmf(mu, alpha).numpy())
    return np.concatenate(pieces, axis=0)


def _score(name: str, pmf: np.ndarray, frame: pd.DataFrame) -> dict:
    remaining = frame["remaining_points"].to_numpy()
    over_probability = probability_final_over(pmf, frame["current_total"].to_numpy(), RELIABILITY_THRESHOLD)
    observed = frame["final_total"].to_numpy() > RELIABILITY_THRESHOLD
    return {
        "name": name,
        "nll": nll_from_pmf(pmf, remaining),
        "crps": crps_from_pmf(pmf, remaining),
        "reliability": reliability_curve(over_probability, observed.astype(float)),
    }


def _plot(curves: list[dict], destination) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    for curve in curves:
        points = [row for row in curve["reliability"] if row["count"] > 0]
        axis.plot(
            [row["predicted_mean"] for row in points],
            [row["observed_rate"] for row in points],
            marker="o",
            label=curve["name"],
        )
    axis.set_xlabel(f"Predicted P(final total > {RELIABILITY_THRESHOLD})")
    axis.set_ylabel("Observed frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def evaluate(phase: int) -> dict:
    frame = load_training_frame(phase)
    frame["split"] = assign_splits(frame)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    model, payload = load_model(MODEL_DIR / f"phase{phase}.pt")
    curves = [_score(f"phase{phase}", network_pmf(model, payload, test), test)]
    if phase == 2 and (MODEL_DIR / "phase1.pt").exists():
        phase1, payload1 = load_model(MODEL_DIR / "phase1.pt")
        curves.append(_score("phase1_same_games", network_pmf(phase1, payload1, test), test))
    if (MODEL_DIR / "baseline_clock.npz").exists():
        clock = EmpiricalRemaining.load(MODEL_DIR / "baseline_clock.npz")
        margin = EmpiricalRemaining.load(MODEL_DIR / "baseline_margin.npz")
        seconds = test["seconds_left_in_regulation"].to_numpy()
        curves.append(_score("clock", clock.pmf(seconds), test))
        curves.append(_score("clock_and_margin", margin.pmf(seconds, test["score_margin"].to_numpy()), test))
    report = {
        "phase": phase,
        "threshold": RELIABILITY_THRESHOLD,
        "test_rows": int(len(test)),
        "test_games": int(test["game_id"].nunique()),
        "models": curves,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / f"phase{phase}.json"
    plot_path = EVAL_DIR / f"phase{phase}_reliability.png"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(curves, plot_path)
    for curve in curves:
        print(f"{curve['name']} nll={curve['nll']:.4f} crps={curve['crps']:.4f}", flush=True)
    print(f"wrote {json_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a held-out half of 2025-26.")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    evaluate(args.phase)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print the failure and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        raise
