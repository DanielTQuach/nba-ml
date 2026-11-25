import numpy as np
import pandas as pd
import pytest
import torch

from nba_ml.models.baseline import EmpiricalRemaining
from nba_ml.models.remaining import RemainingPointsNet, negative_binomial, remaining_pmf, shift_to_final_total
from nba_ml.models.train import assign_splits, train_model


def _tiny_frame() -> pd.DataFrame:
    rows = []
    schedule = [
        ("g0", "2023-01-01"),
        ("g1", "2023-02-01"),
        ("g2", "2024-01-01"),
        ("g3", "2024-02-01"),
        ("g4", "2025-10-20"),
        ("g5", "2025-11-20"),
        ("g6", "2026-01-15"),
        ("g7", "2026-03-15"),
    ]
    for game_id, date in schedule:
        for minute in range(4):
            current = 2 * minute
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(date),
                    "season": "2025-26" if date >= "2025-10-01" else "2023-24",
                    "period": 1,
                    "seconds_left_in_period": 720 - 60 * minute,
                    "seconds_left_in_regulation": 2880 - 60 * minute,
                    "current_total": current,
                    "score_margin": 1.0,
                    "home_fouls_period": 1,
                    "away_fouls_period": 2,
                    "home_team_id": 10,
                    "away_team_id": 20,
                    "home_pace": 100.0,
                    "home_off_rating": 110.0,
                    "home_def_rating": 108.0,
                    "away_pace": 99.0,
                    "away_off_rating": 112.0,
                    "away_def_rating": 107.0,
                    "final_total": 200,
                    "remaining_points": 200 - current,
                }
            )
    return pd.DataFrame(rows)


def test_split_keeps_every_row_of_a_game_on_one_side():
    labels = assign_splits(_tiny_frame())
    frame = _tiny_frame()
    for game_id, part in frame.groupby("game_id"):
        assert labels.loc[part.index].nunique() == 1
    assert set(labels[frame["game_id"] == "g0"]) == {"train"}
    assert set(labels[frame["game_id"] == "g7"]) == {"test"}


def test_negative_binomial_mean_is_mu():
    mu = torch.tensor([12.0, 4.0])
    alpha = torch.tensor([0.25, 0.5])
    assert torch.allclose(negative_binomial(mu, alpha).mean, mu, rtol=1e-4)


def test_final_total_probability_is_the_remaining_distribution_shifted_by_the_current_score():
    final = shift_to_final_total(np.array([0.1, 0.2, 0.7]), current_total=100)
    assert final[100] == pytest.approx(0.1)
    assert final[102] == pytest.approx(0.7)
    assert final.sum() == pytest.approx(1.0)


def test_network_pmf_sums_to_one():
    model = RemainingPointsNet(n_teams=3, numeric_dim=4, use_lineups=False)
    mu, alpha = model(torch.zeros(2, 4), torch.tensor([1, 2]), torch.tensor([2, 1]))
    probability = remaining_pmf(mu, alpha, max_count=30)
    assert torch.allclose(probability.sum(dim=1), torch.ones(2), atol=1e-4)


def test_baseline_pmf_sums_to_one_and_metrics_are_finite():
    seconds = np.array([2000, 2000, 60, 60])
    remaining = np.array([180, 170, 2, 4])
    margin = np.array([1, 1, -2, 8])
    clock = EmpiricalRemaining(with_margin=False)
    clock.fit(seconds, remaining)
    pmf = clock.pmf(seconds)
    assert np.allclose(pmf.sum(axis=1), 1.0)


def test_phase1_training_loop_writes_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("nba_ml.models.train.MODEL_DIR", tmp_path)
    destination = train_model(_tiny_frame(), phase=1, epochs=2, patience=2, batch_size=16)
    assert destination == tmp_path / "phase1.pt"
    assert (tmp_path / "baseline_clock.npz").exists()
    assert destination.exists()
