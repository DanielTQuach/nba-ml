"""Scores for a discrete distribution over remaining points."""

from __future__ import annotations

import numpy as np


def nll_from_pmf(pmf: np.ndarray, remaining: np.ndarray) -> float:
    remaining = np.asarray(remaining, dtype=int)
    clipped = np.clip(remaining, 0, pmf.shape[1] - 1)
    probability = pmf[np.arange(len(remaining)), clipped]
    probability = np.where(remaining >= pmf.shape[1], 1e-12, probability)
    return float(-np.log(np.clip(probability, 1e-12, 1.0)).mean())


def crps_from_pmf(pmf: np.ndarray, remaining: np.ndarray) -> float:
    remaining = np.asarray(remaining, dtype=int)
    cdf = np.cumsum(pmf, axis=1)
    grid = np.arange(pmf.shape[1])
    indicators = grid[None, :] >= remaining[:, None]
    return float(np.mean(np.sum((cdf - indicators) ** 2, axis=1)))


def final_total_sides(final_pmf: np.ndarray, line: float) -> dict[str, float]:
    """Split a final-total distribution into over, under, and push.

    Final totals are whole numbers, so a half-point line has no push.
    """
    final_pmf = np.asarray(final_pmf, dtype=float).reshape(-1)
    totals = np.arange(len(final_pmf))
    over = float(final_pmf[totals > line].sum())
    under = float(final_pmf[totals < line].sum())
    if float(line).is_integer() and 0 <= int(line) < len(final_pmf):
        push = float(final_pmf[int(line)])
    else:
        push = 0.0
    return {"over": over, "under": under, "push": push}


def probability_final_over(pmf: np.ndarray, current_total: np.ndarray, threshold: float) -> np.ndarray:
    """P(final total > threshold) from a remaining-points PMF."""
    current_total = np.asarray(current_total, dtype=float)
    need = threshold - current_total
    probabilities = np.empty(len(pmf), dtype=float)
    for index, minimum in enumerate(need):
        if minimum < 0:
            probabilities[index] = 1.0
            continue
        start = int(np.floor(minimum)) + 1
        if start >= pmf.shape[1]:
            probabilities[index] = 0.0
        else:
            probabilities[index] = float(pmf[index, start:].sum())
    return probabilities


def reliability_curve(
    predicted: np.ndarray, observed: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right == 1.0:
            mask = (predicted >= left) & (predicted <= right)
        else:
            mask = (predicted >= left) & (predicted < right)
        count = int(mask.sum())
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "count": count,
                "predicted_mean": float(predicted[mask].mean()) if count else float("nan"),
                "observed_rate": float(observed[mask].mean()) if count else float("nan"),
            }
        )
    return rows
