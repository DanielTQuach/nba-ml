"""Empirical distributions of remaining points, conditioned on the clock and margin."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nba_ml.config import REMAINING_PMF_MAX

MARGIN_EDGES = np.array([-15, -8, -3, 3, 8, 15], dtype=float)
TIME_BACKOFF = 5.0
MARGIN_BACKOFF = 20.0


def time_bucket(seconds_left_in_regulation: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(seconds_left_in_regulation, dtype=int) // 60, 0, 48)


def margin_bucket(score_margin: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(score_margin, dtype=float), MARGIN_EDGES)


class EmpiricalRemaining:
    """Smoothed histogram of remaining points.

    A margin-aware table backs off to the clock-only table, which backs off to
    the league-wide histogram, so unseen buckets stay defined.
    """

    def __init__(self, with_margin: bool) -> None:
        self.with_margin = with_margin
        self.global_counts = np.zeros(REMAINING_PMF_MAX + 1, dtype=np.float64)
        self.time_counts = np.zeros((49, REMAINING_PMF_MAX + 1), dtype=np.float64)
        self.joint_counts = np.zeros((49, len(MARGIN_EDGES) + 1, REMAINING_PMF_MAX + 1), dtype=np.float64)

    def fit(self, seconds_left: np.ndarray, remaining: np.ndarray, margin: np.ndarray | None = None) -> None:
        remaining = np.clip(np.asarray(remaining, dtype=int), 0, REMAINING_PMF_MAX)
        buckets = time_bucket(seconds_left)
        self.global_counts = np.bincount(remaining, minlength=REMAINING_PMF_MAX + 1).astype(np.float64)
        np.add.at(self.time_counts, (buckets, remaining), 1.0)
        if self.with_margin:
            if margin is None:
                raise ValueError("margin is required when the baseline uses it")
            margins = margin_bucket(margin)
            np.add.at(self.joint_counts, (buckets, margins, remaining), 1.0)

    def _time_pmf(self, buckets: np.ndarray) -> np.ndarray:
        global_pmf = self.global_counts + 0.1
        global_pmf = global_pmf / global_pmf.sum()
        counts = self.time_counts[buckets]
        pmf = counts + TIME_BACKOFF * global_pmf
        return pmf / pmf.sum(axis=1, keepdims=True)

    def pmf(self, seconds_left: np.ndarray, margin: np.ndarray | None = None) -> np.ndarray:
        buckets = time_bucket(seconds_left)
        time_pmf = self._time_pmf(buckets)
        if not self.with_margin:
            return time_pmf
        if margin is None:
            raise ValueError("margin is required when the baseline uses it")
        joint = self.joint_counts[buckets, margin_bucket(margin)]
        pmf = joint + MARGIN_BACKOFF * time_pmf
        return pmf / pmf.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            with_margin=np.array([1 if self.with_margin else 0]),
            global_counts=self.global_counts,
            time_counts=self.time_counts,
            joint_counts=self.joint_counts,
        )

    @classmethod
    def load(cls, path: Path) -> EmpiricalRemaining:
        payload = np.load(path)
        model = cls(with_margin=bool(int(payload["with_margin"][0])))
        model.global_counts = payload["global_counts"]
        model.time_counts = payload["time_counts"]
        model.joint_counts = payload["joint_counts"]
        return model
