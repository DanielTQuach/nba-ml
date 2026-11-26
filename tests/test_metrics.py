import numpy as np
import pytest

from nba_ml.eval.metrics import crps_from_pmf, final_total_sides, nll_from_pmf, probability_final_over
from nba_ml.models.baseline import EmpiricalRemaining


def test_scoring_metrics_are_finite_and_an_already_crossed_total_has_probability_one():
    seconds = np.array([2000, 2000, 60, 60])
    remaining = np.array([180, 170, 2, 4])
    clock = EmpiricalRemaining(with_margin=False)
    clock.fit(seconds, remaining)
    pmf = clock.pmf(seconds)
    assert nll_from_pmf(pmf, remaining) > 0
    assert crps_from_pmf(pmf, remaining) >= 0
    over = probability_final_over(pmf, current_total=np.array([20, 40, 210, 230]), threshold=220)
    assert over[3] == pytest.approx(1.0)


def test_a_half_point_line_has_no_push_and_a_whole_line_keeps_the_tie():
    final = np.zeros(11)
    final[8] = 0.2
    final[10] = 0.8
    whole = final_total_sides(final, 10)
    assert whole["over"] == pytest.approx(0.0)
    assert whole["under"] == pytest.approx(0.2)
    assert whole["push"] == pytest.approx(0.8)
    half = final_total_sides(final, 9.5)
    assert half["push"] == pytest.approx(0.0)
    assert half["over"] == pytest.approx(0.8)
    assert half["under"] == pytest.approx(0.2)
