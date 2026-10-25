"""Unit tests for official VN1 competition scoring and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from demandcast.vn1.metrics import (
    calculate_vn1_score,
    compute_all_vn1_metrics,
)


def test_official_vn1_score_hand_calculated():
    """Verify that calculate_vn1_score exactly matches official formula:

    Score = (sum |y_hat - y| + |sum (y_hat - y)|) / sum y
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 38.0])

    # sum y = 100.0
    # errors: [+2, -2, +5, -2]
    # abs_errors: [2, 2, 5, 2] -> sum = 11.0
    # sum errors = 3.0 -> abs(sum errors) = 3.0
    # expected score = (11.0 + 3.0) / 100.0 = 14.0 / 100.0 = 0.14
    score = calculate_vn1_score(y_true, y_pred)
    assert pytest.approx(score, rel=1e-5) == 0.14


def test_vn1_score_perfect_predictions():
    """Score should be exactly 0.0 for zero forecast error."""
    y = np.array([15.0, 30.0, 45.0, 60.0])
    score = calculate_vn1_score(y, y)
    assert score == 0.0


def test_vn1_score_zero_demand_handling():
    """Handle edge case where total actual demand is zero."""
    # When both actuals and predictions are 0, score is 0.0
    y_zero = np.array([0.0, 0.0, 0.0])
    y_pred_zero = np.array([0.0, 0.0, 0.0])
    assert calculate_vn1_score(y_zero, y_pred_zero) == 0.0

    # When actuals are 0 but predictions are positive, score is nan
    y_pred_pos = np.array([1.0, 2.0, 3.0])
    assert np.isnan(calculate_vn1_score(y_zero, y_pred_pos))


def test_compute_all_vn1_metrics_parity():
    """Verify complete dictionary output."""
    y_true = pd.Series([100.0, 150.0, 200.0, 250.0])
    y_pred = pd.Series([110.0, 140.0, 210.0, 240.0])

    metrics = compute_all_vn1_metrics(y_true, y_pred)
    assert "vn1_score" in metrics
    assert "wape" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "smape" in metrics
    assert "relative_bias" in metrics
    assert metrics["mae"] == 10.0
    assert metrics["wape"] == pytest.approx(40.0 / 700.0, abs=1e-4)
