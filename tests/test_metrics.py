"""Tests for demand forecasting evaluation metrics and edge case handling."""

import numpy as np
import pytest

from demandcast.metrics import (
    forecast_bias,
    mean_absolute_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)


def test_standard_metrics_hand_calculated():
    """Verify metrics match exact hand-calculated figures.

    y_true: [10, 20, 30]
    y_pred: [12, 18, 30]
    diffs: [2, 2, 0]
    MAE = 4/3 = 1.3333
    RMSE = sqrt((4 + 4 + 0)/3) = sqrt(8/3) = 1.63299
    WAPE = 4 / 60 = 0.0667 (6.67%)
    Bias = ((12-10) + (18-20) + (30-30)) / 3 = 0.0
    """
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 30.0])

    mae = mean_absolute_error(yt, yp)
    rmse = root_mean_squared_error(yt, yp)
    wape = weighted_absolute_percentage_error(yt, yp)
    bias = forecast_bias(yt, yp)

    assert pytest.approx(mae, 0.001) == 1.3333
    assert pytest.approx(rmse, 0.001) == 1.6330
    assert pytest.approx(wape, 0.001) == 0.0667
    assert pytest.approx(bias, 0.001) == 0.0


def test_wape_zero_demand_edge_case():
    """Verify that WAPE returns None when total actual demand is zero to prevent division by zero."""
    yt = np.array([0.0, 0.0, 0.0])
    yp = np.array([5.0, 2.0, 0.0])
    assert weighted_absolute_percentage_error(yt, yp) is None


def test_smape_zero_handling():
    """Verify sMAPE handles zero-zero matches without division by zero errors."""
    yt = np.array([0.0, 10.0])
    yp = np.array([0.0, 10.0])
    smape = symmetric_mean_absolute_percentage_error(yt, yp)
    assert smape == 0.0


def test_mase_calculation():
    """Verify MASE scales error against in-sample naive persistence differences."""
    train_hist = np.array([10.0, 12.0, 14.0, 16.0])  # diffs are all 2.0 -> scale = 2.0
    yt = np.array([18.0, 20.0])
    yp = np.array([17.0, 21.0])  # errors: 1.0, 1.0 -> MAE = 1.0
    mase = mean_absolute_scaled_error(yt, yp, train_hist)
    assert mase == 0.5  # 1.0 / 2.0


def test_mase_zero_denominator_constant_history():
    """Verify MASE returns None when in-sample training history is constant (scale denominator = 0)."""
    train_hist_constant = np.array([10.0, 10.0, 10.0, 10.0])  # diffs are all 0 -> scale = 0
    yt = np.array([15.0, 20.0])
    yp = np.array([14.0, 19.0])
    mase = mean_absolute_scaled_error(yt, yp, train_hist_constant)
    assert mase is None, "MASE must return None when training-side scale denominator is zero."


def test_relative_bias_and_raw_bias_definitions():
    """Verify raw bias is in forecast units mean(yp - yt) while relative bias is dimensionless ratio sum(yp - yt)/sum(yt)."""
    from demandcast.metrics import relative_forecast_bias

    yt = np.array([100.0, 200.0])  # sum = 300
    yp = np.array([150.0, 210.0])  # diffs: +50, +10 -> sum diff = +60, mean diff = +30.0

    raw_b = forecast_bias(yt, yp)
    rel_b = relative_forecast_bias(yt, yp)

    assert pytest.approx(raw_b, 0.001) == 30.0  # units
    assert pytest.approx(rel_b, 0.001) == 0.20  # +60/300 = +0.20 (+20%)
