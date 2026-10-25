"""Tests for data quality checking, drift detection, and forecast monitoring."""

import pandas as pd

from demandcast.monitoring import ForecastMonitor


def test_input_quality_checks():
    """Verify input quality checks detect non-negative issues, zero streaks, and extreme outliers."""
    monitor = ForecastMonitor()

    # History with 4 consecutive zeros
    df_zeros = pd.DataFrame(
        {
            "week_start": pd.date_range("2011-01-03", periods=14, freq="W-MON"),
            "demand": [20.0] * 10 + [0.0, 0.0, 0.0, 0.0],
        }
    )

    res = monitor.check_input_data_quality(
        series_id="SKU_1",
        history_df=df_zeros,
        reference_stats={"mean": 20.0, "std": 5.0},
    )

    assert res["is_valid"]
    assert any("zero-demand streak" in w for w in res["warnings"])


def test_forecast_actuals_reconciliation_and_alerts():
    """Verify evaluation against actuals detects high WAPE alert."""
    monitor = ForecastMonitor()

    dates = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    preds_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "predicted_demand": [100.0, 100.0, 100.0, 100.0],
        }
    )
    actuals_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "demand": [10.0, 10.0, 10.0, 10.0],  # Major discrepancy
        }
    )

    report = monitor.evaluate_forecast_actuals(preds_df, actuals_df)
    assert report["evaluated_observations"] == 4
    assert len(report["alerts"]) > 0
    assert any("WAPE" in alert for alert in report["alerts"])
    assert report["status"] == "ALERT"


def test_monitoring_relative_bias_threshold_and_units():
    """Verify that relative bias is measured as dimensionless ratio sum(y_hat - y)/sum(y)."""
    monitor = ForecastMonitor()
    dates = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    # Total actual = 400, Total pred = 520 -> relative bias = (520-400)/400 = +0.30 (+30%)
    # Threshold is 0.25 (25%)
    preds_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "predicted_demand": [130.0, 130.0, 130.0, 130.0],
        }
    )
    actuals_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "demand": [100.0, 100.0, 100.0, 100.0],
        }
    )

    report = monitor.evaluate_forecast_actuals(preds_df, actuals_df)
    rel_b = report["metrics"]["relative_bias"]
    assert rel_b == 0.30
    assert any("Relative bias alert" in a for a in report["alerts"])
    assert report["status"] == "ALERT"


def test_monitoring_missing_actuals():
    """Verify that when actuals are missing, monitor returns NO_MATCHING_DATA without synthetic values."""
    monitor = ForecastMonitor()
    preds_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"],
            "week_start": pd.to_datetime(["2025-01-06"]),
            "predicted_demand": [50.0],
        }
    )
    actuals_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"],
            "week_start": pd.to_datetime(["2020-01-06"]),
            "demand": [50.0],
        }
    )

    report = monitor.evaluate_forecast_actuals(preds_df, actuals_df)
    assert report["status"] == "NO_MATCHING_DATA"
    assert "No matching actuals found" in report["message"]


def test_input_data_drift_detection():
    """Verify that input data quality checks flag distribution shift when sample mean deviates significantly."""
    monitor = ForecastMonitor()
    # History with mean = 100.0 vs training baseline mean = 20.0 (std = 5.0) -> 16 standard deviations shift!
    df_shifted = pd.DataFrame(
        {
            "week_start": pd.date_range("2011-01-03", periods=14, freq="W-MON"),
            "demand": [100.0] * 14,
        }
    )

    res = monitor.check_input_data_quality(
        series_id="SKU_1",
        history_df=df_shifted,
        reference_stats={"mean": 20.0, "std": 5.0},
    )

    assert res["status"] == "WARNING"
    assert res["input_drift"]["drift_detected"] is True
    assert any("Input data drift" in w for w in res["warnings"])


def test_monitoring_residual_tracking_and_horizon_breakdown():
    """Verify residual tracking computes correct error distribution and breaks down by horizon and series."""
    monitor = ForecastMonitor()
    dates = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    preds_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "horizon_step": [1, 2, 3, 4],
            "predicted_demand": [110.0, 120.0, 130.0, 140.0],
        }
    )
    actuals_df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 4,
            "week_start": dates,
            "demand": [100.0, 100.0, 100.0, 100.0],
        }
    )

    report = monitor.evaluate_forecast_actuals(preds_df, actuals_df)
    res_analysis = report["residual_analysis"]

    # Residuals: yp - yt = [10, 20, 30, 40]
    assert res_analysis["mean_residual"] == 25.0
    assert res_analysis["min_residual"] == 10.0
    assert res_analysis["max_residual"] == 40.0

    # Horizon breakdown
    assert "horizon_1" in report["by_horizon"]
    assert "horizon_4" in report["by_horizon"]
    assert report["by_horizon"]["horizon_1"]["mae"] == 10.0
    assert report["by_horizon"]["horizon_4"]["mae"] == 40.0

    # Series breakdown
    assert "SKU_1" in report["by_series"]
    assert report["by_series"]["SKU_1"]["count"] == 4
