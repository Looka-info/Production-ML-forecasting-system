"""Unit tests for VN1 forecasting baselines."""

import pandas as pd
import pytest

from demandcast.vn1.baselines import (
    VN1MovingAverageForecaster,
    VN1NaiveForecaster,
)


@pytest.fixture
def sample_history():
    """Create 20 weeks of synthetic Monday history for 2 series."""
    dates = pd.date_range(start="2023-01-02", periods=20, freq="W-MON")
    records = []
    for d_idx, d in enumerate(dates):
        records.append(
            {
                "series_id": "0_1_100",
                "week_start": d,
                "sales": float(d_idx + 1),  # 1..20
            }
        )
        records.append(
            {
                "series_id": "0_1_200",
                "week_start": d,
                "sales": 50.0,  # constant 50
            }
        )
    return pd.DataFrame(records)


def test_vn1_naive_forecaster(sample_history):
    """Naive forecaster repeats the last observed value across 13 weeks."""
    forecaster = VN1NaiveForecaster()
    preds = forecaster.predict(sample_history, horizon=13)

    assert len(preds) == 26  # 2 series x 13 horizons
    assert set(preds["horizon_step"].unique()) == set(range(1, 14))

    # Series 1: last value is 20.0
    s1_preds = preds[preds["series_id"] == "0_1_100"]
    assert (s1_preds["predicted_sales"] == 20.0).all()

    # Series 2: constant 50.0
    s2_preds = preds[preds["series_id"] == "0_1_200"]
    assert (s2_preds["predicted_sales"] == 50.0).all()


def test_vn1_moving_average_12_forecaster(sample_history):
    """12-week moving average takes the mean of trailing 12 observations."""
    forecaster = VN1MovingAverageForecaster(window_size=12)
    preds = forecaster.predict(sample_history, horizon=13)

    assert len(preds) == 26
    # Series 1: trailing 12 weeks are 9..20. Mean = (9+20)/2 = 14.5
    s1_preds = preds[preds["series_id"] == "0_1_100"]
    assert (s1_preds["predicted_sales"] == 14.5).all()

    # Series 2: constant 50.0 -> mean = 50.0
    s2_preds = preds[preds["series_id"] == "0_1_200"]
    assert (s2_preds["predicted_sales"] == 50.0).all()


def test_vn1_moving_average_4_demandcast(sample_history):
    """4-week moving average takes the mean of trailing 4 observations."""
    forecaster = VN1MovingAverageForecaster(window_size=4)
    preds = forecaster.predict(sample_history, horizon=13)

    # Series 1: trailing 4 weeks are 17, 18, 19, 20. Mean = 18.5
    s1_preds = preds[preds["series_id"] == "0_1_100"]
    assert (s1_preds["predicted_sales"] == 18.5).all()
