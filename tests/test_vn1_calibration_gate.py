"""Unit tests for VN1 Pre-Origin Calibration Eligibility Gate."""

import pandas as pd
import pytest

from demandcast.vn1.calibration_gate import (
    CalibrationGateConfig,
    PreOriginCalibrationGate,
)


@pytest.fixture
def sample_history():
    """Create a controlled 52-week history for 3 test series."""
    dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
    rows = []

    # Series A: High-volume, smooth, holiday surge (MA12 > MA26) -> Eligible
    for i, d in enumerate(dates):
        # Base demand 10 units, surges to 30 in last 12 weeks
        val = 30.0 if i >= 40 else 10.0
        rows.append({"series_id": "series_eligible", "week_start": d, "sales": val})

    # Series B: Low-volume (< 35 total in trailing 12 weeks) -> Ineligible (Volume)
    for i, d in enumerate(dates):
        val = 1.0 if i >= 40 else 0.5
        rows.append({"series_id": "series_low_volume", "week_start": d, "sales": val})

    # Series C: High volume but highly sparse (> 50% zeroes) -> Ineligible (Sparsity)
    for i, d in enumerate(dates):
        # Zeroes on most weeks, occasional 50 units
        val = 50.0 if i in [42, 48] else 0.0
        rows.append({"series_id": "series_sparse", "week_start": d, "sales": val})

    # Series D: High volume, smooth, but declining (MA12 < MA26) -> Ineligible (Elevation)
    for i, d in enumerate(dates):
        val = 5.0 if i >= 40 else 25.0
        rows.append({"series_id": "series_declining", "week_start": d, "sales": val})

    return pd.DataFrame(rows)


def test_gate_eligibility_classification(sample_history):
    origin = pd.to_datetime("2022-12-26")  # Week 52
    config = CalibrationGateConfig(
        min_trailing_12w_volume=35.0,
        max_zero_demand_fraction=0.50,
        require_holiday_surge_elevation=True,
    )
    gate = PreOriginCalibrationGate(config)
    res = gate.evaluate_series_eligibility(sample_history, origin)

    assert "series_eligible" in res.index
    assert res.loc["series_eligible", "is_eligible"]
    assert res.loc["series_eligible", "is_volume_eligible"]
    assert res.loc["series_eligible", "is_sparsity_eligible"]
    assert res.loc["series_eligible", "is_elevation_eligible"]

    assert not res.loc["series_low_volume", "is_eligible"]
    assert not res.loc["series_low_volume", "is_volume_eligible"]

    assert not res.loc["series_sparse", "is_eligible"]
    assert not res.loc["series_sparse", "is_sparsity_eligible"]

    assert not res.loc["series_declining", "is_eligible"]
    assert not res.loc["series_declining", "is_elevation_eligible"]


def test_gate_point_in_time_enforcement(sample_history):
    """Verify observations strictly after origin_date are ignored."""
    origin = pd.to_datetime("2022-06-06")  # Week 23

    # Add a huge spike AFTER origin date to series_low_volume
    leaked_history = sample_history.copy()
    future_mask = (leaked_history["series_id"] == "series_low_volume") & (leaked_history["week_start"] > origin)
    leaked_history.loc[future_mask, "sales"] = 1000.0

    gate = PreOriginCalibrationGate()
    res = gate.evaluate_series_eligibility(leaked_history, origin)

    # Series low volume should remain strictly based on data <= origin
    assert not res.loc["series_low_volume", "is_eligible"]
    assert res.loc["series_low_volume", "trailing_12w_volume"] <= 35.0


def test_gate_fail_closed_empty_history():
    """Verify empty history fail-closes safely."""
    gate = PreOriginCalibrationGate()
    empty_df = pd.DataFrame(columns=["series_id", "week_start", "sales"])
    res = gate.evaluate_series_eligibility(empty_df, "2023-01-02")
    assert res.empty


def test_apply_gated_forecast(sample_history):
    origin = pd.to_datetime("2022-12-26")
    gate = PreOriginCalibrationGate()
    elig_df = gate.evaluate_series_eligibility(sample_history, origin)

    # Mock predictions
    base_rows = []
    cand_rows = []
    for sid in ["series_eligible", "series_low_volume"]:
        for step in range(1, 4):
            wk = origin + pd.Timedelta(weeks=step)
            base_rows.append({"series_id": sid, "week_start": wk, "predicted_sales": 10.0})
            cand_rows.append(
                {
                    "series_id": sid,
                    "week_start": wk,
                    "predicted_sales": 8.0,
                    "calibration_factor": 0.8,
                }
            )

    df_base = pd.DataFrame(base_rows)
    df_cand = pd.DataFrame(cand_rows)

    gated = gate.apply_gated_forecast(df_base, df_cand, elig_df)

    # Eligible series should receive calibrated prediction (8.0)
    elig_preds = gated[gated["series_id"] == "series_eligible"]["predicted_sales"].tolist()
    assert elig_preds == [8.0, 8.0, 8.0]

    # Ineligible series should receive baseline fallback (10.0)
    inelig_preds = gated[gated["series_id"] == "series_low_volume"]["predicted_sales"].tolist()
    assert inelig_preds == [10.0, 10.0, 10.0]
    inelig_factors = gated[gated["series_id"] == "series_low_volume"]["calibration_factor"].tolist()
    assert inelig_factors == [1.0, 1.0, 1.0]
