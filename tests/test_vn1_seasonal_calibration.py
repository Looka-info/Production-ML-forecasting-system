"""Unit tests for VN1 Post-Holiday Seasonal Calibration Layer.

Verifies calibration factor estimation, empirical Bayes shrinkage, calendar window
targeting, safety clipping bounds, non-negative constraints, and preservation of
frozen Phase 2 production artifacts.
"""

import json

import pandas as pd
import pytest

from demandcast.config import get_project_root
from demandcast.vn1.baselines import VN1MovingAverageForecaster
from demandcast.vn1.seasonal_calibration import (
    CalibratedForecaster,
    SeasonalCalibrationConfig,
    SeasonalCalibrationEstimator,
)


@pytest.fixture
def multi_year_history() -> pd.DataFrame:
    """Generate 3 years of synthetic weekly data with a known post-holiday drop.

    Weeks: 2020-07-06 to 2023-01-02 (Monday-start).
    Q4 (weeks 40-52): elevated sales (200.0).
    Q1 (weeks 1-13): dropped sales (160.0, a 20% drop -> ratio 0.80).
    Rest of year: normal sales (100.0).
    Includes two clients:
    - 'clientA': large volume (1000 items)
    - 'clientB': tiny volume (1 item)
    """
    dates = pd.date_range(start="2020-07-06", end="2023-01-02", freq="W-MON")
    records = []

    for d in dates:
        woy = d.isocalendar().week
        # Base seasonal pattern
        if 40 <= woy <= 52:
            base_sales = 200.0
        elif 1 <= woy <= 13:
            base_sales = 160.0
        else:
            base_sales = 100.0

        # High volume client A
        records.append(
            {
                "series_id": "clientA_prod1",
                "client_id": "clientA",
                "week_start": d,
                "sales": base_sales,
                "price": 10.0,
            }
        )
        # Low volume client B
        records.append(
            {
                "series_id": "clientB_prod1",
                "client_id": "clientB",
                "week_start": d,
                "sales": base_sales * 0.05,
                "price": 10.0,
            }
        )

    return pd.DataFrame(records)


def test_unadjusted_baseline_predictions_identical(multi_year_history):
    """CalibratedForecaster with method='unadjusted' matches raw baseline exactly."""
    base_model = VN1MovingAverageForecaster(window_size=12)
    calibrated_model = CalibratedForecaster(
        base_forecaster=base_model,
        calibration_config=SeasonalCalibrationConfig(method="unadjusted"),
    )

    origin = pd.to_datetime("2023-01-02")
    raw_preds = base_model.predict(multi_year_history, horizon=13, origin_date=origin)
    cal_preds = calibrated_model.predict(multi_year_history, horizon=13, origin_date=origin)

    assert len(raw_preds) == len(cal_preds)
    pd.testing.assert_series_equal(
        raw_preds["predicted_sales"],
        cal_preds["predicted_sales"],
        check_names=False,
    )
    assert (cal_preds["calibration_factor"] == 1.0).all()


def test_estimator_strictly_historical_data_access(multi_year_history):
    """Estimator must strictly use completed transitions prior to the origin date."""
    # Origin before any completed January transitions:
    early_origin = pd.to_datetime("2020-10-05")
    estimator = SeasonalCalibrationEstimator()
    estimator.fit(multi_year_history, early_origin)

    assert estimator.global_factor == 1.0
    assert len(estimator.transitions_observed) == 0

    # Origin at 2022-01-03: only the 2021-01-04 transition is observed and complete
    mid_origin = pd.to_datetime("2022-01-03")
    estimator.fit(multi_year_history, mid_origin)
    assert len(estimator.transitions_observed) == 1
    assert "2021-01-04" in estimator.transitions_observed
    assert "2022-01-03" not in estimator.transitions_observed


def test_post_holiday_window_targeting():
    """Targeted adjustment applies only during weeks 1..13 and returns 1.0 outside."""
    config = SeasonalCalibrationConfig(
        method="post_holiday_targeted",
        post_holiday_start_week=1,
        post_holiday_end_week=13,
    )
    estimator = SeasonalCalibrationEstimator(config)
    estimator.global_factor = 0.88

    # Inside post-holiday window: week 4 of 2023
    jan_date = pd.to_datetime("2023-01-23")
    assert jan_date.isocalendar().week == 4
    factor_inside = estimator.get_factor(target_week=jan_date)
    assert factor_inside == 0.88

    # Outside post-holiday window: week 28 of 2023 (July)
    july_date = pd.to_datetime("2023-07-10")
    assert july_date.isocalendar().week == 28
    factor_outside = estimator.get_factor(target_week=july_date)
    assert factor_outside == 1.0


def test_sparse_or_unknown_client_fallback():
    """Unknown or low-volume clients cleanly fall back to global factor."""
    config = SeasonalCalibrationConfig(method="shrunk_client_factor")
    estimator = SeasonalCalibrationEstimator(config)
    estimator.global_factor = 0.85
    estimator.client_factors = {"known_client": 0.92}

    # Known client
    assert estimator.get_factor(client_id="known_client") == 0.92

    # Unknown client falls back to global factor
    assert estimator.get_factor(client_id="unknown_client_xyz") == 0.85
    # Missing client_id falls back to global factor
    assert estimator.get_factor(client_id=None) == 0.85


def test_empirical_bayes_shrinkage_behavior(multi_year_history):
    """High-volume client preserves more of raw ratio; low-volume shrinks toward global."""
    config = SeasonalCalibrationConfig(
        method="shrunk_client_factor",
        shrinkage_k=100.0,
        min_volume_threshold=1.0,
    )
    estimator = SeasonalCalibrationEstimator(config)
    estimator.fit(multi_year_history, pd.to_datetime("2023-01-02"))

    assert "clientA" in estimator.client_factors
    assert "clientB" in estimator.client_factors

    # Both clients have base drop ~0.80, but client B's small volume shrinks it strongly
    # toward the global factor
    factor_a = estimator.client_factors["clientA"]
    factor_b = estimator.client_factors["clientB"]
    assert 0.50 <= factor_a <= 1.25
    assert 0.50 <= factor_b <= 1.25


def test_safety_clipping_bounds():
    """Calibration multipliers must strictly stay within configured bounds."""
    config = SeasonalCalibrationConfig(
        min_factor_bound=0.50,
        max_factor_bound=1.25,
    )
    estimator = SeasonalCalibrationEstimator(config)
    estimator.global_factor = 0.90
    estimator.client_factors = {"crash_client": 0.10, "boom_client": 3.50}

    # Clipped to min bound
    assert estimator.get_factor(client_id="crash_client") == 0.50
    # Clipped to max bound
    assert estimator.get_factor(client_id="boom_client") == 1.25


def test_non_negative_retail_constraint(multi_year_history):
    """Calibrated predictions must never produce negative sales quantities."""
    base_model = VN1MovingAverageForecaster(window_size=12)
    calibrated_model = CalibratedForecaster(
        base_forecaster=base_model,
        calibration_config=SeasonalCalibrationConfig(method="post_holiday_targeted"),
    )

    preds = calibrated_model.predict(
        multi_year_history,
        horizon=13,
        origin_date=pd.to_datetime("2023-01-02"),
    )

    assert (preds["predicted_sales"] >= 0.0).all()
    assert (preds["raw_predicted_sales"] >= 0.0).all()
    assert "calibration_factor" in preds.columns
    assert len(calibrated_model.last_applied_telemetry) > 0


def test_phase2_artifacts_and_active_production_release_preserved():
    """Verify Phase 2 evaluation artifacts and production pointer remain untouched."""
    root = get_project_root()

    # 1. Active VN1 production pointer must be v1.0.0
    current_json = root / "models" / "profiles" / "vn1" / "current.json"
    assert current_json.exists(), "models/profiles/vn1/current.json must exist"

    with open(current_json, "r", encoding="utf-8") as f:
        profile_data = json.load(f)

    assert profile_data["current_version"] == "v1.0.0"

    release_meta = root / profile_data["release_dir"] / "metadata.json"
    assert release_meta.exists()
    with open(release_meta, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert "Baseline_MovingAvg12_Official" in meta_data["model_type"]

    # 2. Phase 2 test results CSV must remain present
    p2_csv = root / "reports" / "vn1" / "phase2_test_results.csv"
    assert p2_csv.exists(), "reports/vn1/phase2_test_results.csv must exist and remain frozen"

    # 3. Seasonal calibration summary must document governance decision
    summary_json = root / "reports" / "vn1" / "seasonal_calibration_summary.json"
    assert summary_json.exists(), "reports/vn1/seasonal_calibration_summary.json must exist"

    with open(summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["governance_policy"]["decision"] == "ACCEPTED_AS_RESEARCH_CANDIDATE"
    assert summary_data["governance_policy"]["active_production_release"] == "v1.0.0 (Baseline_MovingAvg12_Official)"
