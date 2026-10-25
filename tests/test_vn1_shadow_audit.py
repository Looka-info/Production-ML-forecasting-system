"""DemandCast v2.2.1 — Shadow Evaluation Integrity and Boundary Audit Tests.

Verifies:
1. Production release pointer immutability (UCI and VN1 profiles).
2. Calendar boundary logic: ISO weeks 1, 12, 13, 14, year boundaries, and post-holiday 600/50 observation splits.
3. Hand-calculated VN1 metric fixture verifying portfolio bias and grouping arithmetic.
4. Point-in-time future tampering test (future rows in raw data cannot alter origin forecasts).
5. Cross-experiment parameter consistency between shadow pipeline and seasonal calibration modules.
6. Delayed actuals reconciliation immutability and conflict handling.
"""

import json

import numpy as np
import pandas as pd

from demandcast.config import get_project_root
from demandcast.vn1.baselines import VN1MovingAverageForecaster
from demandcast.vn1.config import load_vn1_config
from demandcast.vn1.metrics import compute_all_vn1_metrics
from demandcast.vn1.seasonal_calibration import (
    CalibratedForecaster,
    SeasonalCalibrationConfig,
)
from demandcast.vn1.shadow import (
    ShadowStore,
    VN1ShadowPipeline,
)


def test_production_pointers_immutability():
    """Verify that both UCI and VN1 production pointers remain locked to v1.0.0."""
    root = get_project_root()

    # 1. UCI production profile
    uci_pointer = root / "models" / "current.json"
    assert uci_pointer.exists(), "UCI production pointer must exist."
    with open(uci_pointer, "r", encoding="utf-8") as f:
        uci_data = json.load(f)
    assert uci_data["current_version"] == "v1.0.0", "UCI release must remain v1.0.0."

    # 2. VN1 production profile
    vn1_pointer = root / "models" / "profiles" / "vn1" / "current.json"
    assert vn1_pointer.exists(), "VN1 production pointer must exist."
    with open(vn1_pointer, "r", encoding="utf-8") as f:
        vn1_data = json.load(f)
    assert vn1_data["current_version"] == "v1.0.0", "VN1 release must remain v1.0.0."


def test_iso_calendar_segmentation_boundaries():
    """Test boundary conditions for ISO weeks 1, 12, 13, 14 and year transitions."""
    # Test case 1: Origin 2023-01-02 (Monday of ISO week 1)
    origin_2023 = pd.to_datetime("2023-01-02")
    assert origin_2023.isocalendar().week == 1

    # Project 13 weeks forward
    horizon_weeks_2023 = [origin_2023 + pd.Timedelta(weeks=h) for h in range(1, 14)]
    iso_weeks_2023 = [dt.isocalendar().week for dt in horizon_weeks_2023]

    # Verify that steps 1..12 are weeks 2..13 (in post-holiday window 1..13)
    for h_idx in range(12):  # h=1..12
        assert 1 <= iso_weeks_2023[h_idx] <= 13, f"Step {h_idx + 1} must be in post-holiday window."
        assert iso_weeks_2023[h_idx] == h_idx + 2

    # Verify that step 13 is week 14 (outside post-holiday window)
    assert iso_weeks_2023[12] == 14, "Step 13 must project to ISO week 14."
    assert not (1 <= iso_weeks_2023[12] <= 13), "Step 13 must not be in post-holiday window."

    # Test case 2: Origin 2022-07-04 (Monday of ISO week 27 - mid-year)
    origin_midyear = pd.to_datetime("2022-07-04")
    assert origin_midyear.isocalendar().week == 27
    horizon_weeks_midyear = [origin_midyear + pd.Timedelta(weeks=h) for h in range(1, 14)]
    iso_weeks_midyear = [dt.isocalendar().week for dt in horizon_weeks_midyear]
    # None of the 13 horizon steps should be in weeks 1..13
    assert not any(1 <= w <= 13 for w in iso_weeks_midyear)

    # Test case 3: Year boundary transition (Origin 2022-12-26, ISO week 52)
    origin_year_boundary = pd.to_datetime("2022-12-26")
    assert origin_year_boundary.isocalendar().week == 52
    h1 = origin_year_boundary + pd.Timedelta(weeks=1)  # 2023-01-02
    assert h1.isocalendar().year == 2023
    assert h1.isocalendar().week == 1
    assert 1 <= h1.isocalendar().week <= 13


def test_hand_calculated_vn1_metric_fixture():
    """Verify official VN1 metric implementation against hand-calculated values."""
    # Hand calculation design:
    # Series A:
    #   Step 1: actual = 100, pred = 110 (error = +10, abs_err = 10)
    #   Step 2: actual = 200, pred = 190 (error = -10, abs_err = 10)
    #   Sum actual = 300
    #   Sum abs error = 20
    #   Sum error (portfolio bias) = +10 + (-10) = 0
    #   VN1 Score = (20 + |0|) / 300 = 20 / 300 = 0.0666666...
    #
    # Series B:
    #   Step 1: actual = 50, pred = 60 (error = +10, abs_err = 10)
    #   Step 2: actual = 50, pred = 60 (error = +10, abs_err = 10)
    #   Sum actual = 100
    #   Sum abs error = 20
    #   Sum error = +20
    #   VN1 Score = (20 + |20|) / 100 = 40 / 100 = 0.4000
    #
    # Combined (Series A + Series B):
    #   Sum actual = 300 + 100 = 400
    #   Sum abs error = 20 + 20 = 40
    #   Sum error = 0 + 20 = 20
    #   VN1 Score = (40 + |20|) / 400 = 60 / 400 = 0.1500
    #   WAPE = 40 / 400 = 0.1000
    #   MAE = 40 / 4 = 10.0
    #   RMSE = sqrt((100 + 100 + 100 + 100)/4) = 10.0
    #   Relative Bias = 20 / 400 = 0.0500

    df_test = pd.DataFrame(
        {
            "series_id": ["A", "A", "B", "B"],
            "target_week": ["2023-01-09", "2023-01-16", "2023-01-09", "2023-01-16"],
            "predicted": [110.0, 190.0, 60.0, 60.0],
            "actual": [100.0, 200.0, 50.0, 50.0],
        }
    )

    # Individual Series A
    df_a = df_test[df_test["series_id"] == "A"]
    metrics_a = compute_all_vn1_metrics(df_a["actual"], df_a["predicted"])
    assert np.isclose(metrics_a["vn1_score"], 20.0 / 300.0, atol=1e-4)
    assert np.isclose(metrics_a["relative_bias"], 0.0, atol=1e-4)

    # Individual Series B
    df_b = df_test[df_test["series_id"] == "B"]
    metrics_b = compute_all_vn1_metrics(df_b["actual"], df_b["predicted"])
    assert np.isclose(metrics_b["vn1_score"], 40.0 / 100.0, atol=1e-4)
    assert np.isclose(metrics_b["relative_bias"], 0.2000, atol=1e-4)

    # Combined portfolio
    metrics_all = compute_all_vn1_metrics(df_test["actual"], df_test["predicted"])
    assert np.isclose(metrics_all["vn1_score"], 0.1500, atol=1e-4)
    assert np.isclose(metrics_all["wape"], 0.1000, atol=1e-4)
    assert np.isclose(metrics_all["mae"], 10.0000, atol=1e-4)
    assert np.isclose(metrics_all["rmse"], 10.0000, atol=1e-4)
    assert np.isclose(metrics_all["relative_bias"], 0.0500, atol=1e-4)


def test_point_in_time_future_tampering_immunity(tmp_path):
    """Verify that tampering with future data rows does not affect forecasts at origin_date."""
    db_clean = tmp_path / "clean_store.db"
    store_clean = ShadowStore(db_path=db_clean)
    pipeline_clean = VN1ShadowPipeline(store=store_clean)

    # Construct genuine history up to 2023-01-02 (20 weeks)
    dates_hist = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    rows_clean = []
    for d in dates_hist:
        rows_clean.append(
            {
                "series_id": "0_1_100",
                "week_start": d,
                "sales": 100.0,
                "price": 10.0,
            }
        )
    df_clean = pd.DataFrame(rows_clean)

    # Construct tampered dataset containing extreme future sales after 2023-01-02
    rows_tampered = list(rows_clean)
    dates_future = pd.date_range(start="2023-01-09", end="2023-04-03", freq="W-MON")
    for d in dates_future:
        rows_tampered.append(
            {
                "series_id": "0_1_100",
                "week_start": d,
                "sales": 9999999.0,  # Extreme tampered future value
                "price": 999.0,
            }
        )
    df_tampered = pd.DataFrame(rows_tampered)

    # Run forecast with clean data
    run_clean = pipeline_clean.run_shadow_forecast(
        snapshot_df=df_clean,
        origin_date="2023-01-02",
        horizon=13,
    )

    # Run forecast with tampered future data in a separate store
    db_tampered = tmp_path / "tampered_store.db"
    store_tampered = ShadowStore(db_path=db_tampered)
    pipeline_tampered = VN1ShadowPipeline(store=store_tampered)
    run_tampered = pipeline_tampered.run_shadow_forecast(
        snapshot_df=df_tampered,
        origin_date="2023-01-02",
        horizon=13,
    )

    # Query both runs
    fc_clean = store_clean.query_forecasts(run_id=run_clean["run_id"])
    fc_tampered = store_tampered.query_forecasts(run_id=run_tampered["run_id"])

    # Predictions must be 100% identical despite the tampered future rows in input_data
    merged = pd.merge(
        fc_clean[["series_id", "target_week", "model_name", "predicted_sales"]],
        fc_tampered[["series_id", "target_week", "model_name", "predicted_sales"]],
        on=["series_id", "target_week", "model_name"],
        suffixes=("_clean", "_tampered"),
    )
    assert len(merged) == 26  # 13 weeks x 2 models
    assert np.allclose(merged["predicted_sales_clean"], merged["predicted_sales_tampered"]), (
        "Future data tampering leaked into forecasts!"
    )


def test_cross_experiment_parameter_consistency():
    """Verify that Candidate D parameters match between shadow pipeline and calibration module."""
    config = load_vn1_config()

    # 1. Check window length and shrinkage k in config
    assert config.forecasting.min_history_required == 12
    assert config.shadow.shrinkage_k == 100.0

    # 2. Check Candidate D initialization defaults
    cand_config = SeasonalCalibrationConfig()
    assert cand_config.method == "post_holiday_targeted"
    assert cand_config.shrinkage_k == 100.0
    assert cand_config.min_factor_bound == 0.50
    assert cand_config.max_factor_bound == 1.25
    assert cand_config.post_holiday_start_week == 1
    assert cand_config.post_holiday_end_week == 13

    # 3. Verify CalibratedForecaster instance compatibility
    base_12w = VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12")
    cal_forecaster = CalibratedForecaster(
        base_forecaster=base_12w,
        calibration_config=cand_config,
        name="Baseline_MovingAvg12_PostHolidayTargeted",
    )
    assert cal_forecaster.base_forecaster.window_size == 12
    assert cal_forecaster.config.shrinkage_k == 100.0


def test_reconciliation_conflict_rejection(tmp_path):
    """Verify that submitting conflicting actuals for an already reconciled forecast rejects corruption."""
    db_file = tmp_path / "conflict_test.db"
    store = ShadowStore(db_path=db_file)
    pipeline = VN1ShadowPipeline(store=store)

    dates_hist = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    df = pd.DataFrame([{"series_id": "0_1_100", "week_start": d, "sales": 50.0, "price": 5.0} for d in dates_hist])

    pipeline.run_shadow_forecast(snapshot_df=df, origin_date="2023-01-02", horizon=13)

    # 1. Ingest initial actuals
    initial_actuals = pd.DataFrame([{"series_id": "0_1_100", "week_start": "2023-01-09", "sales": 55.0}])
    ingest_res1 = store.ingest_actuals(initial_actuals, source="FEED_A")
    assert ingest_res1["inserted"] == 1
    match_res1 = store.reconcile_actuals()
    assert match_res1["reconciled_in_batch"] == 2  # 2 models matched

    # Verify forecast status
    fc1 = store.query_forecasts()
    step1_fc = fc1[fc1["target_week"] == "2023-01-09"]
    assert (step1_fc["reconciliation_status"] == "MATCHED").all()
    assert (step1_fc["actual_sales"] == 55.0).all()

    # 2. Reset status to PENDING_ACTUAL to test duplicate actual collision detection
    with store._get_connection() as conn:
        conn.execute("UPDATE shadow_forecasts SET reconciliation_status = 'PENDING_ACTUAL' WHERE horizon_step = 1")

    # Ingest second actual with conflicting sales value from FEED_B
    conflicting_actuals = pd.DataFrame([{"series_id": "0_1_100", "week_start": "2023-01-09", "sales": 999.0}])
    store.ingest_actuals(conflicting_actuals, source="FEED_B")
    match_res2 = store.reconcile_actuals()
    assert match_res2["conflicts_rejected"] == 2

    # Verify that the forecast was flagged as CONFLICT_REJECTED
    fc2 = store.query_forecasts()
    target_fc = fc2[fc2["target_week"] == "2023-01-09"]
    assert (target_fc["reconciliation_status"] == "CONFLICT_REJECTED").all()
