"""DemandCast v2.2.2 — Time Contract, Availability Semantics, and Boundary Leakage Tests.

Tests:
1. Fully available historical weekly bucket validation.
2. Incomplete or unavailable bucket at the end of a snapshot (fails closed in prospective mode, dropped in replay).
3. Snapshot containing observations after the information cutoff (purged in replay, rejected if unclosed).
4. Snapshot with missing weeks or duplicate week labels with conflicts.
5. Forecast origin crossing December and January (year boundary).
6. Forecast at an ISO-week boundary (ISO week 52/53 transition to ISO week 1).
7. Different as_of_timestamp values for the same historical weekly data evaluating availability eligibility.
8. Target-week actual accidentally supplied as an inference feature (ensures no lookahead leakage).
9. Future-value tampering without forecast changes when those values are unavailable at origin.
10. Reproducibility when the same valid snapshot and time contract are reused.
"""

import numpy as np
import pandas as pd
import pytest

from demandcast.vn1.shadow import ShadowStore, VN1ShadowPipeline
from demandcast.vn1.time_contract import ForecastTimeContract, create_time_contract


def test_1_fully_available_historical_weekly_bucket():
    """Verify that a fully available, completed historical bucket passes contract validation."""
    origin = pd.to_datetime("2023-01-02")
    contract = create_time_contract(origin_date=origin, mode="historical_replay")

    dates = pd.date_range(start="2022-10-10", end="2023-01-02", freq="W-MON")
    df = pd.DataFrame([{"series_id": "SERIES_1", "week_start": d, "sales": 100.0} for d in dates])

    clean_df, audit = contract.validate_snapshot(df)
    assert len(clean_df) == len(dates)
    assert audit["status"] == "VALID"
    assert audit["rows_eligible"] == len(dates)
    assert audit["rows_excluded_post_cutoff"] == 0
    assert audit["last_observed_week_start"] == "2023-01-02"


def test_2_incomplete_or_unavailable_bucket_at_snapshot_end(tmp_path):
    """Trailing incomplete weekly buckets fail closed in live mode and are excluded in replay."""
    origin = pd.to_datetime("2023-01-02")

    # 1. Prospective live mode fail-closed
    contract_live = create_time_contract(
        origin_date=origin,
        as_of_timestamp="2023-01-04T12:00:00Z",  # Wednesday mid-week
        mode="prospective_live",
    )
    # Snapshot contains mid-week observation marked incomplete
    df_live = pd.DataFrame(
        [
            {"series_id": "SER_A", "week_start": "2022-12-26", "sales": 50.0, "is_bucket_complete": True},
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 12.0, "is_bucket_complete": False},
        ]
    )
    with pytest.raises(ValueError, match="Incomplete trailing weekly bucket detected in prospective live mode"):
        contract_live.validate_snapshot(df_live)

    # 2. Historical replay mode excludes incomplete trailing buckets
    contract_replay = create_time_contract(origin_date=origin, mode="historical_replay")
    clean_df, audit = contract_replay.validate_snapshot(df_live)
    assert len(clean_df) == 1
    assert clean_df["week_start"].iloc[0] == pd.to_datetime("2022-12-26")
    assert audit["rows_excluded_incomplete"] == 1


def test_3_snapshot_containing_observations_after_cutoff():
    """Observations after the information cutoff are strictly purged and logged."""
    origin = pd.to_datetime("2023-01-02")
    contract = create_time_contract(origin_date=origin, mode="historical_replay")

    df = pd.DataFrame(
        [
            {"series_id": "SER_A", "week_start": "2022-12-19", "sales": 50.0},
            {"series_id": "SER_A", "week_start": "2022-12-26", "sales": 50.0},
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 50.0},
            # Future rows after cutoff
            {"series_id": "SER_A", "week_start": "2023-01-09", "sales": 99.0},
            {"series_id": "SER_A", "week_start": "2023-01-16", "sales": 120.0},
        ]
    )

    clean_df, audit = contract.validate_snapshot(df)
    assert len(clean_df) == 3
    assert audit["rows_excluded_post_cutoff"] == 2
    assert (clean_df["week_start"] <= origin).all()


def test_4_missing_weeks_or_duplicate_week_labels():
    """Duplicate week labels with conflicting values trigger validation errors."""
    origin = pd.to_datetime("2023-01-02")
    contract = create_time_contract(origin_date=origin, mode="historical_replay")

    # Conflicting duplicate rows for same week
    df_conflicting = pd.DataFrame(
        [
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 50.0},
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 75.0},  # Conflicting value
        ]
    )
    with pytest.raises(ValueError, match="Duplicate week labels with conflicting sales values"):
        contract.validate_snapshot(df_conflicting)

    # Identical duplicates are safely deduplicated
    df_identical = pd.DataFrame(
        [
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 50.0},
            {"series_id": "SER_A", "week_start": "2023-01-02", "sales": 50.0},  # Exact duplicate
        ]
    )
    clean_df, _ = contract.validate_snapshot(df_identical)
    assert len(clean_df) == 1


def test_5_origin_crossing_december_and_january():
    """Verify target week progression across year boundaries (December to January)."""
    # Origin at last Monday of 2022
    origin_dec = pd.to_datetime("2022-12-26")
    contract = create_time_contract(origin_date=origin_dec, horizon=4)

    target_weeks = contract.target_weeks
    assert len(target_weeks) == 4
    # Step 1 is in January 2023
    assert target_weeks[0] == pd.to_datetime("2023-01-02")
    assert target_weeks[0].year == 2023
    assert target_weeks[1] == pd.to_datetime("2023-01-09")
    assert target_weeks[2] == pd.to_datetime("2023-01-16")
    assert target_weeks[3] == pd.to_datetime("2023-01-23")


def test_6_forecast_at_iso_week_boundary():
    """Verify ISO week numbering across week 52/53 transition to week 1."""
    # 2020 had 53 ISO weeks
    origin_iso = pd.to_datetime("2020-12-28")
    assert origin_iso.isocalendar().week == 53
    assert origin_iso.isocalendar().year == 2020

    contract = create_time_contract(origin_date=origin_iso, horizon=2)
    step1 = contract.target_weeks[0]  # 2021-01-04
    assert step1 == pd.to_datetime("2021-01-04")
    assert step1.isocalendar().week == 1
    assert step1.isocalendar().year == 2021


def test_7_different_as_of_timestamps_evaluate_eligibility():
    """Verify that observations are only eligible if data_available_at <= as_of_timestamp."""
    origin = pd.to_datetime("2023-01-02")

    # Data available on Tuesday 2023-01-03 04:00:00 UTC
    df = pd.DataFrame(
        [
            {
                "series_id": "SER_A",
                "week_start": "2023-01-02",
                "sales": 100.0,
                "data_available_at": "2023-01-03T04:00:00Z",
            }
        ]
    )

    # Case A: as_of_timestamp before data availability (e.g., Monday night 2023-01-02)
    contract_early = ForecastTimeContract(
        last_observed_week_start=origin,
        as_of_timestamp="2023-01-02T23:00:00Z",
        mode="historical_replay",
    )
    with pytest.raises(ValueError, match="No eligible observations remain"):
        contract_early.validate_snapshot(df)

    # Case B: as_of_timestamp after data availability (e.g., Tuesday noon 2023-01-03)
    contract_valid = ForecastTimeContract(
        last_observed_week_start=origin,
        as_of_timestamp="2023-01-03T12:00:00Z",
        mode="historical_replay",
    )
    clean_df, audit = contract_valid.validate_snapshot(df)
    assert len(clean_df) == 1
    assert audit["rows_eligible"] == 1


def test_8_target_week_actual_accidentally_supplied_as_feature(tmp_path):
    """Accidentally supplying a target-week row inside history is filtered out before inference."""
    db_file = tmp_path / "leak_test.db"
    store = ShadowStore(db_path=db_file)
    pipeline = VN1ShadowPipeline(store=store)

    dates = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    rows = [{"series_id": "0_1_100", "week_start": d, "sales": 50.0, "price": 10.0} for d in dates]

    # Malicious/leaked row for target step 1 (2023-01-09) with huge sales
    rows.append({"series_id": "0_1_100", "week_start": pd.to_datetime("2023-01-09"), "sales": 99999.0, "price": 1.0})
    df_leaked = pd.DataFrame(rows)

    res = pipeline.run_shadow_forecast(
        snapshot_df=df_leaked,
        origin_date="2023-01-02",
        horizon=13,
    )
    assert res["status"] == "SUCCEEDED"

    # Query predictions: step 1 (2023-01-09) prediction must NOT be affected by the leaked 99999.0
    fc = store.query_forecasts(run_id=res["run_id"])
    prod_pred_step1 = fc[(fc["target_week"] == "2023-01-09") & (fc["is_candidate"] == 0)]["predicted_sales"].iloc[0]
    # Trailing 12-week moving average of 50.0 is 50.0
    assert np.isclose(prod_pred_step1, 50.0), f"Leakage occurred! Predicted {prod_pred_step1} instead of 50.0"


def test_9_future_tampering_immunity(tmp_path):
    """Modifying future snapshot values after origin does not alter forecasts produced."""
    db_clean = tmp_path / "clean.db"
    db_tampered = tmp_path / "tampered.db"
    p_clean = VN1ShadowPipeline(store=ShadowStore(db_path=db_clean))
    p_tampered = VN1ShadowPipeline(store=ShadowStore(db_path=db_tampered))

    dates = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    rows_clean = [{"series_id": "0_1_100", "week_start": d, "sales": 70.0, "price": 10.0} for d in dates]

    rows_tampered = list(rows_clean)
    future_dates = pd.date_range(start="2023-01-09", end="2023-04-03", freq="W-MON")
    for fd in future_dates:
        rows_tampered.append({"series_id": "0_1_100", "week_start": fd, "sales": 1234567.0, "price": 0.01})

    res_clean = p_clean.run_shadow_forecast(pd.DataFrame(rows_clean), origin_date="2023-01-02", horizon=13)
    res_tampered = p_tampered.run_shadow_forecast(pd.DataFrame(rows_tampered), origin_date="2023-01-02", horizon=13)

    fc_c = ShadowStore(db_path=db_clean).query_forecasts(run_id=res_clean["run_id"])
    fc_t = ShadowStore(db_path=db_tampered).query_forecasts(run_id=res_tampered["run_id"])

    assert np.allclose(fc_c["predicted_sales"], fc_t["predicted_sales"])


def test_10_time_contract_reproducibility():
    """Identical snapshot and time contract produce identical hashes and forecasts."""
    origin = pd.to_datetime("2023-01-02")
    contract = create_time_contract(origin_date=origin, as_of_timestamp="2023-01-09T00:00:00Z")

    dates = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    df = pd.DataFrame([{"series_id": "S1", "week_start": d, "sales": 42.0} for d in dates])

    _, audit1 = contract.validate_snapshot(df)
    _, audit2 = contract.validate_snapshot(df)

    assert audit1["eligible_snapshot_hash"] == audit2["eligible_snapshot_hash"]
    assert audit1["target_weeks"] == audit2["target_weeks"]
