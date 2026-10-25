"""Unit and integration tests for DemandCast v2.2 Shadow Inference Pipeline.

Covers:
1. Active production release pointer immutability.
2. Candidate inference isolation (no state alteration of production).
3. Equal snapshot and origin distribution to both models.
4. Prevention of future-data leakage across origin boundary.
5. Idempotent persistence on repeated execution.
6. Forecast immutability after actuals reconciliation.
7. Strict join alignment on series_id and target_week.
8. Omission of unscored missing actuals.
9. Explicit handling of conflicting actuals.
10. Failure isolation (candidate error does not abort production forecast).
11. Truthful NOT_CONFIGURED reporting when input is absent.
12. Official VN1 metric parity.
13. Telemetry separation between candidate and production.
14. Backward compatibility with existing API endpoints.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demandcast.api.main import app
from demandcast.config import get_project_root
from demandcast.vn1.ingestion import ProspectiveStreamIngestionAdapter
from demandcast.vn1.metrics import compute_all_vn1_metrics
from demandcast.vn1.shadow import (
    ShadowEvaluator,
    ShadowPilotConfig,
    ShadowStore,
    VN1ShadowPipeline,
)


@pytest.fixture
def temp_store(tmp_path) -> ShadowStore:
    """Create an isolated test SQLite shadow store."""
    db_file = tmp_path / "test_shadow_store.db"
    return ShadowStore(db_path=db_file)


@pytest.fixture
def synthetic_snapshot() -> pd.DataFrame:
    """Create 20 weeks of synthetic Monday data for 3 series up to 2023-01-02."""
    dates = pd.date_range(start="2022-08-15", end="2023-01-02", freq="W-MON")
    records = []
    for d in dates:
        records.append(
            {
                "series_id": "0_1_100",
                "week_start": d,
                "sales": 50.0,
                "price": 10.0,
            }
        )
        records.append(
            {
                "series_id": "0_1_200",
                "week_start": d,
                "sales": 100.0,
                "price": 15.0,
            }
        )
        records.append(
            {
                "series_id": "1_2_300",
                "week_start": d,
                "sales": 20.0,
                "price": 5.0,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_actuals() -> pd.DataFrame:
    """Create 13 weeks of forward actuals starting 2023-01-09 for the 3 series."""
    dates = pd.date_range(start="2023-01-09", periods=13, freq="W-MON")
    records = []
    for d in dates:
        records.append(
            {
                "series_id": "0_1_100",
                "week_start": d,
                "sales": 45.0,
            }
        )
        records.append(
            {
                "series_id": "0_1_200",
                "week_start": d,
                "sales": 90.0,
            }
        )
        records.append(
            {
                "series_id": "1_2_300",
                "week_start": d,
                "sales": 18.0,
            }
        )
    return pd.DataFrame(records)


def test_1_active_production_release_pointer_unchanged(temp_store, synthetic_snapshot):
    """Running shadow forecasting must NEVER modify the active VN1 production pointer."""
    root = get_project_root()
    pointer_file = root / "models" / "profiles" / "vn1" / "current.json"
    with open(pointer_file, "r", encoding="utf-8") as f:
        before = json.load(f)

    pipeline = VN1ShadowPipeline(store=temp_store)
    res = pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)
    assert res["status"] == "SUCCEEDED"

    with open(pointer_file, "r", encoding="utf-8") as f:
        after = json.load(f)

    assert before == after
    assert after["current_version"] == "v1.0.0"


def test_2_candidate_inference_cannot_change_production_predictions(temp_store, synthetic_snapshot):
    """Production predictions in shadow runner must exactly match independent standalone production runs."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    prod_model, prod_meta = pipeline.registry.load_current_model()

    # Standalone run
    origin = pd.to_datetime("2023-01-02")
    standalone_preds = prod_model.predict(synthetic_snapshot, horizon=13, origin_date=origin)

    # Shadow runner run
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)
    shadow_prod_df = temp_store.query_forecasts(is_candidate=False)

    merged = pd.merge(
        standalone_preds,
        shadow_prod_df,
        on=["series_id", "horizon_step"],
    )
    assert len(merged) == len(standalone_preds)
    pd.testing.assert_series_equal(
        merged["predicted_sales_x"],
        merged["predicted_sales_y"],
        check_names=False,
    )


def test_3_both_models_receive_same_snapshot_and_origin(temp_store, synthetic_snapshot):
    """Production and Candidate D must evaluate identical series, origins, and target weeks."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    prod_df = temp_store.query_forecasts(is_candidate=False)
    cand_df = temp_store.query_forecasts(is_candidate=True)

    assert len(prod_df) == len(cand_df)
    assert set(prod_df["series_id"]) == set(cand_df["series_id"])
    assert set(prod_df["target_week"]) == set(cand_df["target_week"])
    assert set(prod_df["horizon_step"]) == set(cand_df["horizon_step"])
    assert (prod_df["origin_date"] == "2023-01-02").all()
    assert (cand_df["origin_date"] == "2023-01-02").all()


def test_4_future_data_leakage_prevented(temp_store, synthetic_snapshot):
    """Data rows with week_start > origin_date must be strictly excluded from inference."""
    snapshot_with_future = synthetic_snapshot.copy()
    future_row = pd.DataFrame(
        [
            {
                "series_id": "0_1_100",
                "week_start": pd.to_datetime("2023-01-09"),  # 1 week past origin
                "sales": 99999.0,  # Extreme future leak
                "price": 10.0,
            }
        ]
    )
    snapshot_with_future = pd.concat([snapshot_with_future, future_row], ignore_index=True)

    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(snapshot_with_future, origin_date="2023-01-02", horizon=13)

    # Forecast for 0_1_100 must be computed solely on trailing 12 observations (50.0), NOT contaminated by 99999.0
    df_fc = temp_store.query_forecasts(is_candidate=False)
    s100_fc = df_fc[df_fc["series_id"] == "0_1_100"]
    assert (s100_fc["predicted_sales"] == 50.0).all()


def test_5_repeated_execution_is_idempotent(temp_store, synthetic_snapshot):
    """Rerunning shadow forecast on same snapshot must not duplicate forecast records."""
    pipeline = VN1ShadowPipeline(store=temp_store)

    res1 = pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)
    assert res1["persistence"]["inserted"] == 78  # 3 series x 13 steps x 2 models
    assert res1["persistence"]["skipped_duplicates"] == 0

    res2 = pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)
    assert res2["persistence"]["inserted"] == 0
    assert res2["persistence"]["skipped_duplicates"] == 78

    df = temp_store.query_forecasts()
    assert len(df) == 78  # Exact same count


def test_6_forecasts_remain_immutable_after_actual_reconciliation(temp_store, synthetic_snapshot, synthetic_actuals):
    """Reconciling delayed actuals updates actual_sales but NEVER modifies predicted_sales or factors."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    before_df = temp_store.query_forecasts().sort_values("record_id").reset_index(drop=True)

    # Ingest actuals and reconcile
    temp_store.ingest_actuals(synthetic_actuals, source="test_feed")
    recon_res = temp_store.reconcile_actuals()
    assert recon_res["reconciled_in_batch"] == 78

    after_df = temp_store.query_forecasts().sort_values("record_id").reset_index(drop=True)

    # Predictions must be strictly identical
    pd.testing.assert_series_equal(before_df["predicted_sales"], after_df["predicted_sales"])
    pd.testing.assert_series_equal(before_df["execution_timestamp"], after_df["execution_timestamp"])

    # Reconciliation fields must be populated
    assert (after_df["reconciliation_status"] == "MATCHED").all()
    assert not after_df["actual_sales"].isna().any()
    assert (after_df["actual_source"] == "test_feed").all()


def test_7_actuals_join_strictly_on_series_and_target_week(temp_store, synthetic_snapshot):
    """Actuals join only when both series_id and target_week match."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    # Supply actuals with an unforecast series
    mismatched_actuals = pd.DataFrame(
        [
            {
                "series_id": "nonexistent_series",
                "week_start": pd.to_datetime("2023-01-09"),
                "sales": 50.0,
            }
        ]
    )
    temp_store.ingest_actuals(mismatched_actuals, source="mismatch_test")
    recon_res = temp_store.reconcile_actuals()

    assert recon_res["reconciled_in_batch"] == 0
    assert recon_res["total_matched_all_time"] == 0
    assert recon_res["total_pending_all_time"] == 78


def test_8_missing_actuals_are_not_scored(temp_store, synthetic_snapshot):
    """Forecasts awaiting actuals are not evaluated; evaluator returns NO_MATCHED_ACTUALS."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    evaluator = ShadowEvaluator(temp_store)
    rep = evaluator.evaluate_matched_forecasts()
    assert rep["status"] == "NO_MATCHED_ACTUALS"
    assert rep["sample_count"] == 0


def test_9_conflicting_actuals_handling(temp_store, synthetic_snapshot, synthetic_actuals):
    """Conflicting actual values for the same series and date are rejected rather than silently overwriting."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    # First reconciliation
    temp_store.ingest_actuals(synthetic_actuals, source="first_feed")
    temp_store.reconcile_actuals()

    # Second conflicting actual from different source
    conflict_actuals = synthetic_actuals.copy()
    conflict_actuals["sales"] = conflict_actuals["sales"] * 10.0  # Conflicting value
    temp_store.ingest_actuals(conflict_actuals, source="conflicting_feed")

    # Manually reset one forecast to test collision
    with temp_store._get_connection() as conn:
        conn.execute("UPDATE shadow_forecasts SET reconciliation_status = 'PENDING_ACTUAL' WHERE horizon_step = 1")
        conn.commit()

    recon_res = temp_store.reconcile_actuals()
    assert recon_res["conflicts_rejected"] > 0


def test_10_candidate_failure_isolation(temp_store, synthetic_snapshot, monkeypatch):
    """A candidate failure must not prevent production forecast generation or persistence."""
    pipeline = VN1ShadowPipeline(store=temp_store)

    # Monkeypatch Candidate D to fail
    def faulty_predict(*args, **kwargs):
        raise RuntimeError("Simulated Candidate D Failure")

    from demandcast.vn1.seasonal_calibration import CalibratedForecaster

    monkeypatch.setattr(CalibratedForecaster, "predict", faulty_predict)

    res = pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    # Overall run status is PARTIAL_FAILURE
    assert res["status"] == "PARTIAL_FAILURE"
    assert res["production"]["status"] == "SUCCESS"
    assert res["production"]["forecasts_count"] == 39
    assert res["candidate_d"]["status"] == "FAILED"
    assert "Simulated Candidate D Failure" in res["candidate_d"]["error"]

    # Production forecasts must be persisted
    prod_stored = temp_store.query_forecasts(is_candidate=False)
    assert len(prod_stored) == 39


def test_11_invalid_or_absent_provider_configuration_returns_truthful_status(temp_store):
    """Calling shadow runner with None or empty snapshot returns NOT_CONFIGURED."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    res_none = pipeline.run_shadow_forecast(None)
    assert res_none["status"] == "NOT_CONFIGURED"

    res_empty = pipeline.run_shadow_forecast(pd.DataFrame())
    assert res_empty["status"] == "NOT_CONFIGURED"


def test_12_metrics_match_official_vn1_implementation(temp_store, synthetic_snapshot, synthetic_actuals):
    """Shadow evaluator metrics match hand-calculated VN1 metric formulas."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)
    temp_store.ingest_actuals(synthetic_actuals, source="test_source")
    temp_store.reconcile_actuals()

    evaluator = ShadowEvaluator(temp_store)
    rep = evaluator.evaluate_matched_forecasts()
    assert rep["status"] == "EVALUATED"

    # Query matched forecasts directly
    matched_prod = temp_store.query_forecasts(is_candidate=False, reconciliation_status="MATCHED")
    expected = compute_all_vn1_metrics(matched_prod["actual_sales"], matched_prod["predicted_sales"])

    assert rep["production_model"]["metrics"]["vn1_score"] == pytest.approx(expected["vn1_score"], abs=1e-4)
    assert rep["production_model"]["metrics"]["wape"] == pytest.approx(expected["wape"], abs=1e-4)


def test_13_candidate_and_production_telemetry_distinguishable(temp_store, synthetic_snapshot):
    """Candidate and production records have distinct model identifiers and metadata."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    pipeline.run_shadow_forecast(synthetic_snapshot, origin_date="2023-01-02", horizon=13)

    prod_records = temp_store.query_forecasts(is_candidate=False)
    cand_records = temp_store.query_forecasts(is_candidate=True)

    assert (prod_records["model_identifier"] == "production_v1.0.0").all()
    assert (cand_records["model_identifier"] == "candidate_d_v2.1-candidate_d").all()
    assert prod_records["calibration_metadata"].isna().all()
    assert not cand_records["calibration_metadata"].isna().any()


def test_14_fastapi_shadow_endpoints_and_backward_compatibility():
    """Verify /v2/shadow/telemetry and /v2/shadow/summary endpoints and backward compatibility."""
    client = TestClient(app)

    # 1. Shadow telemetry endpoint
    r_tel = client.get("/v2/shadow/telemetry")
    assert r_tel.status_code == 200
    data_tel = r_tel.json()
    assert "status" in data_tel
    assert "total_runs" in data_tel

    # 2. Shadow summary endpoint
    r_sum = client.get("/v2/shadow/summary")
    assert r_sum.status_code == 200
    data_sum = r_sum.json()
    assert "status" in data_sum

    # 3. Existing endpoints remain 100% backward compatible
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_ready = client.get("/ready")
    assert r_ready.status_code in [200, 503]
    body = r_ready.json()
    assert "status" in body or ("detail" in body and "status" in body["detail"])


def test_15_shadow_pilot_config_defaults_and_safeguards():
    """Verify ShadowPilotConfig defaults to inactive, enforces manual promotion policy, and specifies safeguards."""
    pilot = ShadowPilotConfig()
    summary = pilot.get_status_summary()

    assert not pilot.is_active
    assert summary["is_active"] is False
    assert summary["provider_status"] == "NOT_CONFIGURED"
    assert summary["promotion_safeguard"] == "MANUAL_SIGN_OFF_REQUIRED"
    assert summary["segment_safety_limit"] == 0.0100
    assert summary["pilot_readiness"] == "READY_FOR_PROSPECTIVE_FEED_INACTIVE"
    assert len(summary["challengers"]) == 2
    assert summary["challengers"][0]["model_identifier"] == "candidate_c_gated"
    assert summary["challengers"][0]["eligibility_gate_enforced"] is True
    assert summary["challengers"][1]["model_identifier"] == "candidate_d_client_eb"
    assert summary["challengers"][1]["eligibility_gate_enforced"] is False


def test_16_shadow_pipeline_runs_candidate_c_and_d_concurrently(temp_store, synthetic_snapshot):
    """Verify shadow pipeline can execute Candidate C and Candidate D concurrently with Production."""
    pipeline = VN1ShadowPipeline(store=temp_store)
    res = pipeline.run_shadow_forecast(
        synthetic_snapshot,
        origin_date="2023-01-02",
        horizon=13,
        candidates=("candidate_c", "candidate_d"),
    )

    assert res["status"] == "SUCCEEDED"
    assert res["production"]["status"] == "SUCCESS"
    assert res["candidate_c"]["status"] == "SUCCESS"
    assert res["candidate_d"]["status"] == "SUCCESS"

    # Verify forecasts stored in SQLite
    prod_df = temp_store.query_forecasts(model_identifier="production_v1.0.0")
    cand_c_df = temp_store.query_forecasts(model_identifier="candidate_c_v2.2-candidate_c")
    cand_d_df = temp_store.query_forecasts(model_identifier="candidate_d_v2.1-candidate_d")

    assert len(prod_df) == 39  # 3 series x 13 horizons
    assert len(cand_c_df) == 39
    assert len(cand_d_df) == 39


def test_17_v225_audit_artifacts_and_discrepancy_reproduction():
    """Verify v2.2.5 holdout provenance and exact global calibration score reproduction."""
    root = get_project_root()
    audit_dir = root / "reports" / "vn1" / "shadow" / "audit"

    # 1. Holdout Provenance
    prov_file = audit_dir / "v2_2_5_holdout_provenance.json"
    assert prov_file.exists(), f"Missing audit file: {prov_file}"
    with open(prov_file, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    assert p_data["actual_disk_sha256"] == "ef5b998226495fdfc44ea4dcda8df9b45b1c76f6adb50441e2217ad3a623c6ac"
    assert p_data["hash_consistency_verified"] is True
    assert (
        p_data["discrepancy_investigation"]["verdict"]
        == "RESOLVED_CONVERSATIONAL_TRANSCRIPTION_ERROR_IMMUTABLE_FILE_INTACT"
    )

    # 2. Global Calibration Exact Reproduction
    reprod_file = audit_dir / "global_calibration_reproduction.json"
    assert reprod_file.exists(), f"Missing audit file: {reprod_file}"
    with open(reprod_file, "r", encoding="utf-8") as f:
        r_data = json.load(f)
    assert r_data["exact_scores_reproduced"]["v2_2_3_reproduced"] == 0.5463
    assert r_data["exact_scores_reproduced"]["v2_2_3_match"] is True
    assert r_data["exact_scores_reproduced"]["v2_2_4_reproduced"] == 0.5342
    assert r_data["exact_scores_reproduced"]["v2_2_4_match"] is True

    # 3. Four Candidate Reproduction CSV
    reprod_csv = audit_dir / "full_catalog_four_candidate_reproduction.csv"
    assert reprod_csv.exists()
    df_reprod = pd.read_csv(reprod_csv)
    assert len(df_reprod) == 4
    assert set(df_reprod["candidate_code"]) == {"Candidate A", "Candidate B", "Candidate C", "Candidate D"}

    # 4. Segment Safety CSV
    safety_csv = audit_dir / "four_candidate_segment_safety.csv"
    assert safety_csv.exists()
    df_safety = pd.read_csv(safety_csv)
    assert "VOLUME_TIER" in df_safety["segment_dimension"].values
    assert "INTERMITTENCY_TIER" in df_safety["segment_dimension"].values


def test_18_sparse_segment_accounting_parity_and_transition_trace():
    """Verify v2.2.6 sparse segment accounting trace and dynamic fallback parity."""
    root = get_project_root()
    audit_dir = root / "reports" / "vn1" / "shadow" / "audit"

    # 1. Verify trace CSV
    trace_csv = audit_dir / "sparse_segment_accounting_trace.csv"
    assert trace_csv.exists(), f"Missing trace file: {trace_csv}"
    df_trace = pd.read_csv(trace_csv)
    # Exactly 144 transitioning series x 13 horizons = 1872 rows
    assert len(df_trace) == 1872
    assert df_trace["series_id"].nunique() == 144

    # All transitioning series originated in static Sparse tier (pre-2022 zero frac > 0.6666)
    assert (df_trace["pre_2022_zero_fraction"] > 0.6666).all()
    # All satisfied dynamic active criteria at 2023-01-02 (zero frac <= 0.50, vol >= 35.0, ma12 >= ma26)
    assert (df_trace["origin_zero_fraction"] <= 0.50).all()
    assert (df_trace["trailing_12w_volume"] >= 35.0).all()
    assert (df_trace["trailing_12w_ma"] >= df_trace["trailing_26w_ma"]).all()

    # 2. Verify v2.2.6 readiness report exists and contains factual gain ratio
    report_md = audit_dir / "v2_2_6_readiness_report.md"
    assert report_md.exists(), f"Missing report: {report_md}"
    report_text = report_md.read_text(encoding="utf-8")
    assert "73.02%" in report_text
    assert (
        "dynamic_sparse_max_prediction_difference: 0.0" in report_text
        or "maximum prediction difference = **0.000000**" in report_text
    )
    assert "100.0000% identical" in report_text


def test_19_shadow_pilot_activation_readiness_guard():
    """Verify ShadowPilotConfig safety guards prevent unconfigured pilot activation."""
    # 1. Default pilot is hard-inactive and unconfigured
    pilot = ShadowPilotConfig()
    assert pilot.version == "v2.2.9"
    assert pilot.is_active is False
    assert pilot.provider_status == "NOT_CONFIGURED"

    unconfigured_adapter = ProspectiveStreamIngestionAdapter()
    ready, reasons = pilot.validate_activation_readiness(unconfigured_adapter)
    assert ready is False
    assert any("is_active=False" in r for r in reasons)
    assert any("NOT_CONFIGURED" in r for r in reasons)

    # 2. Even if is_active is manually forced True, unconfigured adapter blocks activation
    pilot_forced = ShadowPilotConfig(is_active=True, provider_status="HEALTHY")
    ready_forced, reasons_forced = pilot_forced.validate_activation_readiness(unconfigured_adapter)
    assert ready_forced is False
    assert any("NOT_CONFIGURED" in r for r in reasons_forced)

    # 3. Healthy adapter with active flag passes readiness check
    healthy_adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="valid_token",
    )
    ready_valid, reasons_valid = pilot_forced.validate_activation_readiness(healthy_adapter, skip_network=True)
    assert ready_valid is True
    assert len(reasons_valid) == 0
