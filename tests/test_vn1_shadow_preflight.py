"""Unit and integration tests for DemandCast v2.2.7 Shadow Preflight and Operator Readiness.

Verifies:
1. Provider preflight diagnostics distinguishing all 7 states:
   NOT_CONFIGURED, INVALID_CONFIGURATION, AUTHENTICATION_FAILED,
   PROVIDER_UNAVAILABLE, SCHEMA_INVALID, INCOMPLETE_INPUT, and READY.
2. Secret redaction from telemetry, exception strings, URLs, and reports.
3. Network reachability, authentication semantics, and retry limits.
4. Independent candidate failure isolation (Candidates C and D fail independently without affecting Production).
5. End-to-end prospective replay, persistence, and actuals reconciliation on labeled test fixtures.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from demandcast.vn1.ingestion import (
    PreflightResult,
    ProspectiveStreamIngestionAdapter,
)
from demandcast.vn1.shadow import (
    ShadowEvaluator,
    ShadowPilotConfig,
    ShadowStore,
    VN1ShadowPipeline,
)


@pytest.fixture
def isolated_store(tmp_path) -> ShadowStore:
    """Isolated SQLite shadow store for testing."""
    db_file = tmp_path / "test_preflight_store.db"
    return ShadowStore(db_path=db_file)


@pytest.fixture
def labeled_test_payload() -> dict:
    """Labeled synthetic Monday sales payload for preflight schema testing."""
    return {
        "schema_version": "v1.0",
        "records": [
            {
                "series_id": "0_1_100",
                "week_start": "2023-01-02",
                "sales": 50.0,
                "price": 10.0,
                "is_bucket_complete": True,
            },
            {
                "series_id": "0_1_200",
                "week_start": "2023-01-02",
                "sales": 75.0,
                "price": 15.0,
                "is_bucket_complete": True,
            },
        ],
    }


def test_01_preflight_missing_credentials_returns_not_configured(monkeypatch, isolated_store):
    """When no endpoint or token is set, preflight returns NOT_CONFIGURED."""
    monkeypatch.delenv("DEMANDCAST_STREAM_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("DEMANDCAST_STREAM_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DEMANDCAST_STREAM_API_KEY", raising=False)

    adapter = ProspectiveStreamIngestionAdapter()
    res: PreflightResult = adapter.run_preflight(store=isolated_store)

    assert res.status == "NOT_CONFIGURED"
    assert res.provider_configured is False
    assert res.credentials_present is False
    assert res.persistence_ready is True
    assert res.pilot_activation_ready is False
    assert any("not configured" in r for r in res.reasons)
    assert any("DEMANDCAST_STREAM_ENDPOINT_URL" in step for step in res.remediation_steps)


def test_02_preflight_invalid_endpoint_scheme_returns_invalid_configuration(isolated_store):
    """Insecure HTTP or missing token returns INVALID_CONFIGURATION."""
    # 1. Plain HTTP without allow_insecure_http
    adapter_insecure = ProspectiveStreamIngestionAdapter(
        endpoint_url="http://insecure-remote.retail.internal/feed",
        auth_token="valid_token_secret",
    )
    res_insecure = adapter_insecure.run_preflight(store=isolated_store)
    assert res_insecure.status == "INVALID_CONFIGURATION"
    assert res_insecure.endpoint_valid is False
    assert any("HTTPS" in r for r in res_insecure.reasons)

    # 2. Configured endpoint but missing credentials
    adapter_no_auth = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://secure-feed.internal/stream",
        auth_token=None,
    )
    res_no_auth = adapter_no_auth.run_preflight(store=isolated_store)
    assert res_no_auth.status == "INVALID_CONFIGURATION"
    assert res_no_auth.credentials_present is False
    assert any("Authentication token is missing" in r for r in res_no_auth.reasons)


def test_03_preflight_remote_auth_failure_returns_authentication_failed(isolated_store):
    """When remote server returns HTTP 401 or 403, preflight reports AUTHENTICATION_FAILED with secret redacted."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_session.get.return_value = mock_resp

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="SUPER_SECRET_RETAIL_TOKEN_XYZ",
    )

    res = adapter.run_preflight(store=isolated_store, http_session=mock_session)
    assert res.status == "AUTHENTICATION_FAILED"
    assert res.network_reachable is True
    assert res.remote_authenticated is False
    assert any("HTTP 401" in r for r in res.reasons)

    # Verify secret is not in reasons or details
    res_str = str(res.to_dict())
    assert "SUPER_SECRET_RETAIL_TOKEN_XYZ" not in res_str


def test_04_preflight_network_timeout_returns_provider_unavailable(isolated_store):
    """When connection times out or fails, preflight reports PROVIDER_UNAVAILABLE with redacted URL query params."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectTimeout(
        "Connection timed out to https://mock-retail-feed.internal/sales?token=SUPER_SECRET_TOKEN"
    )

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales?token=SUPER_SECRET_TOKEN",
        auth_token="SUPER_SECRET_TOKEN",
    )

    res = adapter.run_preflight(store=isolated_store, http_session=mock_session)
    assert res.status == "PROVIDER_UNAVAILABLE"
    assert res.network_reachable is False
    assert res.remote_authenticated is False

    # Check secret redaction from exception output
    res_str = str(res.to_dict())
    assert "SUPER_SECRET_TOKEN" not in res_str


def test_05_preflight_malformed_schema_returns_schema_invalid(isolated_store):
    """Payload schema violation reports SCHEMA_INVALID."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"schema_version": "v999.0_invalid", "records": []}
    mock_session.get.return_value = mock_resp

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="test_token",
    )

    res = adapter.run_preflight(store=isolated_store, http_session=mock_session)
    assert res.status == "SCHEMA_INVALID"
    assert res.schema_valid is False
    assert any("Schema version mismatch" in r for r in res.reasons)


def test_06_preflight_incomplete_weekly_bucket_returns_incomplete_input(isolated_store):
    """Payload containing is_bucket_complete=False reports INCOMPLETE_INPUT."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "schema_version": "v1.0",
        "records": [
            {
                "series_id": "0_1_100",
                "week_start": "2023-01-02",
                "sales": 10.0,
                "is_bucket_complete": False,  # Incomplete weekly bucket!
            }
        ],
    }
    mock_session.get.return_value = mock_resp

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="test_token",
    )

    res = adapter.run_preflight(store=isolated_store, http_session=mock_session)
    assert res.status == "INCOMPLETE_INPUT"
    assert res.schema_valid is True
    assert res.data_completeness is False
    assert any("Incomplete weekly accounting buckets" in r for r in res.reasons)


def test_07_preflight_temporal_leakage_returns_incomplete_input(isolated_store):
    """Weekly bucket end exceeding as_of_timestamp reports INCOMPLETE_INPUT."""
    # Jan 02 bucket ends Jan 08 23:59:59. If as_of is Jan 05, bucket is not closed yet!
    sample_payload = {
        "schema_version": "v1.0",
        "records": [
            {
                "series_id": "0_1_100",
                "week_start": "2023-01-02",
                "sales": 10.0,
                "is_bucket_complete": True,
            }
        ],
    }

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="test_token",
    )

    res = adapter.run_preflight(
        store=isolated_store,
        sample_payload=sample_payload,
        skip_network=True,
        as_of_timestamp="2023-01-05T12:00:00Z",
    )
    assert res.status == "INCOMPLETE_INPUT"
    assert any("temporal leakage violation" in r for r in res.reasons)


def test_08_preflight_all_checks_pass_returns_ready_but_remains_inactive(isolated_store, labeled_test_payload):
    """When all checks pass, status is READY, but pilot activation remains strictly locked to False."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = labeled_test_payload
    mock_session.get.return_value = mock_resp

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_production_secret",
    )

    res = adapter.run_preflight(
        store=isolated_store,
        http_session=mock_session,
        as_of_timestamp="2023-01-16T00:00:00Z",
    )
    assert res.status == "READY"
    assert res.provider_configured is True
    assert res.network_reachable is True
    assert res.remote_authenticated is True
    assert res.schema_valid is True
    assert res.data_completeness is True
    assert res.persistence_ready is True

    # Critical governance assertion: A successful preflight NEVER activates the pilot
    assert res.pilot_activation_ready is False
    pilot = ShadowPilotConfig()
    assert pilot.is_active is False
    assert pilot.version == "v2.2.7"
    assert pilot.active_production_model == "Baseline_MovingAvg12_Official"


def test_09_pull_remote_batch_retry_and_success(labeled_test_payload):
    """pull_remote_batch retries on transient 503 and succeeds on 200."""
    mock_session = MagicMock()
    resp_503 = MagicMock()
    resp_503.status_code = 503

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = labeled_test_payload

    # First call returns 503, second returns 200
    mock_session.get.side_effect = [resp_503, resp_200]

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_token",
    )

    ingest_res = adapter.pull_remote_batch(
        as_of_timestamp="2023-01-16T00:00:00Z",
        http_session=mock_session,
        max_retries=1,
    )
    assert ingest_res.status == "SUCCESS"
    assert ingest_res.record_count == 2
    assert adapter.total_ingested_batches == 1


def test_10_independent_candidate_c_failure_isolation(isolated_store):
    """Failure inside Candidate C does not crash Production or Candidate D forecasts."""
    # Synthetic 20 weeks of history for 2 series
    dates = pd.date_range("2022-08-15", "2023-01-02", freq="W-MON")
    records = []
    for d in dates:
        records.append({"series_id": "0_1_100", "week_start": d, "sales": 50.0})
        records.append({"series_id": "0_1_200", "week_start": d, "sales": 80.0})
    df_snapshot = pd.DataFrame(records)

    pipeline = VN1ShadowPipeline(store=isolated_store)

    # Patch PreOriginCalibrationGate to raise an unhandled exception
    with patch("demandcast.vn1.shadow.PreOriginCalibrationGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.evaluate_series_eligibility.side_effect = RuntimeError("Candidate C memory allocation error")
        mock_gate_cls.return_value = mock_gate

        res = pipeline.run_shadow_forecast(
            df_snapshot,
            origin_date="2023-01-02",
            horizon=13,
            candidates=("candidate_c", "candidate_d"),
        )

    # Overall run status is PARTIAL_FAILURE, but Production and Candidate D succeeded
    assert res["status"] == "PARTIAL_FAILURE"
    assert res["production"]["status"] == "SUCCESS"
    assert res["candidate_d"]["status"] == "SUCCESS"
    assert res["candidate_c"]["status"] == "FAILED"
    assert "Candidate C memory allocation error" in res["candidate_c"]["error"]

    # Verify Production and Candidate D forecasts are saved in the store
    prod_saved = isolated_store.query_forecasts(model_identifier="production_v1.0.0")
    cand_d_saved = isolated_store.query_forecasts(model_identifier="candidate_d_v2.1-candidate_d")
    cand_c_saved = isolated_store.query_forecasts(model_identifier="candidate_c_v2.2-candidate_c")

    assert len(prod_saved) == 26  # 2 series x 13 horizons
    assert len(cand_d_saved) == 26
    assert len(cand_c_saved) == 0  # Failed model persisted 0 forecasts


def test_11_independent_candidate_d_failure_isolation(isolated_store):
    """Failure inside Candidate D does not crash Production forecasts."""
    dates = pd.date_range("2022-08-15", "2023-01-02", freq="W-MON")
    records = []
    for d in dates:
        records.append({"series_id": "0_1_100", "week_start": d, "sales": 50.0})
    df_snapshot = pd.DataFrame(records)

    pipeline = VN1ShadowPipeline(store=isolated_store)

    with patch("demandcast.vn1.shadow.CalibratedForecaster") as mock_cal_cls:
        mock_cal = MagicMock()
        mock_cal.predict.side_effect = ValueError("Candidate D parameter defect")
        mock_cal_cls.return_value = mock_cal

        res = pipeline.run_shadow_forecast(
            df_snapshot,
            origin_date="2023-01-02",
            horizon=13,
            candidates=("candidate_d",),
        )

    assert res["status"] == "PARTIAL_FAILURE"
    assert res["production"]["status"] == "SUCCESS"
    assert res["candidate_d"]["status"] == "FAILED"
    assert "Candidate D parameter defect" in res["candidate_d"]["error"]

    prod_saved = isolated_store.query_forecasts(model_identifier="production_v1.0.0")
    assert len(prod_saved) == 13


def test_12_end_to_end_replay_inference_persistence_and_reconciliation(isolated_store):
    """Complete end-to-end lifecycle on labeled test fixtures: ingestion, 3-model inference, persistence, and reconciliation."""
    # 1. Prepare labeled test snapshot (25 weeks of Monday data for 2 series)
    dates_hist = pd.date_range("2022-07-11", "2023-01-02", freq="W-MON")
    hist_records = []
    for d in dates_hist:
        hist_records.append({"series_id": "0_1_100", "week_start": d.strftime("%Y-%m-%d"), "sales": 40.0})
        hist_records.append({"series_id": "0_1_200", "week_start": d.strftime("%Y-%m-%d"), "sales": 70.0})
    df_hist = pd.DataFrame(hist_records)

    # 2. Run shadow forecast with Production, Candidate C, and Candidate D
    pipeline = VN1ShadowPipeline(store=isolated_store)
    fc_res = pipeline.run_shadow_forecast(
        df_hist,
        origin_date="2023-01-02",
        horizon=13,
        candidates=("candidate_c", "candidate_d"),
        mode="historical_replay",
        provider="labeled_unit_test_fixture",
    )

    assert fc_res["status"] == "SUCCEEDED"
    assert fc_res["production"]["status"] == "SUCCESS"
    assert fc_res["candidate_c"]["status"] == "SUCCESS"
    assert fc_res["candidate_d"]["status"] == "SUCCESS"

    # Total 2 series x 13 steps x 3 models = 78 forecasts
    telem = isolated_store.get_telemetry()
    assert telem["total_forecasts"] == 78
    assert telem["awaiting_actuals_forecasts"] == 78
    assert telem["matched_forecasts"] == 0

    # 3. Ingest forward test actuals (13 future weeks starting 2023-01-09)
    future_dates = pd.date_range("2023-01-09", periods=13, freq="W-MON")
    actuals_records = []
    for d in future_dates:
        actuals_records.append({"series_id": "0_1_100", "week_start": d.strftime("%Y-%m-%d"), "sales": 42.0})
        actuals_records.append({"series_id": "0_1_200", "week_start": d.strftime("%Y-%m-%d"), "sales": 68.0})
    df_actuals = pd.DataFrame(actuals_records)

    ingest_res = isolated_store.ingest_actuals(df_actuals, source="labeled_test_actuals_fixture")
    assert ingest_res["inserted"] == 26

    rec_res = isolated_store.reconcile_actuals()
    assert rec_res["reconciled_in_batch"] == 78
    assert rec_res["conflicts_rejected"] == 0
    assert rec_res["total_matched_all_time"] == 78
    assert rec_res["total_pending_all_time"] == 0

    # 4. Evaluate matched forecasts with ShadowEvaluator
    evaluator = ShadowEvaluator(store=isolated_store)
    eval_rep = evaluator.evaluate_matched_forecasts(run_id=fc_res["run_id"])

    assert eval_rep["status"] == "EVALUATED"
    assert eval_rep["series_count"] == 2
    assert eval_rep["production_model"]["metrics"]["vn1_score"] is not None
    assert eval_rep["candidate_d"]["metrics"]["vn1_score"] is not None
    assert eval_rep["comparison_delta"]["delta_vn1_score"] is not None
