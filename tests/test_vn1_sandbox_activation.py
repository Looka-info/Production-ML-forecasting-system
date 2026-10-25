"""Comprehensive integration and safety audit tests for DemandCast v2.2.8.

Verifies:
1. Strict HTTPS boundary and loopback restriction.
2. Complete plain HTTP rejection in production environments (DEMANDCAST_ENV=prod/production).
3. Secret redaction, URL sanitization, and credential whitespace handling.
4. Upstream authentication failures (401/403) and timeout retry exhaustion.
5. Incomplete weekly bucket rejection (is_bucket_complete=False fail-closed).
6. Deterministic snapshot hash idempotency.
7. The 6-gate activation architecture in ShadowPilotConfig.
8. CLI prospective live mode fails closed on failed activation gates.
9. End-to-end 3-model prospective shadow forecast, persistence, and reconciliation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from scripts.run_vn1_shadow import cmd_forecast


@pytest.fixture
def isolated_store(tmp_path) -> ShadowStore:
    """Isolated SQLite shadow store for testing."""
    db_file = tmp_path / "test_sandbox_activation.db"
    return ShadowStore(db_path=db_file)


@pytest.fixture
def sample_valid_payload() -> dict:
    """Canonical v1.0 Monday sales batch with completed accounting buckets."""
    return {
        "schema_version": "v1.0",
        "provider_id": "erp_sandbox_test",
        "records": [
            {
                "series_id": "0_1_100",
                "week_start": "2023-01-02",
                "sales": 55.0,
                "price": 12.0,
                "is_bucket_complete": True,
            },
            {
                "series_id": "0_1_200",
                "week_start": "2023-01-02",
                "sales": 80.0,
                "price": 14.5,
                "is_bucket_complete": True,
            },
        ],
    }


def test_01_production_mode_http_rejection(monkeypatch):
    """Plain HTTP is strictly forbidden in production mode, even for localhost."""
    monkeypatch.setenv("DEMANDCAST_ENV", "production")

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="http://localhost:8080/sales",
        auth_token="valid_token",
        allow_insecure_http=True,
    )
    assert adapter.endpoint_valid is False
    assert adapter.get_status() == "FAILED"
    assert "production environment" in adapter.endpoint_error.lower()

    # Also verify 'prod' shorthand
    monkeypatch.setenv("DEMANDCAST_ENV", "prod")
    adapter_prod = ProspectiveStreamIngestionAdapter(
        endpoint_url="http://127.0.0.1:8080/sales",
        auth_token="valid_token",
        allow_insecure_http=True,
    )
    assert adapter_prod.endpoint_valid is False
    assert adapter_prod.get_status() == "FAILED"


def test_02_remote_http_rejection_even_with_allow_insecure_http(monkeypatch):
    """allow_insecure_http cannot bypass HTTPS enforcement on remote endpoints."""
    monkeypatch.delenv("DEMANDCAST_ENV", raising=False)

    remote_adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="http://erp-sandbox.external-corp.internal/sales",
        auth_token="valid_token",
        allow_insecure_http=True,
    )
    assert remote_adapter.endpoint_valid is False
    assert remote_adapter.get_status() == "FAILED"
    assert "remote endpoints must use https" in remote_adapter.endpoint_error.lower()


def test_03_credential_whitespace_stripping_and_empty_handling():
    """Credentials with surrounding whitespace are stripped; empty strings are treated as None."""
    # 1. Whitespace stripped from token
    adapter1 = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="   secret_bearer_token   \t\n",
    )
    assert adapter1._auth_token == "secret_bearer_token"
    assert adapter1.get_status() == "HEALTHY"

    # 2. Whitespace stripped from api_key
    adapter2 = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        api_key="   secret_api_key   \n",
    )
    assert adapter2._api_key == "secret_api_key"
    assert adapter2.get_status() == "HEALTHY"

    # 3. All whitespace treated as None
    adapter3 = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="    ",
    )
    assert adapter3._auth_token is None
    assert adapter3.get_status() == "NOT_CONFIGURED"

    # 4. secret_getter stripping
    adapter4 = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        secret_getter=lambda k: "   secret_from_vault   " if "TOKEN" in k or "KEY" in k else None,
    )
    assert adapter4._auth_token == "secret_from_vault"
    assert adapter4.get_status() == "HEALTHY"


def test_04_secret_redaction_and_url_masking():
    """Sensitive query params, passwords, and tokens are scrubbed from telemetry and error strings."""
    secret_token = "ultra_sensitive_jwt_token_99999"
    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://user:password123@erp.corporate.internal/api/v1/sales?api_key=secret99&region=us",
        auth_token=secret_token,
    )

    # 1. URL masking removes user:password and query strings
    masked = adapter._mask_url(adapter.endpoint_url)
    assert "password123" not in masked
    assert "api_key=secret99" not in masked
    assert "user@" not in masked
    assert masked == "https://erp.corporate.internal/api/v1/sales"

    # 2. Error message sanitization strips token and query strings
    raw_error = f"HTTP 500 from https://user:password123@erp.corporate.internal/api/v1/sales?api_key=secret99 with token {secret_token}"
    sanitized = adapter._sanitize_error_message(raw_error)
    assert secret_token not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert "password123" not in sanitized


def test_05_remote_auth_failure_handling(isolated_store, sample_valid_payload):
    """HTTP 401/403 responses return AUTHENTICATION_FAILED without leaking secrets."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_session.get.return_value = mock_resp

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="revoked_or_invalid_secret_token",
    )

    res = adapter.run_preflight(
        store=isolated_store,
        sample_payload=sample_valid_payload,
        http_session=mock_session,
    )
    assert res.status == "AUTHENTICATION_FAILED"
    assert res.network_reachable is True
    assert res.remote_authenticated is False
    assert any("rejected authentication" in r.lower() for r in res.reasons)
    assert "revoked_or_invalid_secret_token" not in str(res.reasons)


def test_06_provider_timeout_and_retry_limits():
    """pull_remote_batch retries transient errors and safely returns PROVIDER_UNAVAILABLE on exhaustion."""
    mock_session = MagicMock()
    mock_session.get.side_effect = TimeoutError("Connection to host timed out after 10000ms")

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_secret",
        timeout_seconds=0.1,
    )

    res = adapter.pull_remote_batch(http_session=mock_session, max_retries=2)
    assert res.status == "PROVIDER_UNAVAILABLE"
    assert mock_session.get.call_count == 3  # initial + 2 retries
    assert adapter.total_rejected_batches == 1
    assert "timed out" in res.errors[0].lower()


def test_07_incomplete_weekly_buckets_fail_closed(isolated_store):
    """Incomplete accounting buckets (is_bucket_complete=False) are rejected fail-closed."""
    incomplete_payload = {
        "schema_version": "v1.0",
        "records": [
            {
                "series_id": "0_1_100",
                "week_start": "2023-01-02",
                "sales": 40.0,
                "is_bucket_complete": False,  # Partial accounting bucket!
            }
        ],
    }

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_secret",
    )

    # Ingestion fails
    ingest_res = adapter.ingest(incomplete_payload)
    assert ingest_res.status == "REJECTED_INCOMPLETE"
    assert any("is_bucket_complete=False" in e for e in ingest_res.errors)

    # Preflight fails
    pf_res = adapter.run_preflight(
        store=isolated_store,
        sample_payload=incomplete_payload,
        skip_network=True,
    )
    assert pf_res.status == "INCOMPLETE_INPUT"
    assert pf_res.data_completeness is False


def test_08_snapshot_hash_idempotency_on_duplicate_ingestion():
    """Snapshot hashing is deterministic regardless of row insertion order."""
    payload_a = {
        "schema_version": "v1.0",
        "records": [
            {"series_id": "0_1_100", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True},
            {"series_id": "0_1_200", "week_start": "2023-01-02", "sales": 20.0, "is_bucket_complete": True},
        ],
    }
    payload_b = {
        "schema_version": "v1.0",
        "records": [
            {"series_id": "0_1_200", "week_start": "2023-01-02", "sales": 20.0, "is_bucket_complete": True},
            {"series_id": "0_1_100", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True},
        ],
    }

    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_secret",
    )

    res_a = adapter.ingest(payload_a)
    res_b = adapter.ingest(payload_b)

    assert res_a.status == "SUCCESS"
    assert res_b.status == "SUCCESS"
    assert res_a.snapshot_hash == res_b.snapshot_hash
    assert len(res_a.snapshot_hash) == 64


def test_09_all_six_activation_gates(isolated_store, sample_valid_payload):
    """Verify each of the 6 activation gates behaves strictly and fails closed."""
    # Base unconfigured pilot
    pilot = ShadowPilotConfig()

    # Gate 1 Failure: is_active is False
    eval1 = pilot.evaluate_activation_gates()
    assert eval1["all_gates_passed"] is False
    assert eval1["gates"]["gate_1_operator_authorization"] is False
    assert any("Gate 1" in r for r in eval1["reasons"])

    # Gate 2 Failure: Ingestion adapter is NOT_CONFIGURED
    pilot_active = ShadowPilotConfig(is_active=True, provider_status="HEALTHY")
    unconfigured_adapter = ProspectiveStreamIngestionAdapter()
    eval2 = pilot_active.evaluate_activation_gates(adapter=unconfigured_adapter)
    assert eval2["all_gates_passed"] is False
    assert eval2["gates"]["gate_2_provider_configuration"] is False

    # Gate 3 Failure: Preflight status is not READY
    failing_pf = PreflightResult(
        status="PROVIDER_UNAVAILABLE",
        provider_id="mock",
        schema_version="v1.0",
        provider_configured=True,
    )
    eval3 = pilot_active.evaluate_activation_gates(
        adapter=ProspectiveStreamIngestionAdapter(
            endpoint_url="https://mock-retail-feed.internal/sales",
            auth_token="token",
        ),
        preflight_result=failing_pf,
    )
    assert eval3["all_gates_passed"] is False
    assert eval3["gates"]["gate_3_remote_auth_and_preflight"] is False

    # Gate 4 Failure: Incomplete weekly records
    incomplete_payload = {
        "records": [{"series_id": "0_1_100", "week_start": "2023-01-02", "sales": 1.0, "is_bucket_complete": False}]
    }
    adapter_healthy = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="token",
    )
    ready_pf = PreflightResult(
        status="READY",
        provider_id="mock",
        schema_version="v1.0",
        provider_configured=True,
    )
    eval4 = pilot_active.evaluate_activation_gates(
        adapter=adapter_healthy,
        preflight_result=ready_pf,
        sample_payload=incomplete_payload,
        store=isolated_store,
    )
    assert eval4["all_gates_passed"] is False
    assert eval4["gates"]["gate_4_data_completeness"] is False

    # Gate 5 Failure: Store is broken
    broken_store = MagicMock()
    broken_store.get_telemetry.side_effect = RuntimeError("Database disk corruption")
    eval5 = pilot_active.evaluate_activation_gates(
        adapter=adapter_healthy,
        preflight_result=ready_pf,
        sample_payload=sample_valid_payload,
        store=broken_store,
    )
    assert eval5["all_gates_passed"] is False
    assert eval5["gates"]["gate_5_persistence_integrity"] is False

    # Gate 6 Failure: Shutdown procedure not tested
    eval6 = pilot_active.evaluate_activation_gates(
        adapter=adapter_healthy,
        preflight_result=ready_pf,
        sample_payload=sample_valid_payload,
        store=isolated_store,
        shutdown_procedure_tested=False,
    )
    assert eval6["all_gates_passed"] is False
    assert eval6["gates"]["gate_6_shutdown_procedure_verified"] is False

    # All 6 Gates Pass
    eval_all_pass = pilot_active.evaluate_activation_gates(
        adapter=adapter_healthy,
        preflight_result=ready_pf,
        sample_payload=sample_valid_payload,
        store=isolated_store,
        shutdown_procedure_tested=True,
    )
    assert eval_all_pass["all_gates_passed"] is True
    assert all(eval_all_pass["gates"].values())
    assert len(eval_all_pass["reasons"]) == 0


def test_10_prospective_live_mode_blocks_when_gates_fail(tmp_path):
    """Running scripts/run_vn1_shadow.py with --mode prospective_live terminates if gates fail."""
    dummy_snapshot = tmp_path / "snapshot.parquet"
    df = pd.DataFrame(
        [
            {"series_id": "0_1_100", "week_start": "2023-01-02", "sales": 10.0},
        ]
    )
    df.to_parquet(dummy_snapshot)

    args = argparse.Namespace(
        snapshot_path=str(dummy_snapshot),
        origin_date="2023-01-02",
        horizon=13,
        candidates="candidate_c,candidate_d",
        mode="prospective_live",
        provider="erp_sandbox_unauthorized",
        db_path=str(tmp_path / "shadow.db"),
    )

    exit_code = cmd_forecast(args)
    assert exit_code == 1


def test_11_end_to_end_3_model_prospective_shadow_forecast_persistence_and_reconciliation(isolated_store):
    """End-to-end prospective shadow execution with 3 models, persistence, and actuals reconciliation."""
    # 1. Prepare 26 historical weeks for 2 series (up to 2023-01-02)
    dates_hist = pd.date_range("2022-07-11", "2023-01-02", freq="W-MON")
    hist_records = []
    for d in dates_hist:
        hist_records.append({"series_id": "0_1_100", "week_start": d.strftime("%Y-%m-%d"), "sales": 45.0})
        hist_records.append({"series_id": "0_1_200", "week_start": d.strftime("%Y-%m-%d"), "sales": 80.0})
    df_hist = pd.DataFrame(hist_records)

    # 2. Run shadow forecast with Production, Candidate C, and Candidate D
    pipeline = VN1ShadowPipeline(store=isolated_store)
    fc_res = pipeline.run_shadow_forecast(
        df_hist,
        origin_date="2023-01-02",
        horizon=13,
        candidates=("candidate_c", "candidate_d"),
        mode="historical_replay",
        provider="erp_wms_sandbox_fixture",
    )

    assert fc_res["status"] == "SUCCEEDED"
    assert fc_res["production"]["status"] == "SUCCESS"
    assert fc_res["candidate_c"]["status"] == "SUCCESS"
    assert fc_res["candidate_d"]["status"] == "SUCCESS"

    # Total forecasts: 2 series x 13 steps x 3 models = 78
    telemetry = isolated_store.get_telemetry()
    assert telemetry["total_forecasts"] == 78
    assert telemetry["awaiting_actuals_forecasts"] == 78
    assert telemetry["matched_forecasts"] == 0

    # 3. Ingest future actuals (13 future weeks starting 2023-01-09)
    future_dates = pd.date_range("2023-01-09", periods=13, freq="W-MON")
    actuals = []
    for d in future_dates:
        actuals.append({"series_id": "0_1_100", "week_start": d.strftime("%Y-%m-%d"), "sales": 44.0})
        actuals.append({"series_id": "0_1_200", "week_start": d.strftime("%Y-%m-%d"), "sales": 78.0})
    df_actuals = pd.DataFrame(actuals)

    ingest_res = isolated_store.ingest_actuals(df_actuals, source="erp_future_actuals_feed")
    assert ingest_res["inserted"] == 26

    reconcile_res = isolated_store.reconcile_actuals()
    assert reconcile_res["reconciled_in_batch"] == 78
    assert reconcile_res["conflicts_rejected"] == 0

    # 4. Evaluate matched forecasts
    evaluator = ShadowEvaluator(store=isolated_store)
    eval_rep = evaluator.evaluate_matched_forecasts(run_id=fc_res["run_id"])
    assert eval_rep["status"] == "EVALUATED"
    assert eval_rep["series_count"] == 2
    assert "production_model" in eval_rep
    assert "candidate_d" in eval_rep
    assert "comparison_delta" in eval_rep
