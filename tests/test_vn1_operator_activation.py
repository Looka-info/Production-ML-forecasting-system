"""Comprehensive unit and integration tests for DemandCast v2.2.9 Operator Activation.

Verifies:
1. Operator activation pathways (environment variable, config file, CLI flag, constructor).
2. Default inactive state (is_active=False) per governance lock.
3. Multi-gate fail-closed enforcement: is_active=True alone cannot activate without all gates passing.
4. Preflight isolation: preflight READY never activates the pilot automatically.
5. Production model immutability (VN1 v1.0.0, UCI v1.0.0, zero promotion).
6. Missing credentials truthful reporting (NOT_CONFIGURED).
7. Prospective live inference CLI execution safeguards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demandcast.config import get_project_root
from demandcast.vn1.ingestion import (
    PreflightResult,
    ProspectiveStreamIngestionAdapter,
)
from demandcast.vn1.shadow import (
    ShadowPilotConfig,
    ShadowStore,
)
from scripts.run_vn1_shadow import cmd_forecast


@pytest.fixture
def isolated_store(tmp_path) -> ShadowStore:
    """Isolated SQLite shadow store for testing."""
    db_file = tmp_path / "test_operator_activation.db"
    return ShadowStore(db_path=db_file)


@pytest.fixture
def valid_monday_payload() -> dict:
    """Canonical v1.0 payload with complete weekly accounting buckets."""
    return {
        "schema_version": "v1.0",
        "provider_id": "erp_operator_test",
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


def test_01_default_pilot_config_is_inactive():
    """Default ShadowPilotConfig is strictly locked inactive per governance."""
    pilot = ShadowPilotConfig()
    assert pilot.version == "v2.2.9"
    assert pilot.is_active is False
    assert pilot.activation_mechanism == "DEFAULT_INACTIVE"
    assert pilot.provider_id == "NOT_CONFIGURED"
    assert pilot.provider_status == "NOT_CONFIGURED"
    assert pilot.promotion_policy == "MANUAL_SIGN_OFF_REQUIRED"


def test_02_environment_variable_activation_path(monkeypatch):
    """Setting DEMANDCAST_SHADOW_PILOT_ACTIVE=true activates Gate 1."""
    # 1. Active env var
    monkeypatch.setenv("DEMANDCAST_SHADOW_PILOT_ACTIVE", "true")
    monkeypatch.setenv("DEMANDCAST_OPERATOR_SIGN_OFF_BY", "lead_mlops_engineer")

    pilot = ShadowPilotConfig()
    assert pilot.is_active is True
    assert pilot.activation_mechanism == "ENVIRONMENT_VARIABLE"
    assert pilot.operator_sign_off_by == "lead_mlops_engineer"
    assert pilot.operator_sign_off_date is not None

    # 2. Inactive env var variations
    for val in ("false", "0", "no", "DISABLED", ""):
        monkeypatch.setenv("DEMANDCAST_SHADOW_PILOT_ACTIVE", val)
        pilot_inactive = ShadowPilotConfig()
        assert pilot_inactive.is_active is False
        assert pilot_inactive.activation_mechanism == "DEFAULT_INACTIVE"


def test_03_cli_and_from_operator_environment_path(tmp_path, monkeypatch):
    """from_operator_environment correctly checks CLI override and config file."""
    monkeypatch.delenv("DEMANDCAST_SHADOW_PILOT_ACTIVE", raising=False)

    # 1. Explicit CLI override
    pilot_cli = ShadowPilotConfig.from_operator_environment(
        authorize_override=True,
        operator_name="senior_platform_engineer",
    )
    assert pilot_cli.is_active is True
    assert pilot_cli.activation_mechanism == "CLI_OPERATOR_SIGN_OFF"
    assert pilot_cli.operator_sign_off_by == "senior_platform_engineer"

    # 2. Config file path
    cfg_file = tmp_path / "pilot_signoff.yaml"
    cfg_file.write_text(
        "is_active: true\noperator_sign_off_by: test_operator\noperator_sign_off_date: '2026-10-26T00:00:00Z'\n",
        encoding="utf-8",
    )
    pilot_cfg = ShadowPilotConfig.from_operator_environment(config_path=cfg_file)
    assert pilot_cfg.is_active is True
    assert pilot_cfg.activation_mechanism == "OPERATOR_CONFIG_FILE"
    assert pilot_cfg.operator_sign_off_by == "test_operator"


def test_04_operator_activation_alone_does_not_permit_inference_when_gates_fail(isolated_store):
    """Even if an operator sets is_active=True, missing provider/preflight blocks activation."""
    pilot = ShadowPilotConfig(is_active=True, operator_sign_off_by="operator_signoff")

    # Ingestion adapter unconfigured
    adapter = ProspectiveStreamIngestionAdapter()
    ready, reasons = pilot.validate_activation_readiness(adapter=adapter, store=isolated_store)

    assert ready is False
    assert any("Gate 2" in r for r in reasons)
    assert any("Gate 3" in r for r in reasons)


def test_05_preflight_ready_never_activates_pilot_automatically(isolated_store, valid_monday_payload):
    """Preflight returning status READY must never alter pilot.is_active or auto-activate."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = valid_monday_payload
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
    assert res.pilot_activation_ready is False  # Strictly locked

    pilot = ShadowPilotConfig()
    assert pilot.is_active is False


def test_06_all_six_gates_pass_when_authorized_and_verified(isolated_store, valid_monday_payload):
    """When both operator authorization and verified provider checks are present, all gates pass."""
    pilot_auth = ShadowPilotConfig(
        is_active=True,
        provider_status="HEALTHY",
        operator_sign_off_by="lead_mlops",
    )
    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/sales",
        auth_token="valid_production_secret",
    )
    pf_ready = PreflightResult(
        status="READY",
        provider_id="mock",
        schema_version="v1.0",
        provider_configured=True,
        endpoint_url_masked="https://mock-retail-feed.internal/sales",
        endpoint_valid=True,
        endpoint_scheme="https",
        credentials_present=True,
        network_reachable=True,
        remote_authenticated=True,
        schema_valid=True,
        data_completeness=True,
        persistence_ready=True,
    )

    eval_res = pilot_auth.evaluate_activation_gates(
        adapter=adapter,
        preflight_result=pf_ready,
        sample_payload=valid_monday_payload,
        store=isolated_store,
        shutdown_procedure_tested=True,
    )
    assert eval_res["all_gates_passed"] is True
    assert all(eval_res["gates"].values())
    assert len(eval_res["reasons"]) == 0


def test_07_production_pointers_remain_untouched():
    """Verify that VN1 and UCI active production profiles remain pinned at v1.0.0."""
    root = get_project_root()

    vn1_curr = json.loads((root / "models" / "profiles" / "vn1" / "current.json").read_text(encoding="utf-8"))
    assert vn1_curr["current_version"] == "v1.0.0"

    uci_curr = json.loads((root / "models" / "current.json").read_text(encoding="utf-8"))
    assert uci_curr["current_version"] == "v1.0.0"


def test_08_cli_prospective_live_mode_blocks_without_complete_gates(tmp_path):
    """CLI forecast command fails closed with exit code 1 if activation gates fail."""
    dummy_snapshot = tmp_path / "snapshot.parquet"
    df = pd.DataFrame([{"series_id": "0_1_100", "week_start": "2023-01-02", "sales": 10.0}])
    df.to_parquet(dummy_snapshot)

    # Calling with --authorize-pilot but without configured adapter or database must fail closed
    args = argparse.Namespace(
        snapshot_path=str(dummy_snapshot),
        origin_date="2023-01-02",
        horizon=13,
        candidates="candidate_c,candidate_d",
        mode="prospective_live",
        provider="erp_sandbox_test",
        authorize_pilot=True,
        operator_sign_off="auditor",
        pilot_config=None,
        db_path=str(tmp_path / "shadow.db"),
    )

    exit_code = cmd_forecast(args)
    assert exit_code == 1
