"""Tests for VN1 isolated model registry, packaging, checksum verification, and governance gates."""

import json

import pytest

from demandcast.vn1.baselines import VN1MovingAverageForecaster
from demandcast.vn1.config import VN1Config
from demandcast.vn1.registry import VN1ModelRegistry


@pytest.fixture
def dummy_vn1_model():
    """Create a dummy VN1 moving average model."""
    forecaster = VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12_Official")
    forecaster.supported_series = ["0_1_PROD1"]
    forecaster.feature_cols = []
    return forecaster


@pytest.fixture
def temp_vn1_registry(tmp_path):
    """Create a VN1ModelRegistry with temporary directories."""
    cfg = VN1Config()
    cfg.registry.releases_dir = str(tmp_path / "vn1" / "releases")
    cfg.registry.current_pointer = str(tmp_path / "vn1" / "current.json")
    return VN1ModelRegistry(config=cfg)


def test_vn1_registry_package_and_verify(dummy_vn1_model, temp_vn1_registry, tmp_path):
    """Verify VN1 release packaging, checksum generation, verification, and tamper detection."""
    rel_dir = temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.0.0",
        mlflow_run_id="run_vn1_test",
        validation_metrics={"vn1_score": 0.4325, "wape": 0.4276, "mae": 207.24},
        notes="Official 12-week benchmark",
    )

    assert rel_dir.exists()
    assert (rel_dir / "model.joblib").exists()
    assert (rel_dir / "metadata.json").exists()

    is_valid, msg = temp_vn1_registry.verify_release("v1.0.0")
    assert is_valid, msg

    # Tamper detection: modify artifact
    with open(rel_dir / "model.joblib", "ab") as f:
        f.write(b"tampered_data")

    is_valid_tampered, tamper_msg = temp_vn1_registry.verify_release("v1.0.0")
    assert not is_valid_tampered
    assert "Checksum mismatch" in tamper_msg


def test_vn1_registry_promotion_and_load(dummy_vn1_model, temp_vn1_registry):
    """Verify atomic promotion and loading of active VN1 model bundle."""
    temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.0.0",
        mlflow_run_id="run_vn1_promo",
        validation_metrics={"vn1_score": 0.4325, "wape": 0.4276},
    )

    temp_vn1_registry.promote_release("v1.0.0", reason="Phase 1 best benchmark")
    info = temp_vn1_registry.get_current_release_info()
    assert info is not None
    assert info["current_version"] == "v1.0.0"
    assert info["quality_status"] == "HEALTHY"

    loaded_model, meta = temp_vn1_registry.load_current_model()
    assert loaded_model.name == "Baseline_MovingAvg12_Official"
    assert meta["version"] == "v1.0.0"


def test_vn1_registry_quality_gate_enforcement(dummy_vn1_model, temp_vn1_registry):
    """Verify that models with VN1 score > 0.70 require explicit override and audit reason."""
    rel_dir = temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.0.0-alert",
        mlflow_run_id="run_vn1_alert",
        validation_metrics={"vn1_score": 0.8500, "wape": 0.6500},
    )

    with open(rel_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["quality_status"] == "ALERT"
    assert "exceeds target threshold" in meta["quality_alert"]

    # Blocked without override
    with pytest.raises(ValueError, match="Promotion blocked: Candidate validation VN1 score"):
        temp_vn1_registry.promote_release("v1.0.0-alert")

    # Blocked without audit reason
    with pytest.raises(ValueError, match="non-empty audit reason"):
        temp_vn1_registry.promote_release("v1.0.0-alert", allow_alert_override=True, reason="")

    # Allowed with audit override
    temp_vn1_registry.promote_release(
        "v1.0.0-alert",
        allow_alert_override=True,
        reason="Audited benchmark override for research comparison",
    )
    current = temp_vn1_registry.get_current_release_info()
    assert current["current_version"] == "v1.0.0-alert"
    assert current["quality_status"] == "ALERT"


def test_vn1_registry_rollback(dummy_vn1_model, temp_vn1_registry):
    """Verify rollback restores previous valid release pointer."""
    temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.0.0",
        mlflow_run_id="run_v1",
        validation_metrics={"vn1_score": 0.45},
    )
    temp_vn1_registry.promote_release("v1.0.0", reason="First release")

    temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.1.0",
        mlflow_run_id="run_v2",
        validation_metrics={"vn1_score": 0.42},
    )
    temp_vn1_registry.promote_release("v1.1.0", reason="Second release")

    assert temp_vn1_registry.get_current_release_info()["current_version"] == "v1.1.0"

    temp_vn1_registry.rollback_to("v1.0.0", reason="Reverting due to downstream issues")
    assert temp_vn1_registry.get_current_release_info()["current_version"] == "v1.0.0"


def test_vn1_registry_profile_isolation(dummy_vn1_model, temp_vn1_registry):
    """Verify VN1 registry operates in isolated directory without modifying root UCI registry."""
    from demandcast.config import get_project_root

    root = get_project_root()
    uci_pointer = root / "models" / "current.json"

    # Save existing mtime of UCI pointer if it exists
    original_mtime = uci_pointer.stat().st_mtime if uci_pointer.exists() else None

    temp_vn1_registry.package_release(
        model=dummy_vn1_model,
        version="v1.0.0",
        mlflow_run_id="run_iso",
        validation_metrics={"vn1_score": 0.4325},
    )
    temp_vn1_registry.promote_release("v1.0.0", reason="Isolation check")

    if uci_pointer.exists():
        assert uci_pointer.stat().st_mtime == original_mtime
