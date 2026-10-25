"""Tests for model release packaging, checksum verification, and registry operations."""


from demandcast.config import RegistryConfig
from demandcast.registry import ModelRegistry


def test_registry_package_and_verify(trained_dummy_model, tmp_path):
    """Verify release packaging, metadata creation, checksum matching, and promotion."""
    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases"),
        current_pointer=str(tmp_path / "current.json"),
    )
    registry = ModelRegistry(cfg)

    # Package release v0.1.0
    rel_dir = registry.package_release(
        model=trained_dummy_model,
        version="v0.1.0",
        mlflow_run_id="run_123",
        validation_metrics={"mae": 2.5, "wape": 0.08},
        training_date_range={"min_date": "2010-01-04", "max_date": "2010-10-01"},
    )

    assert rel_dir.exists()
    assert (rel_dir / "model.joblib").exists()
    assert (rel_dir / "metadata.json").exists()

    # Verify release
    is_valid, msg = registry.verify_release("v0.1.0")
    assert is_valid, msg

    # Promote release
    registry.promote_release("v0.1.0", reason="Initial test promotion")
    info = registry.get_current_release_info()
    assert info is not None
    assert info["current_version"] == "v0.1.0"

    # Test tampering detection (corrupt artifact)
    with open(rel_dir / "model.joblib", "ab") as f:
        f.write(b"corrupted_bytes")

    is_valid_tampered, tamper_msg = registry.verify_release("v0.1.0")
    assert not is_valid_tampered
    assert "Checksum mismatch" in tamper_msg


def test_registry_package_with_quality_alert_and_freshness(trained_dummy_model, tmp_path):
    """Verify registry attaches quality alert status and freshness when WAPE exceeds threshold."""
    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases_alert"),
        current_pointer=str(tmp_path / "current_alert.json"),
    )
    registry = ModelRegistry(cfg)

    # Candidate with WAPE = 55.16% (> 50%)
    rel_dir = registry.package_release(
        model=trained_dummy_model,
        version="v1.0.0-alert",
        mlflow_run_id="run_alert",
        validation_metrics={"mae": 255.94, "wape": 0.5516},
        training_date_range={"min_date": "2009-11-30", "max_date": "2011-11-07"},
    )

    import json
    with open(rel_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["quality_status"] == "ALERT"
    assert "Validation WAPE (55.16%) exceeds target threshold" in meta["quality_alert"]
    assert meta["model_freshness"]["status"] == "stale_historical_dataset"
    assert meta["model_freshness"]["is_calibrated_for_modern_retail"] is False

    # Promote under audited override
    registry.promote_release(
        "v1.0.0-alert",
        reason="Audited override promotion",
        allow_alert_override=True,
        quality_status="ALERT",
        quality_alert=meta["quality_alert"],
    )
    current = registry.get_current_release_info()
    assert current["quality_status"] == "ALERT"
    assert current["model_freshness_status"] == "stale_historical_dataset"


def test_promotion_denied_when_wape_exceeds_threshold_without_override(trained_dummy_model, tmp_path):
    """Verify that promote_release denies promotion by default if validation WAPE > threshold."""
    import pytest

    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases_gate"),
        current_pointer=str(tmp_path / "current_gate.json"),
    )
    registry = ModelRegistry(cfg)

    registry.package_release(
        model=trained_dummy_model,
        version="v1.0.0-failing",
        mlflow_run_id="run_failing",
        validation_metrics={"mae": 400.0, "wape": 0.65},  # 65% > 50%
        training_date_range={"min_date": "2009-11-30", "max_date": "2011-11-07"},
    )

    with pytest.raises(ValueError, match="Promotion blocked: Candidate validation WAPE"):
        registry.promote_release("v1.0.0-failing", reason="Attempt without override", allow_alert_override=False)


def test_promotion_denied_when_override_missing_reason(trained_dummy_model, tmp_path):
    """Verify that promote_release rejects override promotion if an explicit reason string is missing."""
    import pytest

    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases_gate2"),
        current_pointer=str(tmp_path / "current_gate2.json"),
    )
    registry = ModelRegistry(cfg)

    registry.package_release(
        model=trained_dummy_model,
        version="v1.0.0-failing2",
        mlflow_run_id="run_failing2",
        validation_metrics={"mae": 400.0, "wape": 0.65},
        training_date_range={"min_date": "2009-11-30", "max_date": "2011-11-07"},
    )

    with pytest.raises(ValueError, match="explicit non-empty audit reason string is required"):
        registry.promote_release("v1.0.0-failing2", reason="", allow_alert_override=True)


def test_promotion_denied_on_checksum_failure(trained_dummy_model, tmp_path):
    """Verify that promote_release rejects promotion when artifact checksum is corrupted."""
    import pytest

    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases_checksum"),
        current_pointer=str(tmp_path / "current_checksum.json"),
    )
    registry = ModelRegistry(cfg)

    rel_dir = registry.package_release(
        model=trained_dummy_model,
        version="v1.0.0-corrupt",
        mlflow_run_id="run_corrupt",
        validation_metrics={"mae": 100.0, "wape": 0.10},
        training_date_range={"min_date": "2009-11-30", "max_date": "2011-11-07"},
    )

    # Corrupt model artifact
    with open(rel_dir / "model.joblib", "wb") as f:
        f.write(b"bad_bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        registry.promote_release("v1.0.0-corrupt", reason="Should fail")
