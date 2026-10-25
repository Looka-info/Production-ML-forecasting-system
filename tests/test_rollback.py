"""Tests for model rollback workflow and state persistence."""

from demandcast.config import RegistryConfig
from demandcast.registry import ModelRegistry


def test_model_rollback_lifecycle(trained_dummy_model, tmp_path):
    """Verify promotion of v1, promotion of v2, and clean rollback to v1."""
    cfg = RegistryConfig(
        releases_dir=str(tmp_path / "releases"),
        current_pointer=str(tmp_path / "current.json"),
    )
    registry = ModelRegistry(cfg)

    # 1. Package and promote v1.0.0
    registry.package_release(
        model=trained_dummy_model,
        version="v1.0.0",
        mlflow_run_id="run_v1",
        validation_metrics={"mae": 3.0, "wape": 0.10},
        training_date_range={"min_date": "2010-01-04", "max_date": "2010-08-01"},
    )
    registry.promote_release("v1.0.0", reason="Initial release")
    assert registry.get_current_release_info()["current_version"] == "v1.0.0"

    # 2. Package and promote v1.1.0
    registry.package_release(
        model=trained_dummy_model,
        version="v1.1.0",
        mlflow_run_id="run_v2",
        validation_metrics={"mae": 2.8, "wape": 0.09},
        training_date_range={"min_date": "2010-01-04", "max_date": "2010-09-01"},
    )
    registry.promote_release("v1.1.0", reason="Upgraded feature model")
    assert registry.get_current_release_info()["current_version"] == "v1.1.0"

    # 3. Rollback to v1.0.0
    registry.rollback_to("v1.0.0", reason="Degradation in v1.1.0 detected in monitoring")
    active_info = registry.get_current_release_info()
    assert active_info["current_version"] == "v1.0.0"
    assert "Rollback" in active_info["reason"]

    # 4. Load current model and verify it loads v1.0.0 successfully
    model, meta = registry.load_current_model()
    assert meta["version"] == "v1.0.0"
    assert model.is_fitted
