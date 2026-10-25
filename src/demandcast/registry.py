"""Model registry, version release management, promotion, and rollback for DemandCast."""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demandcast.config import RegistryConfig, get_project_root, load_config

logger = logging.getLogger(__name__)


def calculate_file_sha256(filepath: Path) -> str:
    """Calculate SHA-256 checksum of an artifact."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_forecaster_artifact(model_file: Path) -> Any:
    """Load serialized model artifact dynamically supporting both baselines and ML models."""
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_file}")

    import joblib

    bundle = joblib.load(model_file)
    if isinstance(bundle, dict):
        model_type = bundle.get("model_type", bundle.get("name", ""))
        if "MovingAverage" in model_type or "Baseline_MovingAvg" in model_type:
            from demandcast.baselines import MovingAverageForecaster

            return MovingAverageForecaster.load(model_file)
        elif "pipeline" in bundle or "HistGradientBoosting" in model_type or "DemandForecastModel" in model_type:
            from demandcast.models import DemandForecastModel

            return DemandForecastModel.load(model_file)

    from demandcast.models import DemandForecastModel

    return DemandForecastModel.load(model_file)


class ModelRegistry:
    """Manages immutable model releases, verification, promotion, and rollback."""

    def __init__(self, config: RegistryConfig | None = None):
        self.config = config or load_config().registry
        self.root = get_project_root()

        rel_p = Path(self.config.releases_dir)
        self.releases_dir = rel_p if rel_p.is_absolute() else (self.root / rel_p)

        ptr_p = Path(self.config.current_pointer)
        self.current_pointer_path = ptr_p if ptr_p.is_absolute() else (self.root / ptr_p)

        self.releases_dir.mkdir(parents=True, exist_ok=True)
        self.current_pointer_path.parent.mkdir(parents=True, exist_ok=True)

    def package_release(
        self,
        model: Any,
        version: str,
        mlflow_run_id: str,
        validation_metrics: dict[str, Any],
        training_date_range: dict[str, str],
        test_metrics: dict[str, Any] | None = None,
        notes: str = "",
        quality_status: str | None = None,
        quality_alert: str | None = None,
        model_freshness: dict[str, Any] | None = None,
    ) -> Path:
        """Create an immutable release directory containing model artifact and metadata."""
        release_dir = self.releases_dir / version
        if release_dir.exists():
            raise ValueError(f"Release version '{version}' already exists at {release_dir}. Releases are immutable.")

        release_dir.mkdir(parents=True, exist_ok=False)

        # 1. Save model artifact
        model_file = release_dir / "model.joblib"
        model.save(model_file)
        checksum = calculate_file_sha256(model_file)

        # 2. Build metadata
        config = load_config()
        val_wape = validation_metrics.get("wape")
        threshold = config.governance.wape_promotion_gate_threshold
        has_alert = val_wape is not None and val_wape > threshold
        q_status = quality_status or ("ALERT" if has_alert else "HEALTHY")
        q_alert = quality_alert or (
            f"Validation WAPE ({val_wape * 100:.2f}%) exceeds target threshold ({threshold * 100:.2f}%). "
            "Selected model is best evaluated candidate promoted under audited operational override."
            if has_alert
            else None
        )
        freshness = model_freshness or {
            "status": "stale_historical_dataset",
            "dataset_vintage": config.governance.dataset_vintage,
            "training_cutoff_date": training_date_range.get("max_date", "2011-11-07"),
            "is_calibrated_for_modern_retail": False,
            "stale_threshold_days": config.governance.stale_model_threshold_days,
            "intended_usage": "Historical benchmark research, algorithm evaluation, and reproducible backtesting only.",
            "refresh_requirements": [
                "Ingest contemporary POS/ERP transaction stream with continuous Monday weekly aggregation",
                "Require minimum 52 weeks history for seasonal pattern modeling",
                "Execute expanding-window chronological backtesting across origins",
                "Verify validation WAPE <= 50.00% before non-override promotion",
            ],
        }

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model.name,
            "mlflow_run_id": mlflow_run_id,
            "checksum_sha256": checksum,
            "training_date_range": training_date_range,
            "supported_series": getattr(model, "supported_series", []),
            "feature_columns": getattr(model, "feature_cols", []),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics or {},
            "quality_status": q_status,
            "quality_alert": q_alert,
            "model_freshness": freshness,
            "notes": notes,
        }

        meta_file = release_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Packaged release '{version}' at {release_dir} (SHA-256: {checksum[:8]}...)")
        return release_dir

    def verify_release(self, version: str) -> tuple[bool, str]:
        """Verify release directory integrity: files exist, JSON is valid, checksum matches."""
        release_dir = self.releases_dir / version
        if not release_dir.exists():
            return False, f"Release directory does not exist: {release_dir}"

        model_file = release_dir / "model.joblib"
        meta_file = release_dir / "metadata.json"

        if not model_file.exists():
            return False, f"Missing model artifact file at {model_file}"
        if not meta_file.exists():
            return False, f"Missing metadata file at {meta_file}"

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            return False, f"Corrupted metadata JSON: {e}"

        expected_checksum = metadata.get("checksum_sha256")
        actual_checksum = calculate_file_sha256(model_file)

        if expected_checksum != actual_checksum:
            return False, (f"Checksum mismatch: expected {expected_checksum}, calculated {actual_checksum}")

        # Verify model can be deserialized and run a quick smoke test
        try:
            loaded_model = load_forecaster_artifact(model_file)
            if not getattr(loaded_model, "is_fitted", False):
                return False, "Model is not marked as fitted."
        except Exception as e:
            return False, f"Failed to load or test model artifact: {e}"

        return True, "Release verified successfully."

    def promote_release(
        self,
        version: str,
        reason: str | None = None,
        allow_alert_override: bool = False,
        quality_status: str | None = None,
        quality_alert: str | None = None,
    ) -> None:
        """Promote a release to current active model after strict verification and gate validation."""
        is_valid, msg = self.verify_release(version)
        if not is_valid:
            raise ValueError(f"Cannot promote release '{version}': {msg}")

        release_dir = self.releases_dir / version
        meta_file = release_dir / "metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        config = load_config()
        val_wape = meta.get("validation_metrics", {}).get("wape")
        threshold = config.governance.wape_promotion_gate_threshold
        has_alert = val_wape is not None and val_wape > threshold

        if has_alert:
            if not allow_alert_override:
                raise ValueError(
                    f"Promotion blocked: Candidate validation WAPE ({val_wape:.2%}) exceeds target threshold "
                    f"({threshold:.2%}). Quality gate denies promotion unless explicit allow_alert_override=True is set."
                )
            if not reason or not reason.strip():
                raise ValueError(
                    "Promotion blocked: An explicit non-empty audit reason string is required when promoting a "
                    "candidate that triggers quality alerts under an operational override."
                )
        else:
            if not reason or not reason.strip():
                reason = "Passed validation gates"

        try:
            rel_dir_str = release_dir.relative_to(self.root).as_posix()
        except ValueError:
            rel_dir_str = (
                release_dir.as_posix() if hasattr(release_dir, "as_posix") else str(release_dir).replace("\\", "/")
            )

        q_status = quality_status or meta.get("quality_status", "ALERT" if has_alert else "HEALTHY")
        q_alert = quality_alert or meta.get("quality_alert")

        pointer_data = {
            "current_version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "checksum_sha256": meta["checksum_sha256"],
            "release_dir": rel_dir_str,
            "reason": reason,
            "quality_status": q_status,
            "quality_alert": q_alert,
            "model_freshness_status": meta.get("model_freshness", {}).get("status", "stale_historical_dataset"),
        }

        # Write pointer atomically
        temp_pointer = self.current_pointer_path.with_suffix(".tmp")
        with open(temp_pointer, "w", encoding="utf-8") as f:
            json.dump(pointer_data, f, indent=2)
        shutil.move(str(temp_pointer), str(self.current_pointer_path))

        logger.info(f"Successfully promoted model release '{version}' to active. Reason: {reason}")

    def rollback_to(self, target_version: str, reason: str = "Manual rollback") -> None:
        """Roll back active model pointer to a previous valid version."""
        logger.info(f"Initiating rollback to version '{target_version}'...")
        is_valid, msg = self.verify_release(target_version)
        if not is_valid:
            raise ValueError(f"Cannot rollback to version '{target_version}': {msg}")

        self.promote_release(target_version, reason=f"Rollback: {reason}", allow_alert_override=True)

    def get_current_release_info(self) -> dict[str, Any] | None:
        """Return current pointer and metadata information."""
        if not self.current_pointer_path.exists():
            return None

        with open(self.current_pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)

        version = pointer["current_version"]
        meta_file = self.releases_dir / version / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pointer["metadata"] = meta

        return pointer

    def load_current_model(self) -> tuple[Any, dict[str, Any]]:
        """Load currently promoted model and its verified metadata."""
        info = self.get_current_release_info()
        if info is None:
            raise FileNotFoundError("No active model is promoted. current.json not found.")

        version = info["current_version"]
        model_file = self.releases_dir / version / "model.joblib"
        model = load_forecaster_artifact(model_file)
        return model, info.get("metadata", {})
