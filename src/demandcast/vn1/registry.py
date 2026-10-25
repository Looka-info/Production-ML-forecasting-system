"""Isolated Model Registry for VN1 Retail Forecasting Profile."""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demandcast.config import get_project_root
from demandcast.vn1.config import VN1Config, load_vn1_config
from demandcast.vn1.models import VN1Forecaster

logger = logging.getLogger(__name__)


def calculate_file_sha256(filepath: Path | str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class VN1ModelRegistry:
    """Registry managing versioned model releases and current pointer for VN1 profile."""

    def __init__(self, config: VN1Config | None = None):
        self.config = config or load_vn1_config()
        self.root = get_project_root()
        self.releases_dir = self.root / self.config.registry.releases_dir
        self.current_pointer_path = self.root / self.config.registry.current_pointer
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        self.current_pointer_path.parent.mkdir(parents=True, exist_ok=True)

    def package_release(
        self,
        model: Any,
        version: str,
        mlflow_run_id: str,
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any] | None = None,
        notes: str = "",
        quality_status: str | None = None,
        quality_alert: str | None = None,
    ) -> Path:
        """Create an immutable release directory containing model artifact and metadata."""
        release_dir = self.releases_dir / version
        if release_dir.exists():
            raise ValueError(f"Release version '{version}' already exists at {release_dir}. Releases are immutable.")

        release_dir.mkdir(parents=True, exist_ok=False)

        # 1. Save model artifact
        model_file = release_dir / "model.joblib"
        if hasattr(model, "save"):
            model.save(model_file)
        else:
            import joblib

            joblib.dump(model, model_file)

        checksum = calculate_file_sha256(model_file)

        # 2. Evaluate quality gates
        score = validation_metrics.get("vn1_score")
        threshold = self.config.governance.target_gate_score
        has_alert = score is not None and score > threshold

        q_status = quality_status or ("ALERT" if has_alert else "HEALTHY")
        q_alert = quality_alert or (
            f"Validation VN1 score ({score:.4f}) exceeds target threshold ({threshold:.4f}). "
            "Model promoted under audited operational override."
            if has_alert
            else None
        )

        freshness = {
            "status": "stale_historical_dataset",
            "dataset_vintage": self.config.governance.dataset_vintage,
            "training_cutoff_date": self.config.governance.training_cutoff_date,
            "is_calibrated_for_modern_retail": False,
            "stale_threshold_days": self.config.governance.stale_model_threshold_days,
            "intended_usage": "VN1 Forecasting Challenge research and algorithm benchmarking only.",
            "refresh_requirements": [
                "Ingest contemporary retail transaction and price stream with continuous Monday cadence",
                "Require minimum 52 weeks history for annual seasonality modeling",
                "Execute 13-week chronological multi-step validation backtests",
                "Verify validation VN1 score <= 0.7000 before non-override promotion",
            ],
        }

        feature_cols = getattr(model, "feature_cols", [])
        supported_series = getattr(model, "supported_series", [])

        metadata = {
            "version": version,
            "dataset_profile": "vn1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": getattr(model, "name", "VN1Model"),
            "mlflow_run_id": mlflow_run_id,
            "checksum_sha256": checksum,
            "forecast_horizon": self.config.forecasting.horizon,
            "training_date_range": {
                "start_date": self.config.temporal.phase0_start,
                "end_date": self.config.temporal.phase0_end,
            },
            "supported_series_count": len(supported_series),
            "feature_columns": feature_cols,
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

        logger.info(f"Packaged VN1 release '{version}' at {release_dir} (SHA-256: {checksum[:8]}...)")
        return release_dir

    def verify_release(self, version: str) -> tuple[bool, str]:
        """Verify release integrity: artifact exists and matches SHA-256 in metadata."""
        release_dir = self.releases_dir / version
        if not release_dir.exists():
            return False, f"Release directory does not exist: {release_dir}"

        model_file = release_dir / "model.joblib"
        meta_file = release_dir / "metadata.json"
        if not model_file.exists():
            return False, f"Model artifact missing at {model_file}"
        if not meta_file.exists():
            return False, f"Metadata file missing at {meta_file}"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        actual_sha = calculate_file_sha256(model_file)
        expected_sha = meta.get("checksum_sha256")
        if actual_sha != expected_sha:
            return False, f"Checksum mismatch for version {version}: expected {expected_sha}, got {actual_sha}"

        return True, "Release integrity verified"

    def promote_release(
        self,
        version: str,
        reason: str = "",
        allow_alert_override: bool = False,
    ) -> None:
        """Atomically promote a verified VN1 release to active pointer."""
        is_valid, msg = self.verify_release(version)
        if not is_valid:
            raise ValueError(f"Cannot promote release '{version}': {msg}")

        release_dir = self.releases_dir / version
        with open(release_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        score = meta.get("validation_metrics", {}).get("vn1_score")
        threshold = self.config.governance.target_gate_score
        has_alert = score is not None and score > threshold

        if has_alert:
            if not allow_alert_override:
                raise ValueError(
                    f"Promotion blocked: Candidate validation VN1 score ({score:.4f}) exceeds target threshold "
                    f"({threshold:.4f}). Quality gate denies promotion unless explicit allow_alert_override=True is set."
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
            rel_dir_str = release_dir.as_posix()

        pointer_data = {
            "dataset_profile": "vn1",
            "current_version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "checksum_sha256": meta["checksum_sha256"],
            "release_dir": rel_dir_str,
            "reason": reason,
            "quality_status": meta.get("quality_status", "HEALTHY"),
            "quality_alert": meta.get("quality_alert"),
            "model_freshness_status": meta.get("model_freshness", {}).get("status", "stale_historical_dataset"),
        }

        # Write pointer atomically
        temp_pointer = self.current_pointer_path.with_suffix(".tmp")
        with open(temp_pointer, "w", encoding="utf-8") as f:
            json.dump(pointer_data, f, indent=2)
        shutil.move(str(temp_pointer), str(self.current_pointer_path))

        logger.info(f"Successfully promoted VN1 model release '{version}' to active. Reason: {reason}")

    def rollback_to(self, target_version: str, reason: str = "Manual rollback") -> None:
        """Roll back active VN1 pointer to a prior valid version."""
        is_valid, msg = self.verify_release(target_version)
        if not is_valid:
            raise ValueError(f"Cannot rollback to version '{target_version}': {msg}")
        self.promote_release(target_version, reason=f"Rollback: {reason}", allow_alert_override=True)

    def get_current_release_info(self) -> dict[str, Any] | None:
        """Return active VN1 pointer and metadata."""
        if not self.current_pointer_path.exists():
            return None
        with open(self.current_pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        version = pointer["current_version"]
        meta_file = self.releases_dir / version / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                pointer["metadata"] = json.load(f)
        return pointer

    def load_current_model(self) -> tuple[Any, dict[str, Any]]:
        """Load currently active VN1 model bundle and metadata."""
        info = self.get_current_release_info()
        if info is None:
            raise FileNotFoundError("No active VN1 model is promoted. models/profiles/vn1/current.json not found.")
        version = info["current_version"]
        model_file = self.releases_dir / version / "model.joblib"
        try:
            model = VN1Forecaster.load(model_file)
        except Exception:
            import joblib

            model = joblib.load(model_file)
        return model, info.get("metadata", {})
