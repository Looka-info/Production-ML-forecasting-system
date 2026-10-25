"""Script to promote a verified model release to active serving."""

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from demandcast.config import load_config
from demandcast.forecasting import generate_forecast_response
from demandcast.registry import ModelRegistry, load_forecaster_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("promote_model")


def run_smoke_test(model: Any, version: str, training_cutoff: str | None = None) -> None:
    """Execute an end-to-end inference smoke test before promotion."""
    sample_sku = model.supported_series[0]
    cutoff_dt = pd.to_datetime(training_cutoff or "2011-11-07")
    # Supply 14 consecutive weeks ending precisely on the model's training cutoff date
    start_dt = cutoff_dt - pd.Timedelta(weeks=13)
    proper_history = [
        {
            "week_start": (start_dt + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(25.0 + (i % 5)),
        }
        for i in range(14)
    ]

    res = generate_forecast_response(
        model=model,
        model_version=version,
        series_id=sample_sku,
        history_records=proper_history,
        horizon=4,
        training_cutoff_date=training_cutoff,
    )
    assert len(res["forecasts"]) == 4, "Smoke test failed: Expected 4 forecast steps."
    logger.info(
        f"Smoke test passed for SKU '{sample_sku}'. Forecast: {[f['predicted_demand'] for f in res['forecasts']]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Promote a verified model release to active serving.")
    parser.add_argument("--version", type=str, required=True, help="Release version to promote (e.g. v1.0.0)")
    parser.add_argument(
        "--reason", type=str, default="", help="Audit reason for promotion. Required if overriding quality alerts."
    )
    parser.add_argument(
        "--allow-alert-override",
        action="store_true",
        default=False,
        help="Explicitly permit promotion of a candidate that exceeds quality alert thresholds.",
    )
    args = parser.parse_args()

    registry = ModelRegistry()
    config = load_config()

    logger.info(f"1. Verifying release '{args.version}' checksum and metadata...")
    is_valid, msg = registry.verify_release(args.version)
    if not is_valid:
        logger.error(f"Verification failed: {msg}")
        sys.exit(1)

    logger.info("2. Running inference smoke test on candidate artifact...")
    model_file = registry.releases_dir / args.version / "model.joblib"
    meta_file = registry.releases_dir / args.version / "metadata.json"
    import json

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cutoff = meta.get("training_date_range", {}).get("max_date")
    model = load_forecaster_artifact(model_file)
    run_smoke_test(model, args.version, training_cutoff=cutoff)

    # Inspect promotion quality gates
    val_wape = meta.get("validation_metrics", {}).get("wape")
    gate_threshold = config.governance.wape_promotion_gate_threshold
    has_alert = val_wape is not None and val_wape > gate_threshold

    if has_alert:
        logger.warning(
            f"Quality Alert: Candidate '{args.version}' validation WAPE ({val_wape * 100:.2f}%) "
            f"exceeds target threshold ({gate_threshold * 100:.2f}%)."
        )
        if not args.allow_alert_override:
            logger.error(
                f"Promotion BLOCKED: Candidate validation WAPE ({val_wape * 100:.2f}%) exceeds target threshold "
                f"({gate_threshold * 100:.2f}%). By default, overrides are denied. "
                "To promote this candidate as the best available model, provide '--allow-alert-override' "
                "with an explicit, audited '--reason <text>'."
            )
            sys.exit(1)

        if not args.reason or not args.reason.strip() or args.reason == "Validation gates passed":
            logger.error(
                "Promotion BLOCKED: When --allow-alert-override is specified, an explicit non-empty "
                "--reason string documenting the operational justification is required."
            )
            sys.exit(1)

        logger.info(
            f"Audited Override: Proceeding with promotion because release '{args.version}' is the "
            f"best evaluated candidate under documented business objective. Reason: {args.reason}"
        )

    q_status = meta.get("quality_status", "ALERT" if has_alert else "HEALTHY")
    q_alert = meta.get("quality_alert")
    if not q_alert and has_alert:
        q_alert = (
            f"Validation WAPE ({val_wape * 100:.2f}%) exceeds target threshold ({gate_threshold * 100:.2f}%). "
            "Selected model is best evaluated candidate promoted under audited operational override."
        )

    reason = args.reason.strip() if args.reason.strip() else "Passed validation gates"

    logger.info(f"3. Promoting release '{args.version}' to active (updating current.json)...")
    try:
        registry.promote_release(
            args.version,
            reason=reason,
            allow_alert_override=args.allow_alert_override,
            quality_status=q_status,
            quality_alert=q_alert,
        )
    except ValueError as e:
        logger.error(f"Promotion failed: {e}")
        sys.exit(1)

    current_info = registry.get_current_release_info()
    logger.info("=" * 60)
    logger.info("Model Promotion Succeeded:")
    logger.info(f"Active Version: {current_info.get('current_version')}")
    logger.info(f"Checksum: {current_info.get('checksum_sha256')}")
    logger.info(f"Quality Status: {current_info.get('quality_status')}")
    if current_info.get("quality_alert"):
        logger.info(f"Quality Alert: {current_info.get('quality_alert')}")
    logger.info(f"Freshness Status: {current_info.get('model_freshness_status')}")
    logger.info(f"Promoted At: {current_info.get('promoted_at')}")
    logger.info(f"Reason: {current_info.get('reason')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
