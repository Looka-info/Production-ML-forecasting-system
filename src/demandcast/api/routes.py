"""REST API route handlers for DemandCast."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from demandcast.api.dependencies import get_model_container, require_ready_model
from demandcast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetadataResponse,
    ReadyResponse,
)
from demandcast.config import load_config
from demandcast.forecasting import generate_forecast_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Lightweight service liveness check",
)
def health() -> HealthResponse:
    """Returns 200 OK if the API server process is alive and receiving traffic."""
    return HealthResponse(
        status="healthy",
        service="DemandCast",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
    summary="Service readiness check",
)
def ready(container=Depends(get_model_container)) -> ReadyResponse:
    """Checks whether the promoted model artifact, metadata, and feature configuration

    are loaded and ready to serve inference requests.
    Returns HTTP 503 if the model is not ready.
    """
    if not container.is_ready or container.model is None or container.version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "message": container.init_error or "Model is not loaded into memory.",
            },
        )

    meta = container.metadata or {}
    val_wape = meta.get("validation_metrics", {}).get("wape")
    has_alert = val_wape is not None and val_wape > 0.50
    q_status = meta.get("quality_status", "ALERT" if has_alert else "HEALTHY")
    q_alert = meta.get("quality_alert")
    if not q_alert and has_alert:
        q_alert = (
            f"Validation WAPE ({val_wape * 100:.2f}%) exceeds target threshold (50.00%). "
            "Selected model is best evaluated candidate promoted under audited operational override."
        )
    freshness_status = meta.get("model_freshness", {}).get("status", "stale_historical_dataset")

    return ReadyResponse(
        status="ready",
        model_version=container.version,
        model_name=container.model.name,
        supported_series_count=len(container.model.supported_series),
        checksum_sha256=container.metadata.get("checksum_sha256", "UNKNOWN"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        quality_status=q_status,
        quality_alert=q_alert,
        model_freshness_status=freshness_status,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["Governance"],
    summary="Model governance and operational metadata",
)
def metadata(model_data=Depends(require_ready_model)) -> MetadataResponse:
    """Returns model version, target definition, supported series catalog,

    training date range, validation metrics, model freshness, and known operational limitations.
    """
    model, meta, version = model_data
    min_hist = getattr(model, "min_history_required", 13)
    feature_cols = meta.get("feature_columns", getattr(model, "feature_cols", []))
    config = load_config()

    val_wape = meta.get("validation_metrics", {}).get("wape")
    has_alert = val_wape is not None and val_wape > 0.50
    q_status = meta.get("quality_status", "ALERT" if has_alert else "HEALTHY")
    q_alert = meta.get("quality_alert")
    if not q_alert and has_alert:
        q_alert = (
            f"Validation WAPE ({val_wape * 100:.2f}%) exceeds target threshold (50.00%). "
            "Selected model is best evaluated candidate promoted under audited operational override."
        )

    freshness_info = meta.get("model_freshness", {
        "status": "stale_historical_dataset",
        "dataset_vintage": config.governance.dataset_vintage,
        "training_cutoff_date": meta.get("training_date_range", {}).get("max_date", "2011-11-07"),
        "is_calibrated_for_modern_retail": False,
        "stale_threshold_days": config.governance.stale_model_threshold_days,
        "intended_usage": "Historical benchmark research, algorithm evaluation, and reproducible backtesting only.",
        "refresh_requirements": [
            "Ingest contemporary POS/ERP transaction stream with continuous Monday weekly aggregation",
            "Require minimum 52 weeks history for seasonal pattern modeling",
            "Execute expanding-window chronological backtesting across origins",
            "Verify validation WAPE <= 50.00% before non-override promotion",
        ],
    })

    return MetadataResponse(
        service_name="DemandCast",
        model_version=version,
        model_type=meta.get("model_type", model.name),
        target_definition="Weekly units sold per selected product (sum of positive sales quantities as demand proxy)",
        forecast_horizon_weeks=4,
        minimum_history_weeks_required=min_hist,
        supported_series=getattr(model, "supported_series", []),
        training_date_range=meta.get("training_date_range", {}),
        validation_metrics=meta.get("validation_metrics", {}),
        feature_columns=feature_cols,
        quality_status=q_status,
        quality_alert=q_alert,
        model_freshness=freshness_info,
        limitations=[
            "Quality alert status: Validation WAPE of 55.16% exceeds the 50.00% target error threshold; promoted under audited override as the best evaluated candidate.",
            "Historical data limitation: Built on UCI Online Retail II (2009-2011). Not calibrated for modern retail market dynamics without retraining on contemporary data.",
            "Historical replay vs live: Predictions for inputs dated after the training cutoff represent historical replay / backtesting and DO NOT establish validity in modern retail environments.",
            "Sales proxy limitation: Observed positive sales quantities do not measure unconstrained demand during stockouts.",
            "Temporal boundary: Request forecast origin must not predate active model training cutoff.",
            "Product scope: Supported products are restricted to the validated training catalog.",
            "Weekly convention: Observations must follow strictly consecutive Monday week-start intervals.",
        ],
    )


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Generate weekly demand forecast",
)
def forecast(
    request: ForecastRequest,
    model_data=Depends(require_ready_model),
) -> ForecastResponse:
    """Accepts a product SKU and consecutive historical weekly sales observations.

    Generates recursive point forecasts up to 4 weeks ahead.
    """
    model, meta, version = model_data
    cutoff = meta.get("training_date_range", {}).get("max_date")

    try:
        raw_history = [item.model_dump() for item in request.history]
        response_dict = generate_forecast_response(
            model=model,
            model_version=version,
            series_id=request.series_id,
            history_records=raw_history,
            horizon=request.horizon,
            training_cutoff_date=cutoff,
        )
        return ForecastResponse(**response_dict)
    except ValueError as e:
        msg = str(e)
        logger.warning(f"Inference input validation error: {msg}")
        err_type = "TemporalContractViolation" if "Temporal contract violation" in msg else "InvalidInputData"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": err_type, "message": msg},
        )
    except Exception as e:
        logger.error(f"Unexpected error during forecast generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "ForecastGenerationFailed", "message": "An internal error occurred."},
        )
