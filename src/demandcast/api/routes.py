"""REST API route handlers for DemandCast (UCI & VN1 Retail Profiles)."""

import logging
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from demandcast.api.dependencies import (
    get_model_container,
    get_vn1_model_container,
    require_ready_model,
    require_ready_vn1_model,
)
from demandcast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetadataResponse,
    ReadyResponse,
    VN1ForecastItem,
    VN1ForecastRequest,
    VN1ForecastResponse,
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
def ready(
    container=Depends(get_model_container),
    vn1_container=Depends(get_vn1_model_container),
) -> ReadyResponse:
    """Checks whether promoted model releases (UCI and VN1) are loaded and ready.

    Returns HTTP 503 if the primary model is not ready.
    """
    if not container.is_ready or container.model is None or container.version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "message": container.init_error or "UCI primary model is not loaded into memory.",
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
        active_profiles=["uci", "vn1"] if vn1_container.is_ready else ["uci"],
        vn1_ready=vn1_container.is_ready,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    tags=["Governance"],
    summary="Model governance and operational metadata",
)
def metadata(
    model_data=Depends(require_ready_model),
    vn1_container=Depends(get_vn1_model_container),
) -> MetadataResponse:
    """Returns model versions, governance metadata, freshness status, and limitations for UCI and VN1 profiles."""
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

    freshness_info = meta.get(
        "model_freshness",
        {
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
        },
    )

    vn1_meta = None
    if vn1_container.is_ready and vn1_container.metadata:
        vn1_meta = {
            "model_version": vn1_container.version,
            "model_type": vn1_container.metadata.get("model_type"),
            "dataset_profile": "vn1",
            "forecast_horizon_weeks": vn1_container.metadata.get("forecast_horizon", 13),
            "training_date_range": vn1_container.metadata.get("training_date_range", {}),
            "validation_metrics": vn1_container.metadata.get("validation_metrics", {}),
            "quality_status": vn1_container.metadata.get("quality_status", "HEALTHY"),
            "model_freshness": vn1_container.metadata.get("model_freshness", {}),
        }

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
        active_profiles=["uci", "vn1"] if vn1_container.is_ready else ["uci"],
        vn1_profile=vn1_meta,
        limitations=[
            "Quality alert status: Validation WAPE of 55.16% exceeds the 50.00% target error threshold; promoted under audited override as the best evaluated candidate.",
            "Historical data limitation: Built on UCI Online Retail II (2009-2011) and VN1 Challenge (2020-2023). Not calibrated for modern retail market dynamics without contemporary retraining.",
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
    summary="Generate weekly demand forecast (UCI profile)",
)
def forecast(
    request: ForecastRequest,
    model_data=Depends(require_ready_model),
) -> ForecastResponse:
    """Accepts a product SKU and consecutive historical weekly sales observations.

    Generates recursive point forecasts up to 4 weeks ahead using active UCI profile model.
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


@router.post(
    "/v2/forecast",
    response_model=VN1ForecastResponse,
    tags=["Forecasting (v2 - VN1)"],
    summary="Generate 13-week retail sales forecast (VN1 profile)",
)
def forecast_v2(
    request: VN1ForecastRequest,
    model_data=Depends(require_ready_vn1_model),
) -> VN1ForecastResponse:
    """Accepts a product-warehouse series and weekly sales observations (1–13 week horizon).

    Uses active VN1 model release with strict chronological validation and non-negative output clipping.
    """
    model, meta, version = model_data
    cutoff_str = meta.get("training_date_range", {}).get("end_date", "2023-10-02")
    cutoff_date = pd.to_datetime(cutoff_str)

    try:
        # 1. Parse and validate history
        rows = [item.model_dump() for item in request.history]
        df_hist = pd.DataFrame(rows)
        df_hist["week_start"] = pd.to_datetime(df_hist["week_start"])
        df_hist = df_hist.sort_values("week_start").reset_index(drop=True)

        # Validate unique dates
        if df_hist["week_start"].duplicated().any():
            raise ValueError("Duplicate week_start dates detected in historical sequence.")

        # Validate Monday convention
        if not (df_hist["week_start"].dt.day_of_week == 0).all():
            raise ValueError("All historical week_start dates must conform strictly to Monday week-start convention.")

        # Validate continuous weekly intervals (no skipped weeks)
        date_diffs = df_hist["week_start"].diff().dropna()
        if not (date_diffs == pd.Timedelta(days=7)).all():
            raise ValueError(
                "Historical sequence contains gaps or irregular intervals; expected strictly consecutive 7-day intervals."
            )

        # Minimum history validation
        if len(df_hist) < 12:
            raise ValueError(f"Insufficient historical observations: supplied {len(df_hist)}, minimum 12 required.")

        origin_date = df_hist["week_start"].iloc[-1]

        # In-sample lookahead check
        if origin_date < cutoff_date:
            raise ValueError(
                f"Temporal contract violation: Forecast origin ({origin_date.strftime('%Y-%m-%d')}) predates "
                f"active VN1 model training cutoff ({cutoff_str}). Forecasting past dates using a model trained "
                "on subsequent observations violates temporal integrity (in-sample lookahead bias)."
            )

        # Attach series identifiers to dataframe
        parts = request.series_id.split("_")
        client_id = parts[0] if len(parts) > 0 else "0"
        warehouse_id = parts[1] if len(parts) > 1 else "1"
        product_id = parts[2] if len(parts) > 2 else request.series_id

        df_hist["series_id"] = request.series_id
        df_hist["client_id"] = client_id
        df_hist["warehouse_id"] = warehouse_id
        df_hist["product_id"] = product_id
        if "price" not in df_hist.columns or df_hist["price"].isna().all():
            df_hist["price"] = 0.0

        warnings = []
        days_diff = (origin_date - cutoff_date).days
        if days_diff > 90:
            warnings.append(
                f"StaleModelWarning: Forecast origin ({origin_date.strftime('%Y-%m-%d')}) is {days_diff} days past "
                f"model training cutoff ({cutoff_str}). Model was trained on historical VN1 dataset (2020-2023); "
                "predictions beyond the training horizon represent historical backtesting/replay and do not establish "
                "validity for modern retail markets."
            )

        # Generate predictions
        if hasattr(model, "predict_recursive"):
            preds = model.predict_recursive(df_hist, horizon=request.horizon, origin_date=origin_date)
        else:
            preds = model.predict(df_hist, horizon=request.horizon, origin_date=origin_date)

        # Format forecast items
        forecast_items = []
        for _, row in preds.iterrows():
            fc_step = int(row["horizon_step"])
            fc_date = (
                row["week_start"].strftime("%Y-%m-%d")
                if hasattr(row["week_start"], "strftime")
                else str(row["week_start"])
            )
            pred_val = float(max(0.0, row["predicted_sales"]))
            forecast_items.append(
                VN1ForecastItem(
                    week_start=fc_date,
                    horizon_step=fc_step,
                    predicted_sales=round(pred_val, 2),
                )
            )

        start_date = forecast_items[0].week_start
        end_date = forecast_items[-1].week_start

        return VN1ForecastResponse(
            forecast_run_id=str(uuid.uuid4()),
            dataset_profile="vn1",
            created_at=datetime.now(timezone.utc).isoformat(),
            series_id=request.series_id,
            model_version=version,
            model_name=getattr(model, "name", "VN1Model"),
            last_historical_date=origin_date.strftime("%Y-%m-%d"),
            forecast_start_date=start_date,
            forecast_end_date=end_date,
            horizon=request.horizon,
            forecasts=forecast_items,
            warnings=warnings,
        )

    except ValueError as e:
        msg = str(e)
        logger.warning(f"VN1 Inference input validation error: {msg}")
        err_type = "TemporalContractViolation" if "Temporal contract violation" in msg else "InvalidInputData"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": err_type, "message": msg},
        )
    except Exception as e:
        logger.error(f"Unexpected error during VN1 forecast generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "VN1ForecastGenerationFailed", "message": f"An internal error occurred: {e}"},
        )
