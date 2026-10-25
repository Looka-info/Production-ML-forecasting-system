"""Inference orchestration and response formatting for DemandCast."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from demandcast.config import load_config

logger = logging.getLogger(__name__)


def validate_inference_history(
    series_id: str,
    history_records: list[dict[str, Any]],
    supported_series: list[str],
    min_history_required: int = 13,
    training_cutoff_date: str | pd.Timestamp | None = None,
    stale_threshold_days: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Strictly validate input historical series for inference.

    Validation checks:
    1. series_id in supported_series.
    2. Non-empty history with at least min_history_required observations.
    3. Strictly chronological ordering with no duplicate dates.
    4. Exact 7-day weekly step continuity (no skipped or missing weeks).
    5. Nonnegative, finite demand values.
    6. Temporal contract: forecast origin must not predate model training cutoff.
    7. Stale-model warning: origin substantially past training cutoff.
    """
    warnings: list[str] = []

    if series_id not in supported_series:
        raise ValueError(
            f"Unsupported series_id '{series_id}'. Model supports: {supported_series}"
        )

    if len(history_records) < min_history_required:
        raise ValueError(
            f"Insufficient history for series '{series_id}'. Provided {len(history_records)} "
            f"observations, but at least {min_history_required} continuous weeks are required."
        )

    df = pd.DataFrame(history_records)
    if "week_start" not in df.columns or "demand" not in df.columns:
        raise ValueError("Each history item must contain 'week_start' and 'demand' fields.")

    # Convert week_start
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if df["week_start"].isna().any():
        raise ValueError("Invalid date format in history. Expected YYYY-MM-DD.")

    # Check for duplicates
    if df["week_start"].duplicated().any():
        raise ValueError("Duplicate weekly dates detected in supplied history.")

    # Sort
    df = df.sort_values(by="week_start").reset_index(drop=True)

    # Check date continuity (every interval must be exactly 7 days)
    date_diffs = df["week_start"].diff().dropna()
    expected_diff = pd.Timedelta(days=7)
    if not (date_diffs == expected_diff).all():
        raise ValueError(
            "Non-continuous weekly history. Dates must be exactly 7 days apart with no gaps."
        )

    # Check temporal contract against model training cutoff
    last_date = df["week_start"].iloc[-1]
    if training_cutoff_date is not None:
        cutoff = pd.to_datetime(training_cutoff_date)
        if last_date < cutoff:
            raise ValueError(
                f"Temporal contract violation: Forecast origin ({last_date.strftime('%Y-%m-%d')}) "
                f"predates active model training cutoff ({cutoff.strftime('%Y-%m-%d')}). "
                "Forecasting past dates using a model trained on subsequent observations violates "
                "temporal integrity (in-sample lookahead). For historical backtesting, use origin-aligned "
                f"backtests or supply history ending on or after {cutoff.strftime('%Y-%m-%d')}."
            )

        # Check for stale model / historical dataset warning
        config = load_config()
        threshold = (
            stale_threshold_days
            if stale_threshold_days is not None
            else getattr(config.governance, "stale_model_threshold_days", 90)
        )
        days_past_cutoff = (last_date - cutoff).days
        if days_past_cutoff > threshold:
            warnings.append(
                f"StaleModelWarning: Forecast origin ({last_date.strftime('%Y-%m-%d')}) is {days_past_cutoff} "
                f"days past model training cutoff ({cutoff.strftime('%Y-%m-%d')}). Model was trained on historical "
                f"data ({config.governance.dataset_vintage}); predictions beyond the training horizon represent "
                "historical backtesting/replay and do not establish validity for modern retail markets."
            )

    # Demand validation
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    if df["demand"].isna().any() or np.isinf(df["demand"]).any():
        raise ValueError("History contains non-numeric, NaN, or infinite demand values.")

    if (df["demand"] < 0).any():
        raise ValueError("Demand values must be non-negative observed quantities.")

    # Check for warnings: e.g. recent zero streak
    recent_zeros = (df["demand"].tail(4) == 0).sum()
    if recent_zeros >= 3:
        warnings.append(
            f"Series has {recent_zeros} zero-demand weeks in the last 4 observations."
        )

    return df[["week_start", "demand"]], warnings


def generate_forecast_response(
    model: Any,
    model_version: str,
    series_id: str,
    history_records: list[dict[str, Any]],
    horizon: int = 4,
    min_history_required: int | None = None,
    training_cutoff_date: str | pd.Timestamp | None = None,
    stale_threshold_days: int | None = None,
) -> dict[str, Any]:
    """Validate history, run recursive forecast, and return structured prediction payload."""
    config = load_config()
    model_min_hist = getattr(model, "min_history_required", config.forecasting.min_history_required)
    min_hist = min_history_required or model_min_hist

    if horizon < 1 or horizon > 4:
        raise ValueError(f"Requested horizon {horizon} is invalid. Supported range: 1 to 4.")

    history_df, warnings = validate_inference_history(
        series_id=series_id,
        history_records=history_records,
        supported_series=getattr(model, "supported_series", []),
        min_history_required=min_hist,
        training_cutoff_date=training_cutoff_date,
        stale_threshold_days=stale_threshold_days,
    )

    # Generate forecast
    forecast_df = model.forecast(
        series_id=series_id,
        history_df=history_df,
        horizon=horizon,
    )

    last_observed_date = history_df["week_start"].iloc[-1].strftime("%Y-%m-%d")
    forecast_start_date = forecast_df["week_start"].iloc[0].strftime("%Y-%m-%d")
    forecast_end_date = forecast_df["week_start"].iloc[-1].strftime("%Y-%m-%d")

    items = []
    for _, row in forecast_df.iterrows():
        items.append({
            "week_start": row["week_start"].strftime("%Y-%m-%d"),
            "horizon_step": int(row["horizon_step"]),
            "predicted_demand": round(float(row["predicted_demand"]), 2),
        })

    response = {
        "forecast_run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "series_id": series_id,
        "model_version": model_version,
        "model_name": model.name,
        "last_historical_date": last_observed_date,
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": forecast_end_date,
        "horizon": horizon,
        "forecasts": items,
        "warnings": warnings,
    }

    return response
