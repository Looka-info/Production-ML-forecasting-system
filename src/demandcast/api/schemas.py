"""Pydantic schemas for DemandCast REST API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class WeeklyDemandItem(BaseModel):
    """A single weekly historical observation."""

    week_start: str = Field(
        ...,
        description="Weekly observation start date in YYYY-MM-DD format (Monday convention).",
        examples=["2011-09-05"],
    )
    demand: float = Field(
        ...,
        ge=0.0,
        description="Cleaned positive sales units observed in this week.",
        examples=[45.0],
    )


class ForecastRequest(BaseModel):
    """Inference request payload."""

    series_id: str = Field(
        ...,
        description="Supported product SKU / series identifier.",
        examples=["85123A"],
    )
    horizon: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Forecast horizon in weekly steps (1 to 4 weeks ahead).",
        examples=[4],
    )
    history: list[WeeklyDemandItem] = Field(
        ...,
        min_length=13,
        description="Continuous weekly historical observations. Minimum 13 weeks required.",
    )

    @field_validator("series_id")
    @classmethod
    def clean_series_id(cls, v: str) -> str:
        cleaned = str(v).strip().upper()
        if not cleaned:
            raise ValueError("series_id must not be empty.")
        return cleaned


class ForecastItem(BaseModel):
    """Forecast for a single future week."""

    week_start: str = Field(..., description="Target forecast week start date (YYYY-MM-DD).")
    horizon_step: int = Field(..., ge=1, le=4, description="Step index (1..4).")
    predicted_demand: float = Field(..., ge=0.0, description="Predicted non-negative demand units.")


class ForecastResponse(BaseModel):
    """Structured response from the forecasting service."""

    forecast_run_id: str = Field(..., description="Unique UUID for this prediction run.")
    created_at: str = Field(..., description="UTC ISO timestamp of prediction generation.")
    series_id: str = Field(..., description="Product identifier.")
    model_version: str = Field(..., description="Active model release version.")
    model_name: str = Field(..., description="Estimator architecture name.")
    last_historical_date: str = Field(..., description="Most recent supplied history date.")
    forecast_start_date: str = Field(..., description="First forecast week start date.")
    forecast_end_date: str = Field(..., description="Last forecast week start date.")
    horizon: int = Field(..., description="Number of weekly steps predicted.")
    forecasts: list[ForecastItem] = Field(..., description="List of weekly predictions.")
    warnings: list[str] = Field(default_factory=list, description="Data quality or pattern warnings.")


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: str = "healthy"
    service: str = "DemandCast"
    timestamp: str


class ReadyResponse(BaseModel):
    """Service readiness response indicating model availability."""

    status: str = "ready"
    model_version: str
    model_name: str
    supported_series_count: int
    checksum_sha256: str
    timestamp: str
    quality_status: str = Field(default="ALERT", description="Operational quality status: HEALTHY, WARNING, or ALERT.")
    quality_alert: str | None = Field(
        default=None, description="Active quality alert description if thresholds are exceeded."
    )
    model_freshness_status: str = Field(
        default="stale_historical_dataset", description="Model data vintage / freshness category."
    )
    active_profiles: list[str] = Field(
        default_factory=lambda: ["uci", "vn1"], description="List of initialized dataset profiles."
    )
    vn1_ready: bool = Field(default=False, description="Whether VN1 profile is loaded and ready.")


class MetadataResponse(BaseModel):
    """Model governance metadata response."""

    service_name: str
    model_version: str
    model_type: str
    target_definition: str
    forecast_horizon_weeks: int
    minimum_history_weeks_required: int
    supported_series: list[str]
    training_date_range: dict[str, str]
    validation_metrics: dict[str, Any]
    feature_columns: list[str]
    limitations: list[str]
    quality_status: str = Field(default="ALERT", description="Operational quality status (HEALTHY, WARNING, or ALERT).")
    quality_alert: str | None = Field(default=None, description="Detailed quality alert explanation.")
    model_freshness: dict[str, Any] = Field(
        default_factory=dict, description="Structured dataset vintage, freshness, and retraining requirements."
    )
    active_profiles: list[str] = Field(default_factory=lambda: ["uci", "vn1"], description="Active dataset profiles.")
    vn1_profile: dict[str, Any] | None = Field(default=None, description="Metadata for VN1 retail challenge profile.")


# ==============================================================================
# DemandCast v2 — VN1 Retail Forecasting Schemas
# ==============================================================================


class VN1WeeklyObservation(BaseModel):
    """A single weekly sales observation for VN1 product-warehouse series."""

    week_start: str = Field(
        ...,
        description="Monday week-start date in YYYY-MM-DD format.",
        examples=["2023-09-04"],
    )
    sales: float = Field(
        ...,
        ge=0.0,
        description="Non-negative observed weekly sales units.",
        examples=[14.0],
    )
    price: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional unit price observed during this week.",
        examples=[19.99],
    )


class VN1ForecastRequest(BaseModel):
    """Inference request for VN1 13-week forecasting."""

    series_id: str = Field(
        ...,
        description="Product-Warehouse series identifier (e.g. '0_1_367' or '0-1-367').",
        examples=["0_1_367"],
    )
    horizon: int = Field(
        default=13,
        ge=1,
        le=13,
        description="Forecast horizon in weekly steps (1 to 13 weeks).",
        examples=[13],
    )
    history: list[VN1WeeklyObservation] = Field(
        ...,
        min_length=12,
        max_length=250,
        description="Chronological weekly sales history (minimum 12 weeks required).",
    )

    @field_validator("series_id")
    @classmethod
    def clean_series_id(cls, v: str) -> str:
        cleaned = str(v).strip().replace("-", "_")
        if not cleaned:
            raise ValueError("series_id must not be empty.")
        return cleaned


class VN1ForecastItem(BaseModel):
    """Predicted sales for a single forward week."""

    week_start: str = Field(..., description="Target forecast week start date (YYYY-MM-DD).")
    horizon_step: int = Field(..., ge=1, le=13, description="Step index (1..13).")
    predicted_sales: float = Field(..., ge=0.0, description="Predicted sales units (clipped to non-negative).")


class VN1ForecastResponse(BaseModel):
    """Structured response from VN1 forecasting service."""

    forecast_run_id: str = Field(..., description="Unique UUID for this prediction run.")
    dataset_profile: str = "vn1"
    created_at: str = Field(..., description="UTC ISO timestamp of prediction generation.")
    series_id: str = Field(..., description="Product-Warehouse identifier.")
    model_version: str = Field(..., description="Active VN1 model release version.")
    model_name: str = Field(..., description="Estimator architecture name.")
    last_historical_date: str = Field(..., description="Most recent supplied history date.")
    forecast_start_date: str = Field(..., description="First forecast week start date.")
    forecast_end_date: str = Field(..., description="Last forecast week start date.")
    horizon: int = Field(..., description="Number of weekly steps predicted.")
    forecasts: list[VN1ForecastItem] = Field(..., description="List of weekly predictions.")
    warnings: list[str] = Field(default_factory=list, description="Data quality or stale model warnings.")
