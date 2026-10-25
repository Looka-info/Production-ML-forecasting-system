"""VN1 Configuration loader and schema validation for DemandCast v2."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from demandcast.config import get_project_root


class VN1DataConfig(BaseModel):
    raw_dir: str = "data/vn1/raw"
    phase0_sales_path: str = "data/vn1/raw/Phase 0 - Sales.csv"
    phase0_price_path: str = "data/vn1/raw/Phase 0 - Price.csv"
    phase1_sales_path: str = "data/vn1/raw/Phase 1 - Sales.csv"
    phase1_price_path: str = "data/vn1/raw/Phase 1 - Price.csv"
    interim_cube_path: str = "data/vn1/interim/cube.parquet"
    processed_sales_path: str = "data/vn1/processed/weekly_sales.parquet"
    processed_prices_path: str = "data/vn1/processed/weekly_prices.parquet"
    quality_report_path: str = "reports/vn1/data_quality_report.json"
    provenance_path: str = "data/vn1/raw/dataset_provenance.json"


class VN1TemporalConfig(BaseModel):
    frequency: str = "W-MON"
    phase0_start: str = "2020-07-06"
    phase0_end: str = "2023-10-02"
    phase1_start: str = "2023-10-09"
    phase1_end: str = "2024-01-01"
    phase2_start: str = "2024-01-08"
    phase2_end: str = "2024-04-01"


class VN1ForecastingConfig(BaseModel):
    horizon: int = 13
    min_history_required: int = 12
    nonnegative_clipping: bool = True
    supported_series_subset_for_dev: int = 250


class VN1FeaturesConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 8, 12, 13])
    rolling_windows: list[int] = Field(default_factory=lambda: [4, 8, 12])
    rolling_stats: list[str] = Field(default_factory=lambda: ["mean", "std"])
    zero_demand_window: int = 4
    include_price_features: bool = True
    categorical_columns: list[str] = Field(default_factory=lambda: ["client_id", "warehouse_id", "product_id"])


class VN1ValidationConfig(BaseModel):
    validation_origin: str = "2023-10-02"
    validation_horizon: int = 13
    phase2_origin: str = "2024-01-01"
    phase2_horizon: int = 13


class VN1ModelConfig(BaseModel):
    type: str = "LightGBM"
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 120,
            "min_child_samples": 15,
            "random_state": 42,
            "verbose": -1,
        }
    )


class VN1TrackingConfig(BaseModel):
    experiment_name: str = "demandcast-vn1"
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlruns"


class VN1RegistryConfig(BaseModel):
    releases_dir: str = "models/profiles/vn1/releases"
    current_pointer: str = "models/profiles/vn1/current.json"


class VN1GovernanceConfig(BaseModel):
    primary_metric: str = "vn1_score"
    target_gate_score: float = 0.70
    allow_alert_promotion_override: bool = False
    training_cutoff_date: str = "2023-10-02"
    dataset_vintage: str = "VN1 Forecasting Challenge (July 2020 - April 2024)"
    dataset_doi: str = "10.5281/zenodo.23265413"
    competition_url: str = "https://supchains.com/vn1-forecasting-competition/"
    stale_model_threshold_days: int = 90


class VN1ShadowConfig(BaseModel):
    enabled: bool = True
    db_path: str = "data/shadow/shadow_store.db"
    reports_dir: str = "reports/vn1/shadow"
    default_provider: str = "file_batch"
    shrinkage_k: float = 100.0
    min_history_required: int = 12


class VN1Config(BaseModel):
    profile_name: str = "vn1"
    version: str = "1.0.0"
    data: VN1DataConfig = Field(default_factory=VN1DataConfig)
    temporal: VN1TemporalConfig = Field(default_factory=VN1TemporalConfig)
    forecasting: VN1ForecastingConfig = Field(default_factory=VN1ForecastingConfig)
    features: VN1FeaturesConfig = Field(default_factory=VN1FeaturesConfig)
    validation: VN1ValidationConfig = Field(default_factory=VN1ValidationConfig)
    model: VN1ModelConfig = Field(default_factory=VN1ModelConfig)
    tracking: VN1TrackingConfig = Field(default_factory=VN1TrackingConfig)
    registry: VN1RegistryConfig = Field(default_factory=VN1RegistryConfig)
    governance: VN1GovernanceConfig = Field(default_factory=VN1GovernanceConfig)
    shadow: VN1ShadowConfig = Field(default_factory=VN1ShadowConfig)


def load_vn1_config(config_path: Path | str | None = None) -> VN1Config:
    """Load and parse VN1 YAML configuration."""
    if config_path is None:
        root = get_project_root()
        config_path = root / "configs" / "vn1_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return VN1Config()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return VN1Config(**data)
