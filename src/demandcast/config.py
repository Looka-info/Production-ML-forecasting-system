"""Configuration loader and schema validation for DemandCast."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_url: str = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    raw_zip_path: str = "data/raw/online_retail_ii.zip"
    raw_excel_path: str = "data/raw/online_retail_II.xlsx"
    interim_clean_path: str = "data/interim/cleaned_transactions.parquet"
    processed_weekly_path: str = "data/processed/weekly_demand.parquet"
    selected_series_path: str = "data/processed/selected_series.json"
    quality_report_path: str = "reports/data_quality_report.json"
    summary_stats_path: str = "reports/series_summary.csv"


class CleaningConfig(BaseModel):
    min_quantity: int = 1
    min_unit_price: float = 0.001
    exclude_cancelled: bool = True
    non_merchandise_codes: list[str] = Field(
        default_factory=lambda: [
            "POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "TEST", "ADJUST", "ADJUST2"
        ]
    )
    week_start_day: str = "W-MON"


class ProductSelectionConfig(BaseModel):
    selection_cutoff_date: str = "2010-10-01"
    min_history_weeks: int = 35
    min_total_sales: int = 500
    top_n_products: int = 15


class FeaturesConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 8, 12])
    include_seasonal_lag: bool = True
    seasonal_lag: int = 52
    rolling_windows: list[int] = Field(default_factory=lambda: [4, 8, 12])
    rolling_stats: list[str] = Field(default_factory=lambda: ["mean", "std"])
    zero_demand_window: int = 4
    calendar: list[str] = Field(
        default_factory=lambda: ["week_of_year", "month", "sin_week", "cos_week"]
    )


class ForecastingConfig(BaseModel):
    horizon: int = 4
    min_history_required: int = 13
    nonnegative_clipping: bool = True


class BacktestingConfig(BaseModel):
    n_origins: int = 3
    horizon: int = 4
    test_horizon: int = 3
    step_size: int = 4


class ModelParamsConfig(BaseModel):
    learning_rate: float = 0.05
    max_iter: int = 150
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 5
    random_state: int = 42
    loss: str = "squared_error"


class ModelConfig(BaseModel):
    type: str = "HistGradientBoostingRegressor"
    params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)


class TrackingConfig(BaseModel):
    experiment_name: str = "demandcast_forecast_system"
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlruns"


class RegistryConfig(BaseModel):
    releases_dir: str = "models/releases"
    current_pointer: str = "models/current.json"


class MonitoringConfig(BaseModel):
    wape_alert_threshold: float = 0.50
    bias_alert_threshold: float = 0.25
    zero_demand_streak_warning: int = 3
    extreme_value_std_multiplier: float = 4.0


class GovernanceConfig(BaseModel):
    stale_model_threshold_days: int = 90
    wape_promotion_gate_threshold: float = 0.50
    allow_alert_promotion_override: bool = False
    dataset_vintage: str = "UCI Online Retail II (2009-2011)"


class Config(BaseModel):
    project_name: str = "DemandCast"
    version: str = "1.0.0"
    data: DataConfig = Field(default_factory=DataConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    product_selection: ProductSelectionConfig = Field(default_factory=ProductSelectionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    # src/demandcast/config.py -> parents[2] is project root
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from YAML file or return defaults."""
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        return Config()

    with open(target_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return Config(**raw_dict)
