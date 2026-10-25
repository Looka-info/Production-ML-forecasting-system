"""Training pipeline for DemandCast machine learning forecasting model."""

import logging
import time
from typing import Any

import pandas as pd

from demandcast.backtesting import ChronologicalBacktester
from demandcast.config import get_project_root, load_config
from demandcast.models import DemandForecastModel
from demandcast.tracking import log_experiment_run

logger = logging.getLogger(__name__)


def train_forecasting_model(
    weekly_df: pd.DataFrame | None = None,
    save_local_artifact: bool = True,
) -> tuple[DemandForecastModel, dict[str, Any], str]:
    """Train HistGradientBoostingRegressor model on historical observations prior to final test horizon.

    Chronological Safety:
    - Automatically isolates the final test horizon (latest 4 weeks).
    - Fits model strictly on observations <= test_origin_date.
    - Records training duration and logs experiment to MLflow.
    """
    config = load_config()
    root = get_project_root()

    if weekly_df is None:
        data_path = root / config.data.processed_weekly_path
        if not data_path.exists():
            raise FileNotFoundError(f"Processed weekly dataset not found at {data_path}. Run prepare_data.py first.")
        weekly_df = pd.read_parquet(data_path)

    # Generate backtest origins to isolate the final held-out test horizon
    backtester = ChronologicalBacktester(config.backtesting)
    val_origins, test_origin = backtester.generate_origins(weekly_df)

    # Train strictly on data prior to test_origin
    train_mask = weekly_df["week_start"] <= test_origin.origin_date
    train_df = weekly_df[train_mask].copy()

    training_date_range = {
        "min_date": str(train_df["week_start"].min().date()),
        "max_date": str(train_df["week_start"].max().date()),
    }

    logger.info(
        f"Training model on {len(train_df):,} weekly observations from "
        f"{training_date_range['min_date']} to {training_date_range['max_date']}..."
    )

    model = DemandForecastModel(
        model_config=config.model,
        features_config=config.features,
    )

    start_time = time.perf_counter()
    model.fit(train_df)
    training_duration_sec = round(time.perf_counter() - start_time, 2)
    logger.info(f"Model fitting took {training_duration_sec}s.")

    # In-sample validation metrics
    train_metrics = {
        "training_duration_seconds": training_duration_sec,
        "training_sample_count": len(train_df),
        "supported_series_count": len(model.supported_series),
    }

    # Log to MLflow
    params = {
        "model_type": model.name,
        "learning_rate": config.model.params.learning_rate,
        "max_iter": config.model.params.max_iter,
        "max_leaf_nodes": config.model.params.max_leaf_nodes,
        "min_samples_leaf": config.model.params.min_samples_leaf,
        "random_state": config.model.params.random_state,
        "lags": config.features.lags,
        "rolling_windows": config.features.rolling_windows,
        "training_min_date": training_date_range["min_date"],
        "training_max_date": training_date_range["max_date"],
    }

    mlflow_run_id = log_experiment_run(
        run_name="train_hist_gradient_boosting",
        params=params,
        metrics=train_metrics,
        tags={"stage": "training", "target": "weekly_demand"},
    )

    metadata = {
        "training_date_range": training_date_range,
        "training_metrics": train_metrics,
        "mlflow_run_id": mlflow_run_id,
        "supported_series": model.supported_series,
        "feature_cols": model.feature_cols,
    }

    return model, metadata, mlflow_run_id
