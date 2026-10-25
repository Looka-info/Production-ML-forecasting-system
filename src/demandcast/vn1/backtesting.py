"""Chronological Backtesting Engine and Benchmark Evaluation for VN1 Retail Forecasting."""

import logging
from typing import Any

import pandas as pd

from demandcast.vn1.baselines import (
    VN1MovingAverageForecaster,
    VN1NaiveForecaster,
    VN1SeasonalNaiveForecaster,
)
from demandcast.vn1.config import VN1Config, load_vn1_config
from demandcast.vn1.metrics import (
    calculate_metrics_by_horizon,
    calculate_metrics_by_series,
    compute_all_vn1_metrics,
)
from demandcast.vn1.models import VN1Forecaster

logger = logging.getLogger(__name__)


def run_vn1_chronological_evaluation(
    cube: pd.DataFrame,
    config: VN1Config | None = None,
    include_ml_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute rigorous chronological validation on Phase 1 across baselines and ML candidates.

    - Phase 0 history: up to 2023-10-02 (170 weeks).
    - Phase 1 validation actuals: 2023-10-09 to 2024-01-01 (13 weeks).
    - All models are evaluated on strictly identical series and timestamps.
    """
    if config is None:
        config = load_vn1_config()

    origin_date = pd.to_datetime(config.validation.validation_origin)
    horizon = config.forecasting.horizon

    # Split history and validation actuals
    history_df = cube[cube["week_start"] <= origin_date].copy()
    val_actuals = cube[
        (cube["week_start"] > origin_date) & (cube["week_start"] <= origin_date + pd.Timedelta(weeks=horizon))
    ].copy()

    val_series = set(val_actuals["series_id"].unique())
    # Keep only series present in both history and validation actuals
    common_series = sorted(list(val_series.intersection(set(history_df["series_id"].unique()))))
    history_df = history_df[history_df["series_id"].isin(common_series)].copy()
    val_actuals = val_actuals[val_actuals["series_id"].isin(common_series)].copy()

    logger.info(
        f"VN1 Chronological Validation: origin={origin_date.strftime('%Y-%m-%d')}, "
        f"horizon={horizon} weeks, series_count={len(common_series)}, val_observations={len(val_actuals)}"
    )

    models_to_evaluate: list[Any] = [
        VN1NaiveForecaster(name="Baseline_Naive"),
        VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12_Official"),
        VN1MovingAverageForecaster(window_size=4, name="Baseline_MovingAvg4_DemandCast"),
        VN1SeasonalNaiveForecaster(season_length=52, name="Baseline_SeasonalNaive52"),
    ]

    trained_ml_models = {}
    if include_ml_models:
        logger.info("Training Global ML Candidate (NoPrice)...")
        ml_no_price = VN1Forecaster(include_price=False, name="LightGBM_NoPrice")
        ml_no_price.fit(history_df)
        models_to_evaluate.append(ml_no_price)
        trained_ml_models["LightGBM_NoPrice"] = ml_no_price

        logger.info("Training Global ML Candidate (PriceAware)...")
        ml_price = VN1Forecaster(include_price=True, name="LightGBM_PriceAware")
        ml_price.fit(history_df)
        models_to_evaluate.append(ml_price)
        trained_ml_models["LightGBM_PriceAware"] = ml_price

    all_predictions = []

    for model in models_to_evaluate:
        logger.info(f"Generating 13-week forecasts for {model.name}...")
        if hasattr(model, "predict_recursive"):
            preds = model.predict_recursive(history_df, horizon=horizon, origin_date=origin_date)
        else:
            preds = model.predict(history_df, horizon=horizon, origin_date=origin_date)
        all_predictions.append(preds)

    preds_df = pd.concat(all_predictions, ignore_index=True)

    # Merge predictions with validation ground-truth actuals
    val_eval = pd.merge(
        preds_df,
        val_actuals[["series_id", "week_start", "sales"]].rename(columns={"sales": "actual_sales"}),
        on=["series_id", "week_start"],
        how="inner",
    )

    # Compute comparison summary
    comparison_rows = []
    for model_name, group in val_eval.groupby("model_name"):
        metrics = compute_all_vn1_metrics(group["actual_sales"], group["predicted_sales"])
        metrics["model_name"] = model_name
        metrics["evaluation_stage"] = "Phase_1_Validation"
        metrics["forecast_origin"] = origin_date.strftime("%Y-%m-%d")
        metrics["horizon_weeks"] = horizon
        metrics["series_count"] = len(common_series)
        metrics["sample_count"] = len(group)
        comparison_rows.append(metrics)

    comp_df = pd.DataFrame(comparison_rows)
    # Sort by official VN1 Score (primary metric; lower is better)
    comp_df = comp_df.sort_values(by="vn1_score").reset_index(drop=True)

    # Mark winner
    winning_model_name = comp_df.iloc[0]["model_name"]
    comp_df["promotion_status"] = comp_df["model_name"].apply(
        lambda m: "Selected Production Strategy" if m == winning_model_name else "Candidate / Benchmark"
    )

    logger.info(
        f"Validation complete! Winning model: '{winning_model_name}' (Score: {comp_df.iloc[0]['vn1_score']:.4f})"
    )

    # Detailed horizon and series breakdowns
    horizon_metrics = calculate_metrics_by_horizon(val_eval)
    series_metrics = calculate_metrics_by_series(val_eval)

    return (
        comp_df,
        val_eval,
        {
            "winning_model_name": winning_model_name,
            "horizon_metrics": horizon_metrics,
            "series_metrics": series_metrics,
            "trained_models": trained_ml_models,
            "val_eval_df": val_eval,
        },
    )
