"""Comprehensive evaluation, backtesting, and model comparison across baselines and ML model."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from demandcast.backtesting import ChronologicalBacktester
from demandcast.baselines import (
    MovingAverageForecaster,
    NaivePersistenceForecaster,
    SeasonalNaiveForecaster,
)
from demandcast.config import get_project_root, load_config
from demandcast.models import DemandForecastModel
from demandcast.tracking import log_experiment_run

logger = logging.getLogger(__name__)


def generate_evaluation_figures(
    backtest_predictions_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate high-quality evaluation plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Plot actual vs predictions for top products
    sample_skus = backtest_predictions_df["series_id"].unique()[:3]
    fig, axes = plt.subplots(len(sample_skus), 1, figsize=(12, 4 * len(sample_skus)), sharex=False)
    if len(sample_skus) == 1:
        axes = [axes]

    for ax, sku in zip(axes, sample_skus):
        sub_df = backtest_predictions_df[backtest_predictions_df["series_id"] == sku].copy()
        sub_df = sub_df.sort_values(by="week_start")

        # Plot actuals once
        actuals_slice = sub_df[["week_start", "actual_demand"]].drop_duplicates().sort_values("week_start")
        ax.plot(
            actuals_slice["week_start"],
            actuals_slice["actual_demand"],
            label="Observed Demand (Actual)",
            color="black",
            linewidth=2.5,
            marker="o",
        )

        for m_name in sub_df["model_name"].unique():
            m_slice = sub_df[sub_df["model_name"] == m_name].sort_values("week_start")
            ax.plot(
                m_slice["week_start"],
                m_slice["predicted_demand"],
                label=f"Forecast: {m_name}",
                linestyle="--",
                marker="x",
                alpha=0.85,
            )

        ax.set_title(f"Demand Forecast vs Actuals — Product SKU: {sku}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Weekly Units Sold")
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig1_path = figures_dir / "forecast_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    generated_paths.append(fig1_path)

    # 2. Metric by Horizon Step (Error growth over 1..4 weeks)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    horizon_mae = (
        backtest_predictions_df.groupby(["model_name", "horizon_step"])
        .apply(lambda g: np.mean(np.abs(g["actual_demand"] - g["predicted_demand"])))
        .reset_index(name="mae")
    )
    sns.lineplot(
        data=horizon_mae,
        x="horizon_step",
        y="mae",
        hue="model_name",
        marker="o",
        ax=ax1,
    )
    ax1.set_title("MAE Growth by Forecast Horizon Step", fontweight="bold")
    ax1.set_xlabel("Horizon Step (Weeks Ahead)")
    ax1.set_ylabel("Mean Absolute Error (Units)")
    ax1.set_xticks([1, 2, 3, 4])

    # Horizon WAPE
    def calc_group_wape(g):
        denom = g["actual_demand"].sum()
        return (np.abs(g["actual_demand"] - g["predicted_demand"]).sum() / denom) if denom > 0 else 0.0

    horizon_wape = (
        backtest_predictions_df.groupby(["model_name", "horizon_step"]).apply(calc_group_wape).reset_index(name="wape")
    )
    sns.lineplot(
        data=horizon_wape,
        x="horizon_step",
        y="wape",
        hue="model_name",
        marker="s",
        ax=ax2,
    )
    ax2.set_title("WAPE Growth by Forecast Horizon Step", fontweight="bold")
    ax2.set_xlabel("Horizon Step (Weeks Ahead)")
    ax2.set_ylabel("Weighted Absolute % Error")
    ax2.set_xticks([1, 2, 3, 4])

    plt.tight_layout()
    fig2_path = figures_dir / "metrics_by_horizon.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    generated_paths.append(fig2_path)

    # 3. Residual Error Distribution for ML Model
    ml_preds = backtest_predictions_df[backtest_predictions_df["model_name"] == "HistGradientBoosting"].copy()
    if len(ml_preds) > 0:
        ml_preds["residual"] = ml_preds["predicted_demand"] - ml_preds["actual_demand"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(ml_preds["residual"], kde=True, ax=ax, color="#2b5c8f", bins=25)
        ax.axvline(0, color="red", linestyle="--", label="Zero Error (Unbiased)")
        ax.set_title("Forecast Residual Distribution (Predicted - Actual)", fontweight="bold")
        ax.set_xlabel("Forecast Residual (Units)")
        ax.legend()
        plt.tight_layout()
        fig3_path = figures_dir / "residual_distribution.png"
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        generated_paths.append(fig3_path)

    logger.info(f"Generated {len(generated_paths)} evaluation figures in {figures_dir}")
    return generated_paths


def run_full_evaluation(
    weekly_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run rolling-origin backtests and held-out test evaluation across all models.

    Models evaluated:
    1. Baseline_Naive (Persistence)
    2. Baseline_MovingAvg4 (4-week rolling average)
    3. Baseline_SeasonalNaive52 (52-week seasonal persistence)
    4. HistGradientBoosting (Supervised ML)
    """
    config = load_config()
    root = get_project_root()

    if weekly_df is None:
        data_path = root / config.data.processed_weekly_path
        weekly_df = pd.read_parquet(data_path)

    backtester = ChronologicalBacktester(config.backtesting)
    val_origins, test_origin = backtester.generate_origins(weekly_df)

    # Models to benchmark
    models = [
        NaivePersistenceForecaster(),
        MovingAverageForecaster(window=4),
        SeasonalNaiveForecaster(season_length=52),
        DemandForecastModel(model_config=config.model, features_config=config.features),
    ]

    all_val_preds = []
    comparison_records = []

    # 1. Validation Backtests
    for model in models:
        m_name = model.name
        preds_df, metrics_report = backtester.run_backtest_for_forecaster(
            forecaster_factory_or_instance=model,
            weekly_df=weekly_df,
            origins=val_origins,
            model_name=m_name,
        )
        all_val_preds.append(preds_df)

        ov = metrics_report.get("overall", {})
        comparison_records.append(
            {
                "model_name": m_name,
                "evaluation_stage": "validation_backtests",
                "origins_count": len(val_origins),
                "mae": ov.get("mae"),
                "rmse": ov.get("rmse"),
                "wape": ov.get("wape"),
                "smape": ov.get("smape"),
                "mase": ov.get("mase"),
                "bias": ov.get("bias"),
                "relative_bias": ov.get("relative_bias"),
                "sample_count": metrics_report.get("sample_size"),
            }
        )

    # 2. Final untouched test horizon evaluation
    all_test_preds = []
    for model in models:
        m_name = model.name
        preds_df, test_metrics = backtester.run_backtest_for_forecaster(
            forecaster_factory_or_instance=model,
            weekly_df=weekly_df,
            origins=[test_origin],
            model_name=m_name,
        )
        all_test_preds.append(preds_df)
        ov = test_metrics.get("overall", {})
        comparison_records.append(
            {
                "model_name": m_name,
                "evaluation_stage": "final_heldout_test",
                "origins_count": 1,
                "mae": ov.get("mae"),
                "rmse": ov.get("rmse"),
                "wape": ov.get("wape"),
                "smape": ov.get("smape"),
                "mase": ov.get("mase"),
                "bias": ov.get("bias"),
                "relative_bias": ov.get("relative_bias"),
                "sample_count": test_metrics.get("sample_size"),
            }
        )

    full_val_preds = pd.concat(all_val_preds, ignore_index=True)
    full_test_preds = pd.concat(all_test_preds, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_records)

    # Save reports
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"

    comp_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)

    backtest_path = reports_dir / "backtest_results.csv"
    full_val_preds.to_csv(backtest_path, index=False)

    test_preds_path = reports_dir / "test_horizon_results.csv"
    full_test_preds.to_csv(test_preds_path, index=False)

    # Generate figures
    fig_paths = generate_evaluation_figures(full_val_preds, figures_dir)

    # Log to MLflow
    mlflow_params = {
        "n_validation_origins": len(val_origins),
        "horizon": config.backtesting.horizon,
        "test_horizon": config.backtesting.test_horizon,
        "models_benchmarked": [m.name for m in models],
    }

    # Extract ML model validation metrics
    ml_val = (
        comparison_df[
            (comparison_df["model_name"] == "HistGradientBoosting")
            & (comparison_df["evaluation_stage"] == "validation_backtests")
        ]
        .iloc[0]
        .to_dict()
    )

    ml_metrics = {f"val_{k}": v for k, v in ml_val.items() if isinstance(v, (int, float)) and v is not None}

    art_dict = {
        "model_comparison": comp_path,
        "backtest_results": backtest_path,
    }
    for p in fig_paths:
        art_dict[p.stem] = p

    run_id = log_experiment_run(
        run_name="chronological_backtesting_and_benchmarks",
        params=mlflow_params,
        metrics=ml_metrics,
        artifacts=art_dict,
        tags={"stage": "evaluation", "benchmark": "baselines_vs_ml"},
    )

    summary = {
        "mlflow_run_id": run_id,
        "comparison": comparison_records,
        "val_origins": [{"index": o.origin_index, "date": str(o.origin_date.date())} for o in val_origins],
        "test_origin": {"date": str(test_origin.origin_date.date())},
    }

    logger.info("Chronological backtesting and benchmark evaluation complete.")
    return comparison_df, full_val_preds, summary
