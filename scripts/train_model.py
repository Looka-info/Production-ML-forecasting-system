"""Script to train candidate ML forecasting model, evaluate selection rules, and package releases."""

import argparse
import logging
import shutil
import sys

import pandas as pd

from demandcast.baselines import MovingAverageForecaster
from demandcast.config import get_project_root, load_config
from demandcast.evaluate import run_full_evaluation
from demandcast.registry import ModelRegistry
from demandcast.train import train_forecasting_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_model")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and package DemandCast model releases.")
    parser.add_argument("--candidate-version", type=str, default="v1.0.0-candidate-hgb", help="Candidate version string.")
    parser.add_argument("--prod-version", type=str, default="v1.0.0", help="Production release version string.")
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    data_path = root / config.data.processed_weekly_path

    if not data_path.exists():
        logger.error(f"Processed dataset not found at {data_path}. Run prepare_data.py first.")
        sys.exit(1)

    logger.info(f"Loading processed weekly data from {data_path}...")
    weekly_df = pd.read_parquet(data_path)

    # 1. Run backtests across all models
    logger.info("Evaluating all candidate models via chronological rolling-origin backtests...")
    comparison_df, _, eval_summary = run_full_evaluation(weekly_df)

    registry = ModelRegistry()

    # 2. Extract metrics for HistGradientBoosting candidate
    ml_val_row = comparison_df[
        (comparison_df["model_name"] == "HistGradientBoosting")
        & (comparison_df["evaluation_stage"] == "validation_backtests")
    ].iloc[0].to_dict()

    ml_test_row = comparison_df[
        (comparison_df["model_name"] == "HistGradientBoosting")
        & (comparison_df["evaluation_stage"] == "final_heldout_test")
    ].iloc[0].to_dict()

    # Train final ML model instance
    logger.info("Training candidate HistGradientBoosting artifact...")
    ml_model, train_meta, run_id = train_forecasting_model(weekly_df)

    # Package HistGradientBoosting as candidate release
    candidate_dir = registry.releases_dir / args.candidate_version
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    logger.info(f"Packaging ML candidate release '{args.candidate_version}' in registry...")
    registry.package_release(
        model=ml_model,
        version=args.candidate_version,
        mlflow_run_id=run_id,
        validation_metrics=ml_val_row,
        test_metrics=ml_test_row,
        training_date_range=train_meta["training_date_range"],
        notes="Tracked supervised ML candidate (HistGradientBoosting). Retained in registry for ongoing evaluation.",
    )

    # 3. Model Selection: Select winning strategy strictly on validation performance
    val_table = comparison_df[comparison_df["evaluation_stage"] == "validation_backtests"]
    best_model_row = val_table.sort_values(by="wape").iloc[0]
    best_model_name = best_model_row["model_name"]
    logger.info("=" * 60)
    logger.info("Model Selection Audit: Primary Metric = WAPE (Validation)")
    logger.info(f"Winning Strategy: {best_model_name} (Validation WAPE: {best_model_row['wape']:.2%}, MASE: {best_model_row['mase']})")
    logger.info(f"HistGradientBoosting: (Validation WAPE: {ml_val_row['wape']:.2%}, MASE: {ml_val_row['mase']})")
    logger.info("=" * 60)

    # Extract metrics for winning MovingAverage baseline
    ma_val_row = comparison_df[
        (comparison_df["model_name"] == "Baseline_MovingAvg4")
        & (comparison_df["evaluation_stage"] == "validation_backtests")
    ].iloc[0].to_dict()

    ma_test_row = comparison_df[
        (comparison_df["model_name"] == "Baseline_MovingAvg4")
        & (comparison_df["evaluation_stage"] == "final_heldout_test")
    ].iloc[0].to_dict()

    # Package winning model as active production release
    prod_dir = registry.releases_dir / args.prod_version
    if prod_dir.exists():
        shutil.rmtree(prod_dir)

    prod_model = MovingAverageForecaster(window=4)
    train_slice = weekly_df[weekly_df["week_start"] <= pd.to_datetime(train_meta["training_date_range"]["max_date"])]
    prod_model.fit(train_slice)

    logger.info(f"Packaging production release '{args.prod_version}' ({best_model_name}) in registry...")
    registry.package_release(
        model=prod_model,
        version=args.prod_version,
        mlflow_run_id=eval_summary.get("mlflow_run_id", "baseline_backtests"),
        validation_metrics=ma_val_row,
        test_metrics=ma_test_row,
        training_date_range=train_meta["training_date_range"],
        notes=(
            f"Production release {args.prod_version}. Active strategy: 4-week moving average baseline. "
            f"Promoted strictly based on superior validation performance (WAPE: {ma_val_row.get('wape'):.2%}, "
            f"MASE: {ma_val_row.get('mase')}) over ML candidate (WAPE: {ml_val_row.get('wape'):.2%}, "
            f"MASE: {ml_val_row.get('mase')})."
        ),
    )

    logger.info("Releases packaged successfully.")


if __name__ == "__main__":
    main()
