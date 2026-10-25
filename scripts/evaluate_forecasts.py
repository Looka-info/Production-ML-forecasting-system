"""Script to evaluate post-forecast actuals, detect data drift, and trigger monitoring alerts."""

import argparse
import json
import logging
import sys

import pandas as pd

from demandcast.config import get_project_root
from demandcast.monitoring import ForecastMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_forecasts")


def main():
    parser = argparse.ArgumentParser(description="Evaluate stored forecasts against ground-truth actuals.")
    parser.add_argument("--model-name", type=str, default=None, help="Filter to specific model_name, or None for all.")
    parser.add_argument("--forecasts", type=str, default="reports/backtest_results.csv", help="Predictions CSV file.")
    parser.add_argument("--actuals", type=str, default="data/processed/weekly_demand.parquet", help="Actuals data file.")
    parser.add_argument("--output", type=str, default="reports/monitoring_report.json", help="Output report file.")
    args = parser.parse_args()

    root = get_project_root()
    forecasts_path = root / args.forecasts
    actuals_path = root / args.actuals

    if not forecasts_path.exists():
        logger.error(f"Forecasts file not found at {forecasts_path}")
        sys.exit(1)
    if not actuals_path.exists():
        logger.error(f"Actuals file not found at {actuals_path}")
        sys.exit(1)

    logger.info(f"Loading predictions from {forecasts_path} and actuals from {actuals_path}...")
    preds_df = pd.read_csv(forecasts_path)
    if actuals_path.suffix == ".parquet":
        actuals_df = pd.read_parquet(actuals_path)
    else:
        actuals_df = pd.read_csv(actuals_path)

    # Convert dates
    preds_df["week_start"] = pd.to_datetime(preds_df["week_start"])
    actuals_df["week_start"] = pd.to_datetime(actuals_df["week_start"])

    # If model_name specified, filter predictions
    if args.model_name and "model_name" in preds_df.columns:
        preds_df = preds_df[preds_df["model_name"] == args.model_name]
    elif "model_name" in preds_df.columns and "Baseline_MovingAvg4" in preds_df["model_name"].values:
        # Default to evaluated active strategy
        preds_df = preds_df[preds_df["model_name"] == "Baseline_MovingAvg4"]

    monitor = ForecastMonitor()
    report = monitor.evaluate_forecast_actuals(preds_df, actuals_df)

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Forecast Quality & Monitoring Report saved to {out_path}:")
    logger.info(f"Status: {report.get('status')}")
    logger.info(f"Evaluated Observations: {report.get('evaluated_observations')}")
    logger.info(f"Metrics: {report.get('metrics')}")
    logger.info(f"Alerts Triggered: {report.get('alerts')}")
    logger.info(f"Drift Diagnostic: {report.get('drift_diagnostic')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
