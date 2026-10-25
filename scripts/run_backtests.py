"""Script to run expanding-window chronological backtesting across baselines and ML model."""

import logging
import sys

from demandcast.evaluate import run_full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_backtests")


def main():
    logger.info("Executing expanding-window rolling-origin backtests...")
    try:
        comparison_df, predictions_df, summary = run_full_evaluation()
        logger.info("=" * 60)
        logger.info("Chronological Backtesting & Benchmark Results:")
        print("\n" + comparison_df.to_string(index=False) + "\n")
        logger.info(f"MLflow Run ID: {summary.get('mlflow_run_id')}")
        logger.info(f"Validation Origins: {summary.get('val_origins')}")
        logger.info(f"Held-out Test Origin: {summary.get('test_origin')}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Backtesting failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
