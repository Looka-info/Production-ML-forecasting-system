"""Script to clean raw retail transactions, select products (training-only), and aggregate weekly demand."""

import json
import logging
import sys

from demandcast.aggregation import (
    aggregate_weekly_demand,
    generate_series_summary,
    select_product_series,
)
from demandcast.config import get_project_root, load_config
from demandcast.data_loading import load_raw_transactions
from demandcast.data_validation import clean_and_validate_transactions, save_data_quality_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def main():
    config = load_config()
    root = get_project_root()

    logger.info("Phase 1: Loading raw multi-sheet Excel transactions...")
    excel_path = root / config.data.raw_excel_path
    if not excel_path.exists():
        logger.error(f"Raw Excel file not found at {excel_path}. Please run 'python scripts/download_data.py' first.")
        sys.exit(1)

    raw_df, load_info = load_raw_transactions(excel_path)
    logger.info(f"Loaded {len(raw_df):,} raw transactions across sheets.")

    logger.info("Phase 2: Cleaning and validating transactions (excluding cancellations, returns, non-merchandise)...")
    cleaned_df, quality_report = clean_and_validate_transactions(raw_df, config.cleaning)

    # Save interim cleaned transactions
    interim_path = root / config.data.interim_clean_path
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(interim_path, index=False)
    logger.info(f"Saved interim cleaned transactions to {interim_path}")

    # Save quality report
    report_path = root / config.data.quality_report_path
    save_data_quality_report(quality_report, report_path)

    logger.info("Phase 3: Selecting top product series using TRAINING PERIOD ONLY (leakage-safe)...")
    selected_skus, selection_meta = select_product_series(cleaned_df, config.product_selection)

    # Save selected series
    selected_path = root / config.data.selected_series_path
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)
    logger.info(f"Saved selected series catalog ({len(selected_skus)} products) to {selected_path}")

    logger.info("Phase 4: Aggregating weekly demand on common Monday calendar grid...")
    weekly_df = aggregate_weekly_demand(cleaned_df, selected_skus=selected_skus)

    processed_path = root / config.data.processed_weekly_path
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_parquet(processed_path, index=False)
    logger.info(f"Saved processed weekly demand dataset to {processed_path} ({len(weekly_df):,} rows)")

    # Generate summary stats
    summary_df = generate_series_summary(weekly_df)
    summary_path = root / config.data.summary_stats_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved series summary statistics to {summary_path}")

    logger.info("=" * 60)
    logger.info("Data Preparation Complete:")
    logger.info(f" - Cleaned rows: {len(cleaned_df):,}")
    logger.info(f" - Selected products: {selected_skus}")
    logger.info(f" - Date span: {weekly_df['week_start'].min().date()} to {weekly_df['week_start'].max().date()}")
    logger.info(f" - Total weekly observations: {len(weekly_df):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
