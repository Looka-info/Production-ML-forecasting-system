"""Data cleaning, schema validation, and quality reporting for DemandCast."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from demandcast.config import CleaningConfig, get_project_root, load_config

logger = logging.getLogger(__name__)


def clean_and_validate_transactions(
    df: pd.DataFrame,
    cleaning_config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw retail transactions according to auditable business rules.

    Cleaning Rules:
    1. Parse and validate invoice_date to datetime (UTC).
    2. Identify and flag cancelled invoices (Invoice ID starting with 'C').
    3. Exclude cancellations and returns from positive sales demand.
    4. Exclude nonpositive quantities (<= 0) and invalid prices (<= 0).
    5. Exclude non-merchandise service codes (e.g., POST, BANK CHARGES, D, M).
    6. Standardize product identifiers (stock_code / series_id) to stripped uppercase string.
    7. Remove rows with missing or empty stock_code or invoice_date.
    8. Deduplicate exact duplicate transaction rows.
    """
    config = cleaning_config or load_config().cleaning

    initial_row_count = len(df)
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "initial_raw_rows": initial_row_count,
        "exclusions": {},
        "cleaning_rules": config.model_dump(),
    }

    # Work on a copy
    cleaned = df.copy()

    # 1. Check required columns
    required_cols = {"invoice", "stock_code", "quantity", "invoice_date", "unit_price"}
    missing_cols = required_cols - set(cleaned.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in transactions DataFrame: {missing_cols}")

    # 2. Missing stock_code or invoice_date
    null_stock = cleaned["stock_code"].isna().sum()
    null_date = cleaned["invoice_date"].isna().sum()
    cleaned = cleaned.dropna(subset=["stock_code", "invoice_date"])
    report["exclusions"]["null_stock_code_or_date"] = int(null_stock + null_date)

    # 3. Standardize stock_code string
    cleaned["stock_code"] = cleaned["stock_code"].astype(str).str.strip().str.upper()
    cleaned = cleaned[cleaned["stock_code"] != ""]
    cleaned = cleaned[cleaned["stock_code"] != "NAN"]

    # 4. Parse invoice_date
    cleaned["invoice_date"] = pd.to_datetime(cleaned["invoice_date"], errors="coerce")
    invalid_dates = cleaned["invoice_date"].isna().sum()
    cleaned = cleaned.dropna(subset=["invoice_date"])
    report["exclusions"]["invalid_date_parse"] = int(invalid_dates)

    # 5. Identify and flag cancellations (Invoice starts with 'C')
    cleaned["invoice_str"] = cleaned["invoice"].astype(str).str.strip()
    is_cancelled = cleaned["invoice_str"].str.upper().str.startswith("C")
    cancelled_count = int(is_cancelled.sum())
    report["cancellations_count"] = cancelled_count

    if config.exclude_cancelled:
        cleaned = cleaned[~is_cancelled]
        report["exclusions"]["cancelled_invoices"] = cancelled_count

    # 6. Exclude non-positive quantities (sales proxy must be positive observed sales)
    non_positive_qty = int((cleaned["quantity"] < config.min_quantity).sum())
    cleaned = cleaned[cleaned["quantity"] >= config.min_quantity]
    report["exclusions"]["non_positive_quantity"] = non_positive_qty

    # 7. Exclude invalid unit prices (<= 0)
    invalid_price = int((cleaned["unit_price"] < config.min_unit_price).sum())
    cleaned = cleaned[cleaned["unit_price"] >= config.min_unit_price]
    report["exclusions"]["invalid_unit_price"] = invalid_price

    # 8. Exclude non-merchandise codes
    non_merch_mask = cleaned["stock_code"].isin([code.upper() for code in config.non_merchandise_codes])
    non_merch_count = int(non_merch_mask.sum())
    cleaned = cleaned[~non_merch_mask]
    report["exclusions"]["non_merchandise_codes"] = non_merch_count

    # 9. Deduplicate exact duplicate rows
    dup_cols = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price"]
    available_dup_cols = [c for c in dup_cols if c in cleaned.columns]
    duplicates = int(cleaned.duplicated(subset=available_dup_cols).sum())
    cleaned = cleaned.drop_duplicates(subset=available_dup_cols)
    report["exclusions"]["duplicate_rows"] = duplicates

    # Cleaned DataFrame fields
    cleaned = cleaned.rename(columns={"stock_code": "series_id"})

    # Drop customer identifier and raw temporary columns to preserve privacy and prevent leakage
    drop_cols = [c for c in ["customer_id", "invoice_str"] if c in cleaned.columns]
    cleaned = cleaned.drop(columns=drop_cols)

    # Standardize string datatypes for parquet compatibility
    for col in ["invoice", "series_id", "description", "country"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str)

    final_row_count = len(cleaned)
    report["final_cleaned_rows"] = final_row_count
    report["retention_rate_pct"] = round((final_row_count / initial_row_count) * 100, 2)
    report["date_range"] = {
        "min_date": str(cleaned["invoice_date"].min()),
        "max_date": str(cleaned["invoice_date"].max()),
    }
    report["unique_products"] = int(cleaned["series_id"].nunique())
    report["total_units_sold"] = int(cleaned["quantity"].sum())

    logger.info(
        f"Cleaned transactions: {final_row_count:,} / {initial_row_count:,} rows retained "
        f"({report['retention_rate_pct']}%). Unique products: {report['unique_products']:,}"
    )

    return cleaned, report


def save_data_quality_report(report: dict[str, Any], filepath: Path | None = None) -> None:
    """Save the auditable data quality report to disk as JSON."""
    if filepath is None:
        filepath = get_project_root() / "reports" / "data_quality_report.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved data quality report to {filepath}")
