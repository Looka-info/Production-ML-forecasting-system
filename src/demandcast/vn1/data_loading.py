"""Data loading, schema validation, and cube processing for VN1 Retail Challenge."""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from demandcast.config import get_project_root
from demandcast.vn1.config import VN1Config, load_vn1_config

logger = logging.getLogger(__name__)


def format_series_id(client: int | str, warehouse: int | str, product: int | str) -> str:
    """Standardized series identifier string: {client}_{warehouse}_{product}."""
    return f"{client}_{warehouse}_{product}"


def parse_wide_to_long(
    df_wide: pd.DataFrame,
    value_name: str,
    phase: int,
) -> pd.DataFrame:
    """Convert wide VN1 format (dates as columns) to canonical long time-series format."""
    id_cols = ["Client", "Warehouse", "Product"]
    for col in id_cols:
        if col not in df_wide.columns:
            raise ValueError(f"Missing required identity column: '{col}'")

    date_cols = [c for c in df_wide.columns if c not in id_cols]
    if not date_cols:
        raise ValueError("No date columns found in wide DataFrame.")

    # Unpivot / melt
    df_long = pd.melt(
        df_wide,
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="week_start_str",
        value_name=value_name,
    )

    df_long["week_start"] = pd.to_datetime(df_long["week_start_str"].str.strip())
    df_long["phase"] = phase
    df_long["series_id"] = (
        df_long["Client"].astype(str) + "_" + df_long["Warehouse"].astype(str) + "_" + df_long["Product"].astype(str)
    )
    df_long["client_id"] = df_long["Client"].astype(str)
    df_long["warehouse_id"] = df_long["Warehouse"].astype(str)
    df_long["product_id"] = df_long["Product"].astype(str)

    df_long = df_long.drop(columns=["week_start_str", "Client", "Warehouse", "Product"])
    return df_long


def load_vn1_phase_raw(
    phase: int,
    data_type: str = "sales",
    config: VN1Config | None = None,
) -> pd.DataFrame:
    """Load raw Phase 0 or Phase 1 CSV file."""
    if config is None:
        config = load_vn1_config()
    root = get_project_root()

    if data_type.lower() == "sales":
        path_str = config.data.phase0_sales_path if phase == 0 else config.data.phase1_sales_path
    elif data_type.lower() == "price":
        path_str = config.data.phase0_price_path if phase == 0 else config.data.phase1_price_path
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Must be 'sales' or 'price'.")

    csv_path = root / path_str
    if not csv_path.exists():
        raise FileNotFoundError(
            f"VN1 data file not found at {csv_path}. Run 'python scripts/download_vn1_data.py' first."
        )

    logger.info(f"Reading Phase {phase} {data_type} from {csv_path}...")
    df_wide = pd.read_csv(csv_path)
    return parse_wide_to_long(df_wide, value_name=data_type, phase=phase)


def load_vn1_cube(
    config: VN1Config | None = None,
    subset_n_series: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build unified, merged long sales and price cube across Phase 0 and Phase 1."""
    if config is None:
        config = load_vn1_config()
    root = get_project_root()

    # 1. Load Sales
    s0 = load_vn1_phase_raw(0, "sales", config)
    s1 = load_vn1_phase_raw(1, "sales", config)
    sales = pd.concat([s0, s1], ignore_index=True)

    # 2. Load Prices
    p0 = load_vn1_phase_raw(0, "price", config)
    p1 = load_vn1_phase_raw(1, "price", config)
    prices = pd.concat([p0, p1], ignore_index=True)

    # 3. Merge sales and prices
    logger.info("Merging sales and price cubes on ['series_id', 'week_start']...")
    cube = pd.merge(
        sales,
        prices[["series_id", "week_start", "price"]],
        on=["series_id", "week_start"],
        how="left",
    )

    # Ensure float types and sort chronologically
    cube["sales"] = cube["sales"].astype(np.float32)
    cube["price"] = cube["price"].astype(np.float32)
    cube = cube.sort_values(by=["series_id", "week_start"]).reset_index(drop=True)

    # Compute data quality statistics
    total_rows = len(cube)
    unique_series = int(cube["series_id"].nunique())
    unique_clients = int(cube["client_id"].nunique())
    unique_warehouses = int(cube["warehouse_id"].nunique())
    unique_products = int(cube["product_id"].nunique())
    min_date = str(cube["week_start"].min().strftime("%Y-%m-%d"))
    max_date = str(cube["week_start"].max().strftime("%Y-%m-%d"))
    total_weeks = int(cube["week_start"].nunique())

    zero_sales_count = int((cube["sales"] == 0).sum())
    zero_sales_pct = round(100.0 * zero_sales_count / total_rows, 2)
    negative_sales_count = int((cube["sales"] < 0).sum())
    missing_price_count = int(cube["price"].isna().sum())
    missing_price_pct = round(100.0 * missing_price_count / total_rows, 2)

    quality_report = {
        "dataset_name": "VN1 Forecasting Challenge",
        "total_rows": total_rows,
        "unique_series_count": unique_series,
        "unique_clients_count": unique_clients,
        "unique_warehouses_count": unique_warehouses,
        "unique_products_count": unique_products,
        "total_weeks_count": total_weeks,
        "date_range": {"min_date": min_date, "max_date": max_date},
        "phase0_weeks": int(cube[cube["phase"] == 0]["week_start"].nunique()),
        "phase1_weeks": int(cube[cube["phase"] == 1]["week_start"].nunique()),
        "zero_sales_frequency_pct": zero_sales_pct,
        "negative_sales_rows": negative_sales_count,
        "missing_price_count": missing_price_count,
        "missing_price_pct": missing_price_pct,
        "mean_sales_units": round(float(cube["sales"].mean()), 2),
        "median_sales_units": round(float(cube["sales"].median()), 2),
        "p95_sales_units": round(float(cube["sales"].quantile(0.95)), 2),
        "max_sales_units": round(float(cube["sales"].max()), 2),
    }

    report_path = root / config.data.quality_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"VN1 Data quality report saved to {report_path}")

    # Optionally filter to representative subset for fast dev/testing if requested
    if subset_n_series is not None and subset_n_series < unique_series:
        logger.info(
            f"Filtering to top {subset_n_series} representative series strictly based on Phase 0 total sales..."
        )
        # Select using Phase 0 sales only (NO lookahead leakage into Phase 1 or 2)
        p0_sales = cube[cube["phase"] == 0].groupby("series_id")["sales"].sum().reset_index()
        top_series = p0_sales.sort_values(by="sales", ascending=False).head(subset_n_series)["series_id"].tolist()
        cube = cube[cube["series_id"].isin(top_series)].copy().reset_index(drop=True)
        quality_report["active_eval_series_count"] = len(top_series)
        quality_report["selection_protocol"] = (
            f"Top {subset_n_series} volume series selected strictly using Phase 0 (pre-2023-10-09) observations"
        )
    else:
        quality_report["active_eval_series_count"] = unique_series
        quality_report["selection_protocol"] = "Full benchmark evaluation (all 15,053 series)"

    return cube, quality_report
