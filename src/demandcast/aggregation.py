"""Weekly demand aggregation and training-only product selection for DemandCast."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from demandcast.config import (
    ProductSelectionConfig,
    load_config,
)

logger = logging.getLogger(__name__)


def to_monday_week_start(dates: pd.Series) -> pd.Series:
    """Map timestamps to the Monday (00:00:00) of their respective calendar weeks.

    Explicit Interval Semantics:
    Each calendar week is defined as [Monday 00:00:00, Sunday 23:59:59.999999] (or equivalently
    [Monday_t, Monday_{t+1})). Every timestamp within this interval is mapped strictly to Monday_t (00:00:00).

    Why pandas resample('W-MON') with default arguments differs:
    In pandas, `Series.resample('W-MON')` defaults to `closed='right', label='right'`. Under default
    resample, each weekly bin runs from (Tuesday 00:00:00, Monday 23:59:59] labeled by Monday.
    This splits a calendar week's observations across two periods (with Sunday mapped to the following Monday).
    `to_monday_week_start` computes the exact beginning of the Monday-Sunday interval directly:
    floor(timestamp to day) - day_of_week * 1 day.
    """
    dt = pd.to_datetime(dates)
    day_offset = dt.dt.day_of_week
    return dt.dt.floor("D") - pd.to_timedelta(day_offset, unit="D")


def select_product_series(
    transactions_df: pd.DataFrame,
    selection_config: ProductSelectionConfig | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Select candidate product series using the INITIAL TRAINING PERIOD ONLY.

    CRITICAL LEAKAGE PREVENTION:
    Products are filtered and ranked strictly on transactions occurring before
    `selection_cutoff_date`. Full-dataset sales totals are never used for selection,
    ensuring future volume does not influence SKU inclusion.
    """
    config = selection_config or load_config().product_selection
    cutoff = pd.to_datetime(config.selection_cutoff_date)

    # Restrict to training period only
    train_tx = transactions_df[transactions_df["invoice_date"] < cutoff].copy()
    if len(train_tx) == 0:
        raise ValueError(
            f"No transactions found before selection cutoff date {cutoff}. "
            f"Dataset date range: {transactions_df['invoice_date'].min()} to "
            f"{transactions_df['invoice_date'].max()}"
        )

    # Compute Monday week-start for training transactions
    train_tx["week_start"] = to_monday_week_start(train_tx["invoice_date"])

    # Calculate SKU statistics over training period
    sku_stats = (
        train_tx.groupby("series_id")
        .agg(
            total_train_quantity=("quantity", "sum"),
            train_active_weeks=("week_start", "nunique"),
            first_train_sale=("invoice_date", "min"),
            last_train_sale=("invoice_date", "max"),
        )
        .reset_index()
    )

    total_candidates = len(sku_stats)

    # Filter by minimum history and volume criteria
    eligible = sku_stats[
        (sku_stats["train_active_weeks"] >= config.min_history_weeks)
        & (sku_stats["total_train_quantity"] >= config.min_total_sales)
    ].copy()

    eligible_count = len(eligible)

    # Rank by total training quantity and select top N
    eligible = eligible.sort_values(by="total_train_quantity", ascending=False)
    selected_df = eligible.head(config.top_n_products)
    selected_skus = list(selected_df["series_id"].tolist())

    selection_metadata = {
        "selection_cutoff_date": str(cutoff),
        "total_sku_candidates": total_candidates,
        "eligible_skus_count": eligible_count,
        "selected_skus_count": len(selected_skus),
        "min_history_weeks_required": config.min_history_weeks,
        "min_train_sales_required": config.min_total_sales,
        "selected_series": [
            {
                "series_id": row["series_id"],
                "train_quantity": int(row["total_train_quantity"]),
                "train_active_weeks": int(row["train_active_weeks"]),
                "first_train_sale": str(row["first_train_sale"]),
                "last_train_sale": str(row["last_train_sale"]),
            }
            for _, row in selected_df.iterrows()
        ],
        "excluded_summary": {
            "failed_min_weeks": int((sku_stats["train_active_weeks"] < config.min_history_weeks).sum()),
            "failed_min_sales": int((sku_stats["total_train_quantity"] < config.min_total_sales).sum()),
        },
    }

    logger.info(
        f"Product selection complete: {len(selected_skus)} series selected from "
        f"{eligible_count} eligible series (out of {total_candidates} training candidates)."
    )
    return selected_skus, selection_metadata


def aggregate_weekly_demand(
    transactions_df: pd.DataFrame,
    selected_skus: list[str] | None = None,
    drop_incomplete_final_week: bool = True,
) -> pd.DataFrame:
    """Aggregate cleaned positive sales transactions into a continuous product-by-week dataset.

    Convention & Interval Semantics:
    - Weekly periods begin on Monday ('W-MON') and end on Sunday ('W-SUN').
    - If drop_incomplete_final_week is True (default), any trailing calendar week whose
      observations cut off before the complete Sunday 23:59:59 boundary is excluded so that
      aggregated demand quantities represent only complete 7-day calendar weeks.
    - For each selected SKU, missing weeks between the global first week and last week
      are filled with zero demand.
    - Zero demand represents observed zero sales (not necessarily zero customer demand,
      as out-of-stock events are unobserved).
    """
    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # If selected_skus provided, filter early
    if selected_skus is not None:
        df = df[df["series_id"].isin(selected_skus)].copy()

    # Assign Monday week-start date
    df["week_start"] = to_monday_week_start(df["invoice_date"])

    # Exclude incomplete trailing calendar week if requested
    if drop_incomplete_final_week and len(df) > 0:
        max_tx = df["invoice_date"].max()
        final_week_monday = to_monday_week_start(pd.Series([max_tx])).iloc[0]
        sunday_boundary = final_week_monday + pd.Timedelta(days=6)
        # If the latest transaction date cuts off before Sunday (day 6), the final week is partial/incomplete
        if max_tx.date() < sunday_boundary.date():
            logger.info(
                f"Excluding incomplete final week starting {final_week_monday.date()} "
                f"(transactions end at {max_tx}, prior to complete Sunday boundary {sunday_boundary.date()})."
            )
            df = df[df["week_start"] < final_week_monday].copy()

    # Sum positive sales quantity per series per week
    weekly_agg = df.groupby(["series_id", "week_start"]).agg(demand=("quantity", "sum")).reset_index()

    # Establish common global continuous calendar grid across all series
    min_week = weekly_agg["week_start"].min()
    max_week = weekly_agg["week_start"].max()
    all_weeks = pd.date_range(
        start=min_week,
        end=max_week,
        freq="W-MON",
    )

    skus = [str(s) for s in (selected_skus if selected_skus is not None else weekly_agg["series_id"].unique())]
    grid = (
        pd.MultiIndex.from_product(
            [skus, all_weeks],
            names=["series_id", "week_start"],
        )
        .to_frame()
        .reset_index(drop=True)
    )

    grid["series_id"] = grid["series_id"].astype(str)
    grid["week_start"] = pd.to_datetime(grid["week_start"])
    weekly_agg["series_id"] = weekly_agg["series_id"].astype(str)
    weekly_agg["week_start"] = pd.to_datetime(weekly_agg["week_start"])

    # Merge aggregated demand onto full grid and fill zero
    full_df = pd.merge(grid, weekly_agg, on=["series_id", "week_start"], how="left")
    full_df["demand"] = full_df["demand"].fillna(0.0).astype(float)

    # Sort strictly chronologically by series_id and week_start
    full_df = full_df.sort_values(by=["series_id", "week_start"]).reset_index(drop=True)
    full_df["week_start"] = pd.to_datetime(full_df["week_start"])

    return full_df


def generate_series_summary(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-series summary showing coverage, zero-demand weeks, and statistics."""
    summary_records = []
    for series_id, group in weekly_df.groupby("series_id"):
        total_weeks = len(group)
        zero_weeks = (group["demand"] == 0).sum()
        mean_demand = group["demand"].mean()
        std_demand = group["demand"].std()
        max_demand = group["demand"].max()
        min_demand = group["demand"].min()
        total_demand = group["demand"].sum()

        summary_records.append(
            {
                "series_id": series_id,
                "total_weeks": total_weeks,
                "zero_weeks": int(zero_weeks),
                "zero_percentage": round((zero_weeks / total_weeks) * 100, 2),
                "mean_weekly_demand": round(float(mean_demand), 2),
                "std_weekly_demand": round(float(std_demand), 2) if not np.isnan(std_demand) else 0.0,
                "min_demand": float(min_demand),
                "max_demand": float(max_demand),
                "total_demand": float(total_demand),
                "start_date": str(group["week_start"].min().date()),
                "end_date": str(group["week_start"].max().date()),
            }
        )

    summary_df = pd.DataFrame(summary_records).sort_values(by="total_demand", ascending=False)
    return summary_df
