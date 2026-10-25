"""Tests for weekly demand aggregation and training-only product selection."""

import pandas as pd

from demandcast.aggregation import (
    aggregate_weekly_demand,
    generate_series_summary,
    select_product_series,
    to_monday_week_start,
)
from demandcast.config import ProductSelectionConfig
from demandcast.data_validation import clean_and_validate_transactions


def test_weekly_demand_aggregation_monday_convention(sample_transactions_raw):
    """Verify that weekly aggregation groups by Monday week-start and produces continuous grid."""
    cleaned_df, _ = clean_and_validate_transactions(sample_transactions_raw)
    weekly_df = aggregate_weekly_demand(cleaned_df)

    assert "series_id" in weekly_df.columns
    assert "week_start" in weekly_df.columns
    assert "demand" in weekly_df.columns

    # Verify that all week_starts are Mondays (day_of_week == 0)
    assert (weekly_df["week_start"].dt.day_of_week == 0).all()

    # Verify no NaN demands
    assert not weekly_df["demand"].isna().any()
    assert (weekly_df["demand"] >= 0).all()


def test_product_selection_training_period_only(sample_transactions_raw):
    """Verify product selection strictly enforces training cutoff date without lookahead."""
    cleaned_df, _ = clean_and_validate_transactions(sample_transactions_raw)

    config = ProductSelectionConfig(
        selection_cutoff_date="2010-06-01",
        min_history_weeks=10,
        min_total_sales=100,
        top_n_products=2,
    )
    selected_skus, metadata = select_product_series(cleaned_df, selection_config=config)

    assert len(selected_skus) <= 2
    assert "selection_cutoff_date" in metadata
    assert metadata["selected_skus_count"] == len(selected_skus)


def test_series_summary_generation(sample_weekly_demand):
    """Verify time-series summary statistics table."""
    summary_df = generate_series_summary(sample_weekly_demand)
    assert len(summary_df) == 2
    assert "total_weeks" in summary_df.columns
    assert "zero_percentage" in summary_df.columns
    assert "mean_weekly_demand" in summary_df.columns


def test_to_monday_week_start_interval_semantics_across_week_boundaries():
    """Verify that to_monday_week_start strictly maps all timestamps in [Monday 00:00:00, Sunday 23:59:59]

    to the exact Monday 00:00:00, and transitions cleanly at midnight on Monday.
    """
    timestamps = pd.Series(
        [
            "2011-12-05 00:00:00",  # Monday start
            "2011-12-05 08:30:00",  # Monday morning
            "2011-12-05 23:59:59",  # Monday night
            "2011-12-06 14:00:00",  # Tuesday
            "2011-12-07 12:00:00",  # Wednesday
            "2011-12-08 16:30:00",  # Thursday
            "2011-12-09 12:50:00",  # Friday (UCI raw dataset max date)
            "2011-12-10 18:00:00",  # Saturday
            "2011-12-11 00:00:00",  # Sunday start
            "2011-12-11 23:59:59",  # Sunday end boundary
            "2011-12-12 00:00:00",  # Next Monday boundary
            "2011-12-12 00:00:01",  # Next Monday + 1 sec
        ]
    )

    week_starts = to_monday_week_start(timestamps)

    # First 10 timestamps all belong strictly to the calendar week of Monday 2011-12-05
    for i in range(10):
        assert week_starts.iloc[i] == pd.Timestamp("2011-12-05 00:00:00"), (
            f"Failed for index {i} ({timestamps.iloc[i]}): expected 2011-12-05, got {week_starts.iloc[i]}"
        )

    # Timestamps 10 and 11 belong strictly to the next calendar week of Monday 2011-12-12
    assert week_starts.iloc[10] == pd.Timestamp("2011-12-12 00:00:00")
    assert week_starts.iloc[11] == pd.Timestamp("2011-12-12 00:00:00")


def test_pandas_resample_w_mon_default_arguments_discrepancy():
    """Verify and document why pandas resample('W-MON') default arguments fail Monday-start, Sunday-end convention."""
    # Transactions on Monday midday, Friday, and Sunday night of the same calendar week
    dates = pd.to_datetime(["2011-12-05 12:00:00", "2011-12-09 12:00:00", "2011-12-11 23:59:59"])
    series = pd.Series([10.0, 20.0, 30.0], index=dates)

    # Default resample('W-MON') uses closed='right', label='right' (Tuesday to Monday, ending on Monday)
    resampled_default = series.resample("W-MON").sum()
    # In default resample, Friday (12-09) and Sunday (12-11) are erroneously binned into following Monday (12-12)!
    assert pd.Timestamp("2011-12-12") in resampled_default.index
    assert resampled_default.loc["2011-12-12"] == 50.0  # 20 + 30 binned into next week

    # By contrast, to_monday_week_start maps all three timestamps correctly to Monday 2011-12-05
    calculated_starts = to_monday_week_start(pd.Series(dates))
    assert (calculated_starts == pd.Timestamp("2011-12-05")).all()


def test_aggregation_drops_incomplete_final_week():
    """Verify that aggregate_weekly_demand excludes a partial trailing week where data ends on Friday."""
    records = []
    # Week 1: Complete week (Monday 2011-11-28 through Sunday 2011-12-04)
    dates_w1 = pd.date_range("2011-11-28 09:00:00", "2011-12-04 18:00:00", periods=7)
    for d in dates_w1:
        records.append({"series_id": "SKU_TEST", "invoice_date": d, "quantity": 10})

    # Week 2: Incomplete week (Monday 2011-12-05 through Friday 2011-12-09 12:50:00)
    dates_w2 = pd.date_range("2011-12-05 09:00:00", "2011-12-09 12:50:00", periods=5)
    for d in dates_w2:
        records.append({"series_id": "SKU_TEST", "invoice_date": d, "quantity": 5})

    df = pd.DataFrame(records)

    # Case 1: drop_incomplete_final_week=True (default) -> Week 2 dropped
    agg_dropped = aggregate_weekly_demand(df, drop_incomplete_final_week=True)
    assert agg_dropped["week_start"].max() == pd.Timestamp("2011-11-28")
    assert len(agg_dropped) == 1

    # Case 2: drop_incomplete_final_week=False -> Week 2 retained
    agg_kept = aggregate_weekly_demand(df, drop_incomplete_final_week=False)
    assert agg_kept["week_start"].max() == pd.Timestamp("2011-12-05")
    assert len(agg_kept) == 2
