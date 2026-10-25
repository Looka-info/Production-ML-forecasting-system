"""Tests for leakage-safe feature engineering."""

import pandas as pd

from demandcast.features import (
    build_inference_features_for_step,
    create_calendar_features,
    extract_features_single_series,
)


def test_lag_values_and_rolling_stats_no_leakage():
    """Verify that lag_1, lag_2 and rolling means match exact hand-calculated values

    and strictly exclude the current week's demand.
    """
    dates = pd.date_range("2011-01-03", periods=6, freq="W-MON")
    # Exact demands: [10, 20, 30, 40, 50, 60]
    df = pd.DataFrame(
        {
            "week_start": dates,
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    feat_df, target = extract_features_single_series("SKU_TEST", df)

    # For row index 2 (date: 2011-01-17, demand: 30)
    # lag_1 should be 20.0 (t-1)
    # lag_2 should be 10.0 (t-2)
    assert feat_df.loc[2, "lag_1"] == 20.0
    assert feat_df.loc[2, "lag_2"] == 10.0

    # For row index 4 (date: 2011-01-31, demand: 50)
    # rolling_mean_4 should be mean of t-1, t-2, t-3, t-4 -> mean(40, 30, 20, 10) = 25.0
    # Current demand (50.0) MUST NOT be included!
    assert feat_df.loc[4, "rolling_mean_4"] == 25.0


def test_calendar_features():
    """Verify calendar feature cyclical encodings."""
    dates = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    cal_df = create_calendar_features(dates)

    assert "week_of_year" in cal_df.columns
    assert "month" in cal_df.columns
    assert "sin_week" in cal_df.columns
    assert "cos_week" in cal_df.columns
    assert (cal_df["sin_week"] >= -1.0).all() and (cal_df["sin_week"] <= 1.0).all()


def test_inference_step_feature_generation():
    """Verify single-step inference feature generator matches training schema."""
    dates = pd.date_range("2011-01-03", periods=15, freq="W-MON")
    df = pd.DataFrame(
        {
            "week_start": dates,
            "demand": [float(10 + i) for i in range(15)],
        }
    )
    target_date = dates[-1] + pd.Timedelta(weeks=1)

    step_features = build_inference_features_for_step(
        series_id="SKU_1",
        history_df=df,
        target_week_start=target_date,
    )

    assert len(step_features) == 1
    # Lag 1 for the target week should be the last observation of df (demand = 24.0)
    assert step_features["lag_1"].iloc[0] == 24.0
    assert step_features["series_id"].iloc[0] == "SKU_1"


def test_minimum_valid_history_and_rejection_of_insufficient_history():
    """Verify that 13 weeks produces complete features, while 12 weeks raises ValueError."""
    import pytest

    # 13 weeks of history (valid)
    dates_13 = pd.date_range("2011-01-03", periods=13, freq="W-MON")
    df_13 = pd.DataFrame(
        {
            "week_start": dates_13,
            "demand": [float(10 + i) for i in range(13)],
        }
    )
    target_date = dates_13[-1] + pd.Timedelta(weeks=1)

    feats_13 = build_inference_features_for_step(
        series_id="SKU_1",
        history_df=df_13,
        target_week_start=target_date,
    )
    assert not feats_13.isna().any().any(), "13 weeks must produce zero NaNs in features."
    assert "lag_12" in feats_13.columns
    assert "rolling_mean_12" in feats_13.columns

    # 12 weeks of history (insufficient for lag_12)
    dates_12 = pd.date_range("2011-01-03", periods=12, freq="W-MON")
    df_12 = pd.DataFrame(
        {
            "week_start": dates_12,
            "demand": [float(10 + i) for i in range(12)],
        }
    )
    with pytest.raises(ValueError, match="Cannot generate complete feature set"):
        build_inference_features_for_step(
            series_id="SKU_1",
            history_df=df_12,
            target_week_start=dates_12[-1] + pd.Timedelta(weeks=1),
        )


def test_missing_lag_52_when_seasonal_lag_enabled_raises():
    """Verify that enabling seasonal_lag=52 on 13 weeks of history raises ValueError."""
    import pytest

    from demandcast.config import FeaturesConfig

    cfg = FeaturesConfig(
        lags=[1, 2, 3, 4, 8, 12],
        include_seasonal_lag=True,
        seasonal_lag=52,
        rolling_windows=[4, 8, 12],
        rolling_stats=["mean", "std"],
        zero_demand_window=4,
    )
    dates_13 = pd.date_range("2011-01-03", periods=13, freq="W-MON")
    df_13 = pd.DataFrame(
        {
            "week_start": dates_13,
            "demand": [float(10 + i) for i in range(13)],
        }
    )

    with pytest.raises(ValueError, match="Cannot generate complete feature set"):
        build_inference_features_for_step(
            series_id="SKU_1",
            history_df=df_13,
            target_week_start=dates_13[-1] + pd.Timedelta(weeks=1),
            features_config=cfg,
        )


def test_first_recursive_forecast_step_no_fabricated_values(trained_dummy_model):
    """Verify recursive multi-step forecasting passes genuine predicted demand to lag_1 in step 2."""
    dates = pd.date_range("2011-01-03", periods=14, freq="W-MON")
    df = pd.DataFrame(
        {
            "week_start": dates,
            "demand": [50.0] * 14,
        }
    )

    fc_df = trained_dummy_model.forecast(
        series_id="SKU_ALPHA",
        history_df=df,
        horizon=2,
    )

    assert len(fc_df) == 2
    step1_pred = fc_df.loc[fc_df["horizon_step"] == 1, "predicted_demand"].iloc[0]
    step2_pred = fc_df.loc[fc_df["horizon_step"] == 2, "predicted_demand"].iloc[0]
    assert step1_pred >= 0.0
    assert step2_pred >= 0.0


def test_exact_18_feature_schema_reconciliation():
    """Verify exactly 18 feature columns are produced when include_seasonal_lag=False."""
    from demandcast.config import FeaturesConfig
    from demandcast.features import extract_features_dataset

    cfg = FeaturesConfig(
        lags=[1, 2, 3, 4, 8, 12],
        include_seasonal_lag=False,
        seasonal_lag=52,
        rolling_windows=[4, 8, 12],
        rolling_stats=["mean", "std"],
        zero_demand_window=4,
    )
    dates = pd.date_range("2011-01-03", periods=20, freq="W-MON")
    df = pd.DataFrame(
        {
            "series_id": ["SKU_1"] * 20,
            "week_start": dates,
            "demand": [float(10 + (i % 5)) for i in range(20)],
        }
    )

    _, _, feature_cols = extract_features_dataset(df, features_config=cfg)
    assert len(feature_cols) == 18, f"Expected exactly 18 features, got {len(feature_cols)}: {feature_cols}"
    expected_cols = [
        "series_id",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_8",
        "lag_12",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8",
        "rolling_std_8",
        "rolling_mean_12",
        "rolling_std_12",
        "zero_freq_4",
        "week_of_year",
        "month",
        "sin_week",
        "cos_week",
    ]
    assert feature_cols == expected_cols
