"""Unit tests for VN1 feature engineering and leakage prevention."""

import pandas as pd

from demandcast.vn1.features import generate_vn1_features, get_feature_column_names


def test_vn1_features_no_lookahead_leakage():
    """Verify that lags and rolling stats at index t strictly depend on t-1 and earlier."""
    dates = pd.date_range("2023-01-02", periods=15, freq="W-MON")
    sales = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    prices = [10.0] * 15

    df = pd.DataFrame(
        {
            "series_id": "0_1_1",
            "week_start": dates,
            "sales": sales,
            "price": prices,
            "client_id": "0",
            "warehouse_id": "1",
            "product_id": "1",
        }
    )

    feat_df = generate_vn1_features(df, lags=[1, 2], rolling_windows=[4], include_price=True)

    # Check lag_1 at index 4 (week 5, sales=50) -> should be sales from index 3 (40)
    assert feat_df.iloc[4]["lag_1"] == 40.0
    assert feat_df.iloc[4]["lag_2"] == 30.0

    # Check rolling_mean_4 at index 4 (week 5) -> should be mean of [10, 20, 30, 40] = 25.0
    # Sales at index 4 (50.0) MUST NOT enter the rolling calculation!
    assert feat_df.iloc[4]["rolling_mean_4"] == 25.0


def test_get_feature_column_names():
    """Verify feature names list with and without price."""
    no_price = get_feature_column_names(include_price=False)
    price_aware = get_feature_column_names(include_price=True)

    assert "lag_1" in no_price
    assert "lag_13" in no_price
    assert "rolling_mean_4" in no_price
    assert "price_last" not in no_price

    assert "price_last" in price_aware
    assert "price_ratio_4w" in price_aware
    assert len(price_aware) == len(no_price) + 2
