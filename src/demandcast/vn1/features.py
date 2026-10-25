"""Leakage-free Feature Engineering for VN1 Retail Forecasting."""

import numpy as np
import pandas as pd


def generate_vn1_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_price: bool = True,
) -> pd.DataFrame:
    """Generate strictly leakage-free tabular features for time-series forecasting.

    Requires columns: 'series_id', 'week_start', 'sales' (and 'price' if include_price=True).
    All rolling statistics and lags are strictly shifted (.shift(1)) so current sales
    never enter feature values.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 8, 12, 13]
    if rolling_windows is None:
        rolling_windows = [4, 8, 12]

    df = df.sort_values(by=["series_id", "week_start"]).copy()
    grouped = df.groupby("series_id")

    # 1. Historical Sales Lags
    for lag in lags:
        df[f"lag_{lag}"] = grouped["sales"].shift(lag)

    # 2. Shifted Rolling Statistics
    # .shift(1) ensures the current week's sales is NOT included in the window
    for w in rolling_windows:
        # Groupby rolling on shifted sales
        df[f"rolling_mean_{w}"] = df.groupby("series_id")["sales"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = (
            df.groupby("series_id")["sales"].transform(lambda s: s.shift(1).rolling(w, min_periods=2).std()).fillna(0.0)
        )

    # 3. Trailing Zero-Sales Frequency
    df["zero_freq_4"] = (
        df.groupby("series_id")["sales"]
        .transform(lambda s: (s.shift(1) == 0.0).rolling(4, min_periods=1).mean())
        .fillna(0.0)
    )

    # 4. Calendar Signals
    week = df["week_start"].dt.isocalendar().week.astype(float)
    df["week_of_year"] = week
    df["month"] = df["week_start"].dt.month.astype(float)
    df["sin_week"] = np.sin(2.0 * np.pi * week / 52.0)
    df["cos_week"] = np.cos(2.0 * np.pi * week / 52.0)

    # 5. Price Signals (if available and requested)
    if include_price and "price" in df.columns:
        # Trailing price known at origin: strictly shifted
        df["price_last"] = grouped["price"].shift(1)
        # Price relative to trailing 4-week mean price
        rolling_price_4 = df.groupby("series_id")["price"].transform(
            lambda p: p.shift(1).rolling(4, min_periods=1).mean()
        )
        df["price_ratio_4w"] = (df["price_last"] / rolling_price_4).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    else:
        df["price_last"] = 0.0
        df["price_ratio_4w"] = 1.0

    # Categorical IDs
    if "client_id" in df.columns:
        df["client_id_cat"] = df["client_id"].astype("category").cat.codes
    if "warehouse_id" in df.columns:
        df["warehouse_id_cat"] = df["warehouse_id"].astype("category").cat.codes
    if "product_id" in df.columns:
        df["product_id_cat"] = df["product_id"].astype("category").cat.codes

    return df


def get_feature_column_names(include_price: bool = True) -> list[str]:
    """Return explicit ordered list of numeric/categorical feature names."""
    cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_8",
        "lag_12",
        "lag_13",
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
        "client_id_cat",
        "warehouse_id_cat",
        "product_id_cat",
    ]
    if include_price:
        cols.extend(["price_last", "price_ratio_4w"])
    return cols
