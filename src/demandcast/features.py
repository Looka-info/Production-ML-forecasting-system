"""Leakage-safe feature engineering for weekly retail demand forecasting."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from demandcast.config import FeaturesConfig, load_config

logger = logging.getLogger(__name__)


def create_calendar_features(dates: Any) -> pd.DataFrame:
    """Create calendar features known in advance for any given week_start."""
    dt_series = pd.Series(pd.to_datetime(dates))
    week_of_year = dt_series.dt.isocalendar().week.astype(int)
    month = dt_series.dt.month.astype(int)

    # Cyclical encoding of week of year (52 weeks standard)
    sin_week = np.sin(2 * np.pi * week_of_year / 52.0)
    cos_week = np.cos(2 * np.pi * week_of_year / 52.0)

    cal_df = pd.DataFrame(
        {
            "week_of_year": week_of_year.values,
            "month": month.values,
            "sin_week": sin_week.values,
            "cos_week": cos_week.values,
        },
        index=getattr(dates, "index", None),
    )
    return cal_df


def extract_features_single_series(
    series_id: str,
    history_df: pd.DataFrame,
    features_config: FeaturesConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract historical lags, rolling stats, and calendar features for a single product series.

    LEAKAGE-FREE GUARANTEE:
    - lag_k uses demand at t-k.
    - rolling stats use demand.shift(1).rolling(w).
    - No observation at or after target week t influences feature values at t.
    """
    config = features_config or load_config().features

    # Ensure chronological sort
    df = history_df.sort_values(by="week_start").copy().reset_index(drop=True)
    df["week_start"] = pd.to_datetime(df["week_start"])
    demand = df["demand"].astype(float)

    feature_dict: dict[str, pd.Series] = {}

    # 1. Historical demand lags
    for lag in config.lags:
        feature_dict[f"lag_{lag}"] = demand.shift(lag)

    if config.include_seasonal_lag:
        feature_dict[f"lag_{config.seasonal_lag}"] = demand.shift(config.seasonal_lag)

    # 2. Rolling statistics (strictly shift(1) to exclude current week t)
    shifted_demand = demand.shift(1)
    for window in config.rolling_windows:
        feature_dict[f"rolling_mean_{window}"] = shifted_demand.rolling(
            window, min_periods=max(2, window // 2)
        ).mean()

        if "std" in config.rolling_stats:
            feature_dict[f"rolling_std_{window}"] = shifted_demand.rolling(
                window, min_periods=max(2, window // 2)
            ).std().fillna(0.0)

    # 3. Recent zero-demand frequency
    z_win = config.zero_demand_window
    is_zero = (shifted_demand == 0.0).astype(float)
    feature_dict[f"zero_freq_{z_win}"] = is_zero.rolling(
        z_win, min_periods=max(1, z_win // 2)
    ).mean()

    # 4. Calendar features
    cal_df = create_calendar_features(df["week_start"])
    for col in cal_df.columns:
        feature_dict[col] = cal_df[col]

    # 5. Categorical series identifier
    features_df = pd.DataFrame(feature_dict, index=df.index)
    features_df["series_id"] = series_id
    features_df["week_start"] = df["week_start"]

    target = df["demand"].astype(float)
    return features_df, target


def extract_features_dataset(
    weekly_df: pd.DataFrame,
    features_config: FeaturesConfig | None = None,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build global feature matrix across all series in weekly_df.

    Returns:
    - X: Feature DataFrame including series_id, calendar, lags, and rolling stats.
    - y: Target demand Series.
    - feature_cols: List of feature column names used for modeling.
    """
    config = features_config or load_config().features

    feature_frames = []
    target_series_list = []

    for series_id, group in weekly_df.groupby("series_id"):
        feat_df, target = extract_features_single_series(
            str(series_id), group, features_config=config
        )
        feature_frames.append(feat_df)
        target_series_list.append(target)

    all_features = pd.concat(feature_frames, ignore_index=True)
    all_targets = pd.concat(target_series_list, ignore_index=True)

    if drop_na:
        # Require all configured lag features and rolling features to be non-null
        req_cols = [f"lag_{lag_val}" for lag_val in config.lags if f"lag_{lag_val}" in all_features.columns]
        if config.include_seasonal_lag and f"lag_{config.seasonal_lag}" in all_features.columns:
            req_cols.append(f"lag_{config.seasonal_lag}")
        valid_idx = all_features[req_cols].dropna().index
        all_features = all_features.loc[valid_idx].reset_index(drop=True)
        all_targets = all_targets.loc[valid_idx].reset_index(drop=True)

    # Columns that represent input features to the estimator
    exclude_cols = {"series_id", "week_start"}
    numeric_feature_cols = [c for c in all_features.columns if c not in exclude_cols]
    all_feature_cols = ["series_id"] + numeric_feature_cols

    return all_features, all_targets, all_feature_cols


def build_inference_features_for_step(
    series_id: str,
    history_df: pd.DataFrame,
    target_week_start: pd.Timestamp,
    features_config: FeaturesConfig | None = None,
) -> pd.DataFrame:
    """Generate features for predicting the single future week `target_week_start`.

    History_df contains all observations strictly PRIOR to `target_week_start`.
    Raises ValueError if history is insufficient to compute all required features without fabrication.
    """
    config = features_config or load_config().features

    # Strict history depth validation based on configured lags
    min_required = (config.seasonal_lag + 1) if config.include_seasonal_lag else (max(config.lags) + 1)
    if len(history_df) < min_required:
        raise ValueError(
            f"Cannot generate complete feature set: provided history ({len(history_df)} weeks) "
            f"is insufficient. At least {min_required} weeks of continuous history are required "
            f"to compute all configured features without fabrication."
        )

    # Append a dummy row for the target week at the end
    dummy_row = pd.DataFrame([{
        "week_start": pd.to_datetime(target_week_start),
        "demand": 0.0,
        "series_id": series_id,
    }])
    extended_df = pd.concat([history_df[["week_start", "demand"]], dummy_row], ignore_index=True)

    feat_df, _ = extract_features_single_series(
        series_id=series_id,
        history_df=extended_df,
        features_config=config,
    )

    # The last row corresponds to target_week_start
    last_row_feat = feat_df.iloc[[-1]].copy().reset_index(drop=True)
    num_cols = [c for c in last_row_feat.columns if c not in {"series_id", "week_start"}]

    # Strict integrity: verify no required feature is NaN or fabricated
    nan_cols = [c for c in num_cols if last_row_feat[c].isna().any()]
    if nan_cols:
        raise ValueError(
            f"Cannot generate complete feature set: missing values in {nan_cols}. "
            f"Provided history ({len(history_df)} weeks) is insufficient. "
            f"All features must be computed genuinely without fabricated values."
        )

    return last_row_feat
