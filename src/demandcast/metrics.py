"""Evaluation metrics for time-series demand forecasting."""

from typing import Any

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|y - y_hat|) / sum(y)
    Returns None if total actual demand sum(y) is 0 to avoid silent fabrication.
    """
    total_actual = float(np.sum(y_true))
    if total_actual <= 0:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) in percentage (0-100).

    Handles edge case where y_true == y_pred == 0 cleanly by setting that term to 0.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0, 0.0, diff / denominator)
    return float(np.mean(ratio) * 100.0)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Error / Bias = mean(y_pred - y_true).

    Positive values denote over-forecasting; negative denote under-forecasting.
    """
    return float(np.mean(y_pred - y_true))


def relative_forecast_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """Calculate Relative Forecast Bias = sum(y_pred - y_true) / sum(y_true).

    Dimensionless ratio. Positive values denote over-forecasting; negative denote under-forecasting.
    Returns None if sum(y_true) is zero or negative.
    """
    total_actual = float(np.sum(y_true))
    if total_actual <= 0:
        return None
    return float(np.sum(y_pred - y_true) / total_actual)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_history: np.ndarray,
    seasonality: int = 1,
) -> float | None:
    """Calculate Mean Absolute Scaled Error (MASE).

    Scale denominator = mean(|y_t - y_{t-m}|) computed on the training history.
    Returns None if denominator is 0 (constant historical series) or history is too short.
    """
    if len(train_history) <= seasonality:
        return None
    # In-sample naive errors on train history
    naive_diffs = np.abs(train_history[seasonality:] - train_history[:-seasonality])
    scale = float(np.mean(naive_diffs))
    if scale == 0 or np.isnan(scale):
        return None
    mae = mean_absolute_error(y_true, y_pred)
    return float(mae / scale)


def calculate_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_history: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute complete dictionary of forecasting metrics."""
    # Filter out NaNs (e.g. from unavailable seasonal naive baselines)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {
            "mae": None,
            "rmse": None,
            "wape": None,
            "smape": None,
            "raw_bias": None,
            "bias": None,
            "relative_bias": None,
            "mase": None,
            "count": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    mase_val = None
    if train_history is not None:
        mase_val = mean_absolute_scaled_error(yt, yp, train_history)

    wape_val = weighted_absolute_percentage_error(yt, yp)
    rel_bias = relative_forecast_bias(yt, yp)
    raw_b = forecast_bias(yt, yp)

    return {
        "mae": round(mean_absolute_error(yt, yp), 4),
        "rmse": round(root_mean_squared_error(yt, yp), 4),
        "wape": round(wape_val, 4) if wape_val is not None else None,
        "smape": round(symmetric_mean_absolute_percentage_error(yt, yp), 4),
        "raw_bias": round(raw_b, 4),
        "bias": round(raw_b, 4),
        "relative_bias": round(rel_bias, 4) if rel_bias is not None else None,
        "mase": round(mase_val, 4) if mase_val is not None else None,
        "count": len(yt),
    }


def evaluate_forecast_dataframe(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    train_history_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Join forecast DataFrame with actuals and compute overall, per-series, and per-horizon metrics.

    Forecast_df expected columns:
    ['series_id', 'week_start', 'horizon_step', 'predicted_demand', 'model_name']
    Actual_df expected columns:
    ['series_id', 'week_start', 'demand']
    """
    merged = pd.merge(
        forecast_df,
        actual_df[["series_id", "week_start", "demand"]].rename(columns={"demand": "actual_demand"}),
        on=["series_id", "week_start"],
        how="inner",
    )

    if len(merged) == 0:
        return {"overall": {}, "by_series": {}, "by_horizon": {}}

    yt_all = merged["actual_demand"].values
    yp_all = merged["predicted_demand"].values
    overall_metrics = calculate_metrics_dict(yt_all, yp_all)

    # Per-series breakdown
    by_series = {}
    series_mases = []
    for sid, group in merged.groupby("series_id"):
        hist = train_history_dict.get(str(sid)) if train_history_dict else None
        s_metrics = calculate_metrics_dict(
            group["actual_demand"].values,
            group["predicted_demand"].values,
            train_history=hist,
        )
        by_series[str(sid)] = s_metrics
        if s_metrics.get("mase") is not None:
            series_mases.append(s_metrics["mase"])

    # Aggregate macro-averaged MASE across series if available
    if series_mases:
        overall_metrics["mase"] = round(float(np.mean(series_mases)), 4)

    # Per-horizon breakdown
    by_horizon = {}
    for h_step, group in merged.groupby("horizon_step"):
        by_horizon[f"horizon_{h_step}"] = calculate_metrics_dict(
            group["actual_demand"].values,
            group["predicted_demand"].values,
        )

    return {
        "overall": overall_metrics,
        "by_series": by_series,
        "by_horizon": by_horizon,
        "sample_size": len(merged),
    }

