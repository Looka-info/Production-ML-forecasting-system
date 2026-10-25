"""Official VN1 Scoring Function and Evaluation Metrics."""

import numpy as np
import pandas as pd


def calculate_vn1_score(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calculate the official VN1 competition accuracy score.

    Score = (sum_i |y_hat_i - y_i| + |sum_i (y_hat_i - y_i)|) / sum_i y_i

    Lower values indicate superior forecasting accuracy.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    # Filter out NaNs if any
    mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    y_true_clean = y_true_arr[mask]
    y_pred_clean = y_pred_arr[mask]

    denom = np.sum(y_true_clean)
    if denom == 0.0:
        if np.sum(np.abs(y_pred_clean)) == 0.0:
            return 0.0
        return float("nan")

    abs_err = np.sum(np.abs(y_pred_clean - y_true_clean))
    agg_bias = np.abs(np.sum(y_pred_clean - y_true_clean))

    return float((abs_err + agg_bias) / denom)


def calculate_mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_p[mask] - y_t[mask])))


def calculate_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_p[mask] - y_t[mask]) ** 2)))


def calculate_wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error (Volume-Weighted Error)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    denom = np.sum(y_t[mask])
    if denom == 0.0:
        return 0.0 if np.sum(np.abs(y_p[mask])) == 0.0 else float("nan")
    return float(np.sum(np.abs(y_p[mask] - y_t[mask])) / denom)


def calculate_smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Symmetric Mean Absolute Percentage Error (percentage 0..200%)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]
    denom = np.abs(y_t) + np.abs(y_p)
    zero_mask = denom == 0.0
    numer = 200.0 * np.abs(y_p - y_t)
    result = np.zeros_like(numer)
    result[~zero_mask] = numer[~zero_mask] / denom[~zero_mask]
    return float(np.mean(result))


def calculate_raw_bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean error (Predicted - Actual)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_p[mask] - y_t[mask]))


def calculate_relative_bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Normalized aggregate bias: sum(Predicted - Actual) / sum(Actual)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    denom = np.sum(y_t[mask])
    if denom == 0.0:
        return 0.0 if np.sum(y_p[mask]) == 0.0 else float("nan")
    return float(np.sum(y_p[mask] - y_t[mask]) / denom)


def compute_all_vn1_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute dictionary of all benchmark metrics."""
    return {
        "vn1_score": round(calculate_vn1_score(y_true, y_pred), 4),
        "wape": round(calculate_wape(y_true, y_pred), 4),
        "mae": round(calculate_mae(y_true, y_pred), 4),
        "rmse": round(calculate_rmse(y_true, y_pred), 4),
        "smape": round(calculate_smape(y_true, y_pred), 4),
        "bias": round(calculate_raw_bias(y_true, y_pred), 4),
        "relative_bias": round(calculate_relative_bias(y_true, y_pred), 4),
    }


def calculate_metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics grouped by forecast horizon step (1..13).

    Requires columns: 'horizon_step', 'actual_sales', 'predicted_sales', 'model_name'.
    """
    rows = []
    for (model, h_step), group in df.groupby(["model_name", "horizon_step"]):
        metrics = compute_all_vn1_metrics(group["actual_sales"], group["predicted_sales"])
        metrics["model_name"] = model
        metrics["horizon_step"] = int(h_step)
        metrics["sample_count"] = len(group)
        rows.append(metrics)

    res = pd.DataFrame(rows)
    cols = [
        "model_name",
        "horizon_step",
        "vn1_score",
        "wape",
        "mae",
        "rmse",
        "smape",
        "bias",
        "relative_bias",
        "sample_count",
    ]
    return res[cols].sort_values(by=["model_name", "horizon_step"])


def calculate_metrics_by_series(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics grouped by series_id.

    Requires columns: 'series_id', 'actual_sales', 'predicted_sales', 'model_name'.
    """
    rows = []
    for (model, sid), group in df.groupby(["model_name", "series_id"]):
        metrics = compute_all_vn1_metrics(group["actual_sales"], group["predicted_sales"])
        metrics["model_name"] = model
        metrics["series_id"] = sid
        metrics["sample_count"] = len(group)
        metrics["total_actual"] = float(group["actual_sales"].sum())
        metrics["total_pred"] = float(group["predicted_sales"].sum())
        rows.append(metrics)

    res = pd.DataFrame(rows)
    cols = [
        "model_name",
        "series_id",
        "vn1_score",
        "wape",
        "mae",
        "rmse",
        "smape",
        "relative_bias",
        "total_actual",
        "total_pred",
        "sample_count",
    ]
    return res[cols].sort_values(by=["model_name", "total_actual"], ascending=[True, False])
