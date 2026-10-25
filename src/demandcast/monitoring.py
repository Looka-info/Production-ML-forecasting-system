"""Data quality validation, drift detection, and post-forecast reconciliation monitoring."""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from demandcast.config import MonitoringConfig, load_config
from demandcast.metrics import (
    calculate_metrics_dict,
)

logger = logging.getLogger(__name__)


class ForecastMonitor:
    """Production monitoring for input data quality and post-forecast performance reconciliation."""

    def __init__(self, config: MonitoringConfig | None = None):
        self.config = config or load_config().monitoring

    def check_input_data_quality(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        reference_stats: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive input quality checks and input-data drift assessment on incoming series."""
        issues: list[str] = []
        warnings: list[str] = []

        # 1. Nonnegative and finite
        demand_vals = history_df["demand"].values
        if np.any(np.isnan(demand_vals)) or np.any(np.isinf(demand_vals)):
            issues.append("Contains NaN or Inf values")
        if np.any(demand_vals < 0):
            issues.append("Contains negative demand values")

        # 2. Check for zero demand streak
        streak = 0
        max_streak = 0
        for v in demand_vals:
            if v == 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        if max_streak >= self.config.zero_demand_streak_warning:
            warnings.append(
                f"Consecutive zero-demand streak of {max_streak} weeks detected."
            )

        # 3. Input-data drift check comparing incoming feature distribution against training reference
        input_drift_info: dict[str, Any] = {"drift_detected": False}
        if reference_stats and "mean" in reference_stats and "std" in reference_stats:
            ref_mean = float(reference_stats["mean"])
            ref_std = float(reference_stats["std"])
            incoming_mean = float(np.mean(demand_vals))
            incoming_std = float(np.std(demand_vals))

            # Outlier spikes check
            threshold = ref_mean + (self.config.extreme_value_std_multiplier * max(ref_std, 1.0))
            extreme_points = int(np.sum(demand_vals > threshold))
            if extreme_points > 0:
                warnings.append(
                    f"{extreme_points} observations exceed {self.config.extreme_value_std_multiplier} "
                    f"standard deviations above training reference mean ({ref_mean:.2f})."
                )

            # Mean shift drift diagnostic
            std_diff = abs(incoming_mean - ref_mean) / max(ref_std, 1.0)
            if std_diff > 2.0:
                warnings.append(
                    f"Input data drift: Incoming demand mean ({incoming_mean:.2f}) deviates by "
                    f"{std_diff:.2f} standard deviations from training baseline ({ref_mean:.2f})."
                )
                input_drift_info["drift_detected"] = True

            input_drift_info.update({
                "incoming_mean": round(incoming_mean, 2),
                "incoming_std": round(incoming_std, 2),
                "reference_mean": round(ref_mean, 2),
                "reference_std": round(ref_std, 2),
                "mean_shift_std_ratio": round(std_diff, 2),
            })

        overall_status = "INVALID" if len(issues) > 0 else ("WARNING" if len(warnings) > 0 else "HEALTHY")

        return {
            "status": overall_status,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "series_id": series_id,
            "sample_count": len(history_df),
            "max_zero_streak": max_streak,
            "input_drift": input_drift_info,
        }

    def evaluate_forecast_actuals(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Reconcile stored forecasts against ground-truth actuals as they arrive.

        Calculates performance metrics, evaluates alert triggers, and checks for data drift.
        Forecast performance cannot be measured until corresponding actual demand is observed.
        """
        preds = (
            predictions_df.drop(columns=["actual_demand"])
            if "actual_demand" in predictions_df.columns
            else predictions_df
        )

        merged = pd.merge(
            preds,
            actuals_df[["series_id", "week_start", "demand"]].rename(
                columns={"demand": "actual_demand"}
            ),
            on=["series_id", "week_start"],
            how="inner",
        )

        if len(merged) == 0:
            return {
                "status": "NO_MATCHING_DATA",
                "message": (
                    "No matching actuals found for the supplied forecast dates. "
                    "Future forecasting quality cannot be measured until ground-truth demand is observed."
                ),
            }

        yt = merged["actual_demand"].values
        yp = merged["predicted_demand"].values

        metrics = calculate_metrics_dict(yt, yp)
        wape = metrics.get("wape")
        raw_b = metrics.get("raw_bias")
        rel_bias = metrics.get("relative_bias")

        # Alert evaluation: Relative bias is compared against dimensionless threshold
        alerts = []
        if wape is not None and wape > self.config.wape_alert_threshold:
            alerts.append(
                f"WAPE alert: {wape:.2%} exceeds threshold ({self.config.wape_alert_threshold:.2%}). "
                "Retraining or model review recommended."
            )

        if rel_bias is not None and abs(rel_bias) > self.config.bias_alert_threshold:
            direction = "Over-forecasting" if rel_bias > 0 else "Under-forecasting"
            alerts.append(
                f"Relative bias alert: {rel_bias:.2%} ({direction}, raw bias {raw_b:.2f} units). "
                f"Exceeds normalized threshold ({self.config.bias_alert_threshold:.2%}). "
                "Formula: sum(predicted - actual) / sum(actual)."
            )

        # Residual tracking (error distribution)
        residuals = yp - yt
        residual_analysis = {
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4),
            "median_residual": round(float(np.median(residuals)), 4),
            "min_residual": round(float(np.min(residuals)), 4),
            "max_residual": round(float(np.max(residuals)), 4),
            "q25_residual": round(float(np.percentile(residuals, 25)), 4),
            "q75_residual": round(float(np.percentile(residuals, 75)), 4),
        }

        # Breakdown by horizon step (if horizon_step column is available)
        by_horizon = {}
        if "horizon_step" in merged.columns:
            for h_step, h_group in merged.groupby("horizon_step"):
                by_horizon[f"horizon_{h_step}"] = calculate_metrics_dict(
                    h_group["actual_demand"].values,
                    h_group["predicted_demand"].values,
                )

        # Breakdown by series
        by_series = {}
        for sid, s_group in merged.groupby("series_id"):
            by_series[str(sid)] = calculate_metrics_dict(
                s_group["actual_demand"].values,
                s_group["predicted_demand"].values,
            )

        # Paired KS test labeled as descriptive marginal divergence diagnostic
        ks_stat, ks_pvalue = stats.ks_2samp(yt, yp)
        marginal_divergence = bool(ks_pvalue < 0.05)

        overall_status = "ALERT" if len(alerts) > 0 else "HEALTHY"

        report = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluated_observations": len(merged),
            "metrics": metrics,
            "residual_analysis": residual_analysis,
            "by_horizon": by_horizon,
            "by_series": by_series,
            "alerts": alerts,
            "drift_diagnostic": {
                "diagnostic_type": "paired_marginal_distribution_ks_diagnostic",
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_pvalue), 4),
                "marginal_divergence_flag": marginal_divergence,
                "interpretation": (
                    "Descriptive diagnostic of marginal distribution shape difference between forecasts and observed actuals. "
                    "Note: This is not an independent two-sample hypothesis test because predictions and actuals are paired by series and week."
                ),
            },
            "governance_note": (
                "Forecasting quality cannot be assessed at inference time. "
                "Accuracy metrics and residual tracking require subsequent ground-truth demand observation."
            ),
        }

        return report
