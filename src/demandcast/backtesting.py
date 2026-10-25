"""Expanding-window chronological backtesting and split-integrity validation for DemandCast."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from demandcast.config import BacktestingConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class BacktestOrigin:
    """Represents a single chronological forecast origin cutoff."""

    origin_index: int
    origin_date: pd.Timestamp
    train_end_date: pd.Timestamp
    forecast_dates: list[pd.Timestamp]


class ChronologicalBacktester:
    """Expanding-window rolling-origin backtesting engine.

    Guarantees:
    - Shared timeline across all product series.
    - Zero future target leakage: Training is strictly <= origin_date.
    - Forecast horizon targets are strictly > origin_date.
    - Final held-out test horizon is isolated from validation folds.
    """

    def __init__(
        self,
        config: BacktestingConfig | None = None,
    ):
        self.config = config or load_config().backtesting

    def generate_origins(
        self,
        weekly_df: pd.DataFrame,
    ) -> tuple[list[BacktestOrigin], BacktestOrigin]:
        """Generate rolling validation origins and the final untouched test origin.

        Structure:
        - Latest `test_horizon` weeks are reserved exclusively for final evaluation.
        - Prior to the test horizon, `n_origins` validation origins are generated,
          separated by `step_size` weeks.
        """
        all_unique_weeks = sorted(weekly_df["week_start"].unique())
        total_weeks = len(all_unique_weeks)
        h = self.config.horizon
        test_h = self.config.test_horizon
        n_origins = self.config.n_origins
        step = self.config.step_size

        if total_weeks < (test_h + (n_origins * step) + 20):
            raise ValueError(
                f"Insufficient weekly history ({total_weeks} weeks) to configure "
                f"{n_origins} origins with horizon {h} and test horizon {test_h}."
            )

        # 1. Final held-out test origin
        test_cutoff_idx = total_weeks - test_h - 1
        test_origin_date = pd.to_datetime(all_unique_weeks[test_cutoff_idx])
        test_forecast_dates = [
            pd.to_datetime(all_unique_weeks[i]) for i in range(test_cutoff_idx + 1, test_cutoff_idx + 1 + test_h)
        ]

        test_origin = BacktestOrigin(
            origin_index=0,
            origin_date=test_origin_date,
            train_end_date=test_origin_date,
            forecast_dates=test_forecast_dates,
        )

        # 2. Validation origins (working backwards from test_cutoff_idx)
        val_origins = []
        for i in range(n_origins):
            # Origin cutoffs stepped backwards
            val_cutoff_idx = test_cutoff_idx - ((n_origins - i) * step)
            val_origin_date = pd.to_datetime(all_unique_weeks[val_cutoff_idx])
            val_forecast_dates = [
                pd.to_datetime(all_unique_weeks[j]) for j in range(val_cutoff_idx + 1, val_cutoff_idx + 1 + h)
            ]
            val_origins.append(
                BacktestOrigin(
                    origin_index=i + 1,
                    origin_date=val_origin_date,
                    train_end_date=val_origin_date,
                    forecast_dates=val_forecast_dates,
                )
            )

        # Assert split integrity
        self._verify_split_integrity(val_origins, test_origin, all_unique_weeks)

        return val_origins, test_origin

    def _verify_split_integrity(
        self,
        val_origins: list[BacktestOrigin],
        test_origin: BacktestOrigin,
        all_unique_weeks: list[Any],
    ) -> None:
        """Strict mathematical assertions that chronological boundaries are non-leaking."""
        # 1. Validation origins must be strictly increasing
        for i in range(len(val_origins) - 1):
            assert val_origins[i].origin_date < val_origins[i + 1].origin_date, (
                f"Validation origin {i} date {val_origins[i].origin_date} is not strictly "
                f"before origin {i + 1} date {val_origins[i + 1].origin_date}"
            )

        # 2. Latest validation origin must be strictly before test origin
        assert val_origins[-1].origin_date < test_origin.origin_date, (
            "Latest validation origin must precede the held-out test origin"
        )

        # 3. For every origin, forecast dates must be strictly greater than origin date
        for orig in val_origins + [test_origin]:
            for f_date in orig.forecast_dates:
                assert f_date > orig.origin_date, (
                    f"Forecast date {f_date} must be strictly after origin date {orig.origin_date}"
                )

        logger.info("Split integrity verified: No temporal lookahead or target leakage.")

    def run_backtest_for_forecaster(
        self,
        forecaster_factory_or_instance: Any,
        weekly_df: pd.DataFrame,
        origins: list[BacktestOrigin],
        model_name: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute expanding-window backtesting across all given origins and series.

        Returns:
        - predictions_df: Detailed forecasts matched with actuals
        - metrics_report: Comprehensive metrics (overall, by-series, by-horizon, by-origin)
        """
        all_preds = []

        for origin in origins:
            logger.info(
                f"Evaluating [{model_name}] at Origin #{origin.origin_index} (Cutoff: {origin.origin_date.date()})..."
            )

            # Training set: observations strictly <= origin_date
            train_mask = weekly_df["week_start"] <= origin.origin_date
            train_df = weekly_df[train_mask].copy()

            # Ensure a fresh forecaster instance fitted strictly on this origin's training set
            if callable(forecaster_factory_or_instance):
                forecaster = forecaster_factory_or_instance()
            elif hasattr(forecaster_factory_or_instance, "__class__"):
                cls = forecaster_factory_or_instance.__class__
                if hasattr(forecaster_factory_or_instance, "config") and hasattr(
                    forecaster_factory_or_instance, "features_config"
                ):
                    forecaster = cls(
                        model_config=forecaster_factory_or_instance.config,
                        features_config=forecaster_factory_or_instance.features_config,
                        name=model_name,
                    )
                elif hasattr(forecaster_factory_or_instance, "window"):
                    forecaster = cls(window=forecaster_factory_or_instance.window, name=model_name)
                elif hasattr(forecaster_factory_or_instance, "season_length"):
                    forecaster = cls(season_length=forecaster_factory_or_instance.season_length, name=model_name)
                else:
                    forecaster = cls(name=model_name)
            else:
                forecaster = forecaster_factory_or_instance

            if hasattr(forecaster, "fit"):
                forecaster.fit(train_df)

            # Precompute training-side in-sample naive scale denominator per series for MASE
            train_naive_scales: dict[str, float | None] = {}
            for series_id, s_group in train_df.groupby("series_id"):
                vals = s_group.sort_values(by="week_start")["demand"].values.astype(float)
                if len(vals) > 1:
                    naive_diff = np.abs(np.diff(vals))
                    scale = float(np.mean(naive_diff))
                    train_naive_scales[str(series_id)] = scale if scale > 0 else None
                else:
                    train_naive_scales[str(series_id)] = None

            # Generate forecasts for each product series
            for series_id, s_group in train_df.groupby("series_id"):
                f_df = forecaster.forecast(
                    series_id=str(series_id),
                    history_df=s_group,
                    horizon=len(origin.forecast_dates),
                )
                f_df["origin_date"] = origin.origin_date
                f_df["origin_index"] = origin.origin_index
                f_df["naive_scale"] = train_naive_scales.get(str(series_id))
                all_preds.append(f_df)

        full_preds_df = pd.concat(all_preds, ignore_index=True)

        # Merge with ground truth actuals
        merged = pd.merge(
            full_preds_df,
            weekly_df[["series_id", "week_start", "demand"]].rename(columns={"demand": "actual_demand"}),
            on=["series_id", "week_start"],
            how="inner",
        )

        # Compute overall and breakdown metrics
        from demandcast.metrics import calculate_metrics_dict

        yt_all = merged["actual_demand"].values
        yp_all = merged["predicted_demand"].values
        overall_metrics = calculate_metrics_dict(yt_all, yp_all)

        # Per-series evaluation with training-side MASE
        by_series = {}
        series_mases = []
        for sid, s_group in merged.groupby("series_id"):
            sid_str = str(sid)
            yt_s = s_group["actual_demand"].values
            yp_s = s_group["predicted_demand"].values
            s_dict = calculate_metrics_dict(yt_s, yp_s)

            # Origin-weighted / averaged MASE using stored training scale
            valid_scale_rows = s_group.dropna(subset=["naive_scale"])
            if len(valid_scale_rows) > 0:
                s_mae = s_dict["mae"]
                mean_scale = float(valid_scale_rows["naive_scale"].mean())
                if s_mae is not None and mean_scale > 0:
                    s_mase = round(float(s_mae / mean_scale), 4)
                    s_dict["mase"] = s_mase
                    series_mases.append(s_mase)

            by_series[sid_str] = s_dict

        if series_mases:
            overall_metrics["mase"] = round(float(np.mean(series_mases)), 4)

        # Per-horizon evaluation
        by_horizon = {}
        for h_step, h_group in merged.groupby("horizon_step"):
            by_horizon[f"horizon_{h_step}"] = calculate_metrics_dict(
                h_group["actual_demand"].values,
                h_group["predicted_demand"].values,
            )

        # Per-origin evaluation with origin MASE
        by_origin = {}
        for orig_idx, o_group in merged.groupby("origin_index"):
            o_dict = calculate_metrics_dict(
                o_group["actual_demand"].values,
                o_group["predicted_demand"].values,
            )
            o_mases = []
            for _, s_orig in o_group.groupby("series_id"):
                scale_val = s_orig["naive_scale"].dropna().values
                if len(scale_val) > 0 and scale_val[0] > 0:
                    s_mae = np.mean(np.abs(s_orig["actual_demand"] - s_orig["predicted_demand"]))
                    o_mases.append(s_mae / scale_val[0])
            if o_mases:
                o_dict["mase"] = round(float(np.mean(o_mases)), 4)
            by_origin[f"origin_{orig_idx}"] = o_dict

        metrics_report = {
            "model_name": model_name,
            "overall": overall_metrics,
            "by_series": by_series,
            "by_horizon": by_horizon,
            "by_origin": by_origin,
            "sample_size": len(merged),
        }

        return merged, metrics_report
