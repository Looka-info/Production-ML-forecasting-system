"""Post-Holiday Seasonal Calibration Layer for VN1 Retail Forecasting.

Provides calendar-aware and hierarchy-shrunk calibration adjustments for moving-average
baselines to address post-Q4 retail demand drops without adding unnecessary model complexity.

Strictly enforces nested rolling-origin estimation: calibration factors are estimated
exclusively from historical transitions completed prior to the forecast origin date.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CalibrationMethod = Literal[
    "unadjusted",
    "global_factor",
    "shrunk_client_factor",
    "post_holiday_targeted",
    "price_calibrated",
]


@dataclass
class SeasonalCalibrationConfig:
    """Configuration for seasonal calibration estimator and forecaster."""

    method: CalibrationMethod = "post_holiday_targeted"
    post_holiday_start_week: int = 1  # Calendar week of year (ISO 8601)
    post_holiday_end_week: int = 13  # First quarter post-holiday window
    shrinkage_k: float = 100.0  # Empirical Bayes volume smoothing parameter
    min_volume_threshold: float = 10.0  # Minimum weekly units to consider group
    min_factor_bound: float = 0.50  # Lower safety clipping bound
    max_factor_bound: float = 1.25  # Upper safety clipping bound
    default_fallback_factor: float = 1.0  # Neutral multiplier


class SeasonalCalibrationEstimator:
    """Estimates historical seasonal factors strictly from data prior to forecast origin."""

    def __init__(self, config: SeasonalCalibrationConfig | None = None):
        self.config = config or SeasonalCalibrationConfig()
        self.estimation_origin: pd.Timestamp | None = None
        self.global_factor: float = 1.0
        self.client_factors: dict[str, float] = {}
        self.transitions_observed: list[str] = []
        self.diagnostics: dict[str, Any] = {}

    def fit(
        self,
        history_df: pd.DataFrame,
        origin_date: pd.Timestamp,
    ) -> "SeasonalCalibrationEstimator":
        """Estimate seasonal calibration factors strictly using observations before origin_date.

        Identifies completed annual post-holiday transitions (Q4 -> Q1) strictly prior to origin.
        """
        self.estimation_origin = origin_date
        # Restrict strictly to historical observations <= origin_date
        past_df = history_df[history_df["week_start"] <= origin_date].copy()

        # Identify candidate historical January origins strictly prior to origin_date
        # In VN1, January origins fall on the first Monday of each year:
        known_jan_origins = [
            pd.to_datetime("2021-01-04"),
            pd.to_datetime("2022-01-03"),
            pd.to_datetime("2023-01-02"),
            pd.to_datetime("2024-01-01"),
        ]
        eligible_origins = [d for d in known_jan_origins if d < origin_date]

        if not eligible_origins:
            logger.info(
                f"No completed prior holiday transitions found before {origin_date.strftime('%Y-%m-%d')}. "
                "Defaulting to neutral calibration (factor = 1.0)."
            )
            self.global_factor = 1.0
            self.client_factors = {}
            self.transitions_observed = []
            return self

        transition_global_ratios = []
        client_ratio_records = []

        # Analyze each completed historical post-holiday transition
        for p_orig in eligible_origins:
            # Trailing 12 weeks of Q4 up to p_orig
            trailing_start = p_orig - pd.Timedelta(weeks=11)
            fwd_end = p_orig + pd.Timedelta(weeks=13)

            # Check if history contains the full forward Q1 period
            fwd_slice = past_df[(past_df["week_start"] > p_orig) & (past_df["week_start"] <= fwd_end)]
            trail_slice = past_df[(past_df["week_start"] >= trailing_start) & (past_df["week_start"] <= p_orig)]

            unique_fwd_weeks = fwd_slice["week_start"].nunique()
            if unique_fwd_weeks < 13:
                # Forward window not fully observed in past_df
                continue

            tot_trail = trail_slice["sales"].sum()
            tot_fwd = fwd_slice["sales"].sum()

            trail_weekly_avg = tot_trail / 12.0
            fwd_weekly_avg = tot_fwd / 13.0

            if trail_weekly_avg > 0:
                g_ratio = fwd_weekly_avg / trail_weekly_avg
                transition_global_ratios.append(g_ratio)
                self.transitions_observed.append(p_orig.strftime("%Y-%m-%d"))

            # Group by client for hierarchy analysis
            if "client_id" in past_df.columns:
                c_trail = trail_slice.groupby("client_id")["sales"].sum() / 12.0
                c_fwd = fwd_slice.groupby("client_id")["sales"].sum() / 13.0

                for cid, tm in c_trail.items():
                    if tm >= self.config.min_volume_threshold:
                        fm = c_fwd.get(cid, 0.0)
                        client_ratio_records.append(
                            {
                                "client_id": str(cid),
                                "ratio": fm / tm,
                                "volume": tm,
                            }
                        )

        if not transition_global_ratios:
            self.global_factor = 1.0
            self.client_factors = {}
            return self

        # 1. Global factor: mean historical Q1/Q4 ratio clipped to safety bounds
        raw_global = float(np.mean(transition_global_ratios))
        self.global_factor = float(np.clip(raw_global, self.config.min_factor_bound, self.config.max_factor_bound))

        # 2. Shrunk client factors: Empirical Bayes shrinkage toward global factor
        # w_c = volume / (volume + k)
        # shrunk_ratio = w_c * raw_c_ratio + (1 - w_c) * global_factor
        self.client_factors = {}
        if client_ratio_records:
            df_cr = pd.DataFrame(client_ratio_records)
            for cid, grp in df_cr.groupby("client_id"):
                raw_c_ratio = float(grp["ratio"].mean())
                tot_vol = float(grp["volume"].sum())
                weight = tot_vol / (tot_vol + self.config.shrinkage_k)
                shrunk = weight * raw_c_ratio + (1.0 - weight) * self.global_factor
                self.client_factors[str(cid)] = float(
                    np.clip(shrunk, self.config.min_factor_bound, self.config.max_factor_bound)
                )

        self.diagnostics = {
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "transitions_used": self.transitions_observed,
            "raw_global_ratio": round(raw_global, 4),
            "clipped_global_factor": round(self.global_factor, 4),
            "clients_calibrated_count": len(self.client_factors),
        }

        logger.info(
            f"Fitted seasonal calibration at origin {origin_date.strftime('%Y-%m-%d')}: "
            f"Global Factor={self.global_factor:.4f}, Clients={len(self.client_factors)} "
            f"using {len(self.transitions_observed)} prior transitions."
        )
        return self

    def get_factor(
        self,
        client_id: str | None = None,
        target_week: pd.Timestamp | None = None,
        price_ratio: float | None = None,
    ) -> float:
        """Compute calibration multiplier for a specific series and forecast week."""
        # Candidate A: Unadjusted control
        if self.config.method == "unadjusted":
            return 1.0

        # Check post-holiday window condition
        is_post_holiday = True
        if target_week is not None:
            # ISO calendar week of year (1..53)
            week_of_year = target_week.isocalendar().week
            is_post_holiday = self.config.post_holiday_start_week <= week_of_year <= self.config.post_holiday_end_week

        # Candidate D: Post-holiday targeted (returns 1.0 outside post-holiday window)
        if self.config.method == "post_holiday_targeted" and not is_post_holiday:
            return 1.0

        # Determine base seasonal multiplier
        if self.config.method == "global_factor":
            factor = self.global_factor
        elif self.config.method in [
            "shrunk_client_factor",
            "post_holiday_targeted",
            "price_calibrated",
        ]:
            if client_id is not None and str(client_id) in self.client_factors:
                factor = self.client_factors[str(client_id)]
            else:
                # Fallback to global factor for unseen or low-volume clients
                factor = self.global_factor
        else:
            factor = 1.0

        # Candidate E: Price-aware calibration correction
        if self.config.method == "price_calibrated" and price_ratio is not None and price_ratio > 0:
            # If origin price was elevated relative to trailing average, demand dampens slightly
            # Price dampening sensitivity: factor * (1.0 / sqrt(price_ratio))
            price_mult = float(np.clip(1.0 / np.sqrt(price_ratio), 0.85, 1.15))
            factor = factor * price_mult

        return float(np.clip(factor, self.config.min_factor_bound, self.config.max_factor_bound))


class CalibratedForecaster:
    """Forecaster wrapper applying seasonal calibration layer to any underlying baseline."""

    def __init__(
        self,
        base_forecaster: Any,
        calibration_config: SeasonalCalibrationConfig | None = None,
        name: str | None = None,
    ):
        self.base_forecaster = base_forecaster
        self.config = calibration_config or SeasonalCalibrationConfig()
        self.estimator = SeasonalCalibrationEstimator(self.config)
        self.name = name or f"{getattr(base_forecaster, 'name', 'BaseModel')}_Calibrated_{self.config.method}"
        self.last_applied_telemetry: dict[str, Any] = {}

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int = 13,
        origin_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts from base model and apply fitted calibration factors."""
        if origin_date is None:
            origin_date = history_df["week_start"].max()

        # 1. Fit seasonal calibration strictly from history <= origin_date
        self.estimator.fit(history_df, origin_date)

        # 2. Generate raw predictions from base model
        raw_preds = self.base_forecaster.predict(history_df, horizon=horizon, origin_date=origin_date)

        # 3. Calculate trailing price ratios per series if price calibration is requested
        price_ratios: dict[str, float] = {}
        if self.config.method == "price_calibrated" and "price" in history_df.columns:
            past_df = history_df[history_df["week_start"] <= origin_date]
            for sid, grp in past_df.groupby("series_id"):
                tail_prices = grp.sort_values("week_start").tail(12)["price"].dropna()
                if len(tail_prices) > 0 and tail_prices.mean() > 0:
                    latest_p = tail_prices.iloc[-1]
                    price_ratios[sid] = latest_p / tail_prices.mean()

        # 4. Apply calibration multipliers per row
        calibrated_preds = raw_preds.copy()
        factors_applied = []
        fallbacks_count = 0
        extreme_factor_alerts = 0

        for idx, row in calibrated_preds.iterrows():
            sid = row["series_id"]
            client_id = sid.split("_")[0] if "_" in str(sid) else None
            t_week = pd.to_datetime(row["week_start"])
            p_ratio = price_ratios.get(sid)

            factor = self.estimator.get_factor(
                client_id=client_id,
                target_week=t_week,
                price_ratio=p_ratio,
            )
            factors_applied.append(factor)

            if client_id is not None and str(client_id) not in self.estimator.client_factors:
                fallbacks_count += 1
            if factor <= self.config.min_factor_bound + 0.01 or factor >= self.config.max_factor_bound - 0.01:
                extreme_factor_alerts += 1

        calibrated_preds["calibration_factor"] = factors_applied
        calibrated_preds["raw_predicted_sales"] = calibrated_preds["predicted_sales"]
        # Non-negative retail constraint: max(0.0, raw * factor)
        calibrated_preds["predicted_sales"] = np.maximum(
            0.0, calibrated_preds["raw_predicted_sales"] * calibrated_preds["calibration_factor"]
        )
        calibrated_preds["model_name"] = self.name

        self.last_applied_telemetry = {
            "model_name": self.name,
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "method": self.config.method,
            "mean_factor": round(float(np.mean(factors_applied)), 4),
            "min_factor": round(float(np.min(factors_applied)), 4),
            "max_factor": round(float(np.max(factors_applied)), 4),
            "fallbacks_used": fallbacks_count,
            "extreme_factor_alerts": extreme_factor_alerts,
        }

        return calibrated_preds
