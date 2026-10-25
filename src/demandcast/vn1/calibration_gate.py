"""Pre-Origin Calibration Eligibility Gating for VN1 Retail Forecasting.

Provides evidence-based pre-origin eligibility evaluation to determine whether
a product-warehouse series should receive post-holiday seasonal calibration or
safely fall back to the neutral uncalibrated production baseline.

Strict Point-in-Time Contract:
All eligibility criteria are evaluated strictly using historical observations
available on or before origin_date (last_observed_week_start). Zero future observations
are accessed, guaranteeing strict temporal leakage prevention.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationGateConfig:
    """Configuration for pre-origin calibration eligibility gating."""

    min_trailing_12w_volume: float = 35.0  # Minimum total units across trailing 12 weeks (~3 units/wk)
    max_zero_demand_fraction: float = 0.50  # Maximum fraction of zero-demand weeks in history
    require_holiday_surge_elevation: bool = True  # Require MA12 >= MA26 to confirm holiday inflation
    elevation_lookback_weeks: int = 26  # Baseline window for detecting holiday surge
    fallback_factor: float = 1.0  # Neutral fallback multiplier for ineligible series


class PreOriginCalibrationGate:
    """Evaluates series-level historical evidence strictly prior to forecast origin."""

    def __init__(self, config: CalibrationGateConfig | None = None):
        self.config = config or CalibrationGateConfig()

    def evaluate_series_eligibility(
        self,
        history_df: pd.DataFrame,
        origin_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        """Evaluate calibration eligibility strictly using history <= origin_date.

        Parameters
        ----------
        history_df : pd.DataFrame
            Long-format historical sales with 'series_id', 'week_start', 'sales'.
        origin_date : pd.Timestamp or str
            The forecast origin date (last_observed_week_start).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by series_id with columns:
            ['trailing_12w_volume', 'zero_demand_fraction', 'trailing_12w_ma',
             'trailing_26w_ma', 'is_volume_eligible', 'is_sparsity_eligible',
             'is_elevation_eligible', 'is_eligible']
        """
        orig_dt = pd.to_datetime(origin_date)

        # Enforce strict point-in-time boundary: history <= origin_date
        past_df = history_df[pd.to_datetime(history_df["week_start"]) <= orig_dt].copy()

        if past_df.empty:
            logger.warning(f"No historical records found on or before {orig_dt}; all series marked ineligible.")
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "trailing_12w_volume",
                    "zero_demand_fraction",
                    "trailing_12w_ma",
                    "trailing_26w_ma",
                    "is_volume_eligible",
                    "is_sparsity_eligible",
                    "is_elevation_eligible",
                    "is_eligible",
                ]
            ).set_index("series_id")

        trailing_12_cutoff = orig_dt - pd.Timedelta(weeks=11)
        trailing_26_cutoff = orig_dt - pd.Timedelta(weeks=self.config.elevation_lookback_weeks - 1)

        past_df["week_dt"] = pd.to_datetime(past_df["week_start"])

        # 1. Total trailing 12-week volume
        t12_df = past_df[past_df["week_dt"] >= trailing_12_cutoff]
        vols_12 = t12_df.groupby("series_id")["sales"].sum()
        ma_12 = vols_12 / 12.0

        # 2. Trailing 26-week baseline MA
        t26_df = past_df[past_df["week_dt"] >= trailing_26_cutoff]
        ma_26 = t26_df.groupby("series_id")["sales"].sum() / float(self.config.elevation_lookback_weeks)

        # 3. Overall zero-demand fraction in available history
        all_weeks = past_df.groupby("series_id")["sales"].count()
        zero_weeks = past_df[past_df["sales"] == 0].groupby("series_id")["sales"].count()
        zero_frac = (zero_weeks / all_weeks).fillna(0.0)

        # Align across all unique series
        all_series = past_df["series_id"].unique()
        res_df = pd.DataFrame(index=all_series)
        res_df.index.name = "series_id"

        res_df["trailing_12w_volume"] = vols_12.reindex(all_series).fillna(0.0)
        res_df["trailing_12w_ma"] = ma_12.reindex(all_series).fillna(0.0)
        res_df["trailing_26w_ma"] = ma_26.reindex(all_series).fillna(0.0)
        res_df["zero_demand_fraction"] = zero_frac.reindex(all_series).fillna(1.0)

        # Boolean evaluation
        is_vol = res_df["trailing_12w_volume"] >= self.config.min_trailing_12w_volume
        is_sparse = res_df["zero_demand_fraction"] <= self.config.max_zero_demand_fraction

        if self.config.require_holiday_surge_elevation:
            is_elev = res_df["trailing_12w_ma"] >= res_df["trailing_26w_ma"]
        else:
            is_elev = pd.Series(True, index=res_df.index)

        res_df["is_volume_eligible"] = is_vol
        res_df["is_sparsity_eligible"] = is_sparse
        res_df["is_elevation_eligible"] = is_elev
        res_df["is_eligible"] = is_vol & is_sparse & is_elev

        return res_df

    def apply_gated_forecast(
        self,
        baseline_preds: pd.DataFrame,
        calibrated_preds: pd.DataFrame,
        eligibility_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply candidate calibration only to eligible series; revert to baseline on ineligible series."""
        merged = pd.merge(
            calibrated_preds,
            eligibility_df[["is_eligible"]],
            left_on="series_id",
            right_index=True,
            how="left",
        )
        merged["is_eligible"] = merged["is_eligible"].fillna(False)

        # Baseline predictions lookup
        base_map = baseline_preds.set_index(["series_id", "week_start"])["predicted_sales"].to_dict()

        gated_preds = calibrated_preds.copy()
        gated_sales = []
        factors = []

        for idx, row in merged.iterrows():
            sid = row["series_id"]
            wk = row["week_start"]
            is_elig = bool(row["is_eligible"])

            if is_elig:
                gated_sales.append(row["predicted_sales"])
                factors.append(row.get("calibration_factor", 1.0))
            else:
                fallback_val = base_map.get((sid, wk), row["predicted_sales"])
                gated_sales.append(fallback_val)
                factors.append(self.config.fallback_factor)

        gated_preds["predicted_sales"] = gated_sales
        gated_preds["calibration_factor"] = factors
        gated_preds["is_calibration_eligible"] = merged["is_eligible"].values
        gated_preds["model_name"] = "Candidate_C_PreOriginGated"

        return gated_preds
