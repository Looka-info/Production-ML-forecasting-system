"""Forecasting Baselines for VN1 Challenge (Naive, 12-week MA, 4-week MA, Seasonal Naive)."""

import numpy as np
import pandas as pd


class VN1NaiveForecaster:
    """Naive Benchmark (Last Known Value).

    Repeats the most recent observed weekly sales value across the entire 13-week horizon.
    """

    def __init__(self, name: str = "Baseline_Naive"):
        self.name = name

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int = 13,
        origin_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate 13-week naive forecasts for all series in history_df."""
        if origin_date is not None:
            history_df = history_df[history_df["week_start"] <= origin_date].copy()

        # Find last observed sales per series
        last_obs = history_df.sort_values(by=["series_id", "week_start"]).groupby("series_id").last().reset_index()

        records = []
        for _, row in last_obs.iterrows():
            sid = row["series_id"]
            val = float(max(0.0, row["sales"])) if not np.isnan(row["sales"]) else 0.0
            last_date = row["week_start"]

            for h in range(1, horizon + 1):
                fc_date = last_date + pd.Timedelta(weeks=h)
                records.append(
                    {
                        "series_id": sid,
                        "week_start": fc_date,
                        "horizon_step": h,
                        "predicted_sales": val,
                        "model_name": self.name,
                    }
                )

        return pd.DataFrame(records)


class VN1MovingAverageForecaster:
    """Moving Average Forecaster (Official 12-week benchmark or DemandCast 4-week baseline)."""

    def __init__(self, window_size: int = 12, name: str | None = None):
        self.window_size = window_size
        self.name = name or f"Baseline_MovingAvg{window_size}"

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int = 13,
        origin_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate moving average forecasts for all series in history_df."""
        if origin_date is not None:
            history_df = history_df[history_df["week_start"] <= origin_date].copy()

        # Compute trailing mean over window_size weeks per series
        def _calc_ma(sub_df: pd.DataFrame) -> tuple[float, pd.Timestamp]:
            sub_sorted = sub_df.sort_values("week_start")
            tail = sub_sorted.tail(self.window_size)
            ma_val = float(tail["sales"].mean())
            last_date = sub_sorted["week_start"].iloc[-1]
            return max(0.0, ma_val) if not np.isnan(ma_val) else 0.0, last_date

        records = []
        for sid, group in history_df.groupby("series_id"):
            val, last_date = _calc_ma(group)
            for h in range(1, horizon + 1):
                fc_date = last_date + pd.Timedelta(weeks=h)
                records.append(
                    {
                        "series_id": sid,
                        "week_start": fc_date,
                        "horizon_step": h,
                        "predicted_sales": val,
                        "model_name": self.name,
                    }
                )

        return pd.DataFrame(records)


class VN1SeasonalNaiveForecaster:
    """52-week Seasonal Naive Forecaster.

    Repeats observed sales from exactly 52 weeks prior: y_{t+h} = y_{t+h-52}.
    Falls back to trailing 4-week moving average if 52-week historical context is unavailable.
    """

    def __init__(self, season_length: int = 52, name: str = "Baseline_SeasonalNaive52"):
        self.season_length = season_length
        self.name = name

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int = 13,
        origin_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate seasonal naive forecasts for all series in history_df."""
        if origin_date is not None:
            history_df = history_df[history_df["week_start"] <= origin_date].copy()

        records = []
        for sid, group in history_df.groupby("series_id"):
            group_sorted = group.sort_values("week_start").reset_index(drop=True)
            last_date = group_sorted["week_start"].iloc[-1]
            date_to_sales = dict(zip(group_sorted["week_start"], group_sorted["sales"]))
            fallback_val = float(max(0.0, group_sorted["sales"].tail(4).mean()))

            for h in range(1, horizon + 1):
                fc_date = last_date + pd.Timedelta(weeks=h)
                seasonal_source_date = fc_date - pd.Timedelta(weeks=self.season_length)
                pred_val = date_to_sales.get(seasonal_source_date, fallback_val)
                pred_val = max(0.0, float(pred_val)) if not np.isnan(pred_val) else fallback_val

                records.append(
                    {
                        "series_id": sid,
                        "week_start": fc_date,
                        "horizon_step": h,
                        "predicted_sales": pred_val,
                        "model_name": self.name,
                    }
                )

        return pd.DataFrame(records)
