"""Standard forecasting baselines for benchmark comparisons."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forecast(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        horizon: int = 4,
    ) -> pd.DataFrame:
        """Produce a multi-step forecast given historical weekly observations.

        Parameters:
        - series_id: Product identifier.
        - history_df: DataFrame with ['week_start', 'demand'] sorted chronologically.
        - horizon: Number of future weeks to forecast (1..4).

        Returns:
        DataFrame with columns:
        ['series_id', 'week_start', 'horizon_step', 'predicted_demand', 'model_name']
        """


class NaivePersistenceForecaster(BaseForecaster):
    """Baseline A: Naive forecast (persistence).

    Predicts all future horizon steps using the most recently observed demand value:
    y_hat_{T+h} = y_T
    """

    def __init__(self, name: str = "Baseline_Naive"):
        super().__init__(name=name)

    def forecast(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        horizon: int = 4,
    ) -> pd.DataFrame:
        if len(history_df) == 0:
            raise ValueError(f"History DataFrame is empty for series {series_id}")

        history_df = history_df.sort_values(by="week_start")
        last_date = pd.to_datetime(history_df["week_start"].iloc[-1])
        last_demand = float(history_df["demand"].iloc[-1])

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq="W-MON",
        )

        records = []
        for step, f_date in enumerate(future_dates, start=1):
            records.append(
                {
                    "series_id": series_id,
                    "week_start": f_date,
                    "horizon_step": step,
                    "predicted_demand": max(0.0, last_demand),
                    "model_name": self.name,
                }
            )

        return pd.DataFrame(records)


class MovingAverageForecaster(BaseForecaster):
    """Baseline B: 4-week moving average.

    Predicts future demand using the average of the most recent 4 observed weekly values.
    Uses only trailing observations prior to the forecast origin.
    """

    def __init__(
        self,
        window: int = 4,
        name: str = "Baseline_MovingAvg4",
        supported_series: list[str] | None = None,
    ):
        super().__init__(name=name)
        self.window = window
        self.supported_series = supported_series or []
        self.feature_cols: list[str] = []
        self.min_history_required = window
        self.is_fitted = True

    def fit(self, training_weekly_df: pd.DataFrame) -> "MovingAverageForecaster":
        """Record supported series from training data."""
        self.supported_series = sorted(list(training_weekly_df["series_id"].unique()))
        self.is_fitted = True
        return self

    def forecast(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        horizon: int = 4,
    ) -> pd.DataFrame:
        if len(history_df) < self.min_history_required:
            raise ValueError(
                f"Insufficient history for series {series_id}. "
                f"MovingAverageForecaster requires at least {self.min_history_required} weeks, "
                f"but received {len(history_df)} weeks."
            )

        history_df = history_df.sort_values(by="week_start")
        last_date = pd.to_datetime(history_df["week_start"].iloc[-1])
        # Use available history up to window size
        trailing_demand = history_df["demand"].tail(self.window)
        ma_forecast = float(trailing_demand.mean())

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq="W-MON",
        )

        records = []
        for step, f_date in enumerate(future_dates, start=1):
            records.append(
                {
                    "series_id": series_id,
                    "week_start": f_date,
                    "horizon_step": step,
                    "predicted_demand": max(0.0, ma_forecast),
                    "model_name": self.name,
                }
            )

        return pd.DataFrame(records)

    def save(self, filepath: Any) -> None:
        """Serialize baseline forecaster bundle."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "name": self.name,
            "window": self.window,
            "supported_series": self.supported_series,
            "feature_cols": self.feature_cols,
            "model_type": "MovingAverageForecaster",
            "min_history_required": self.min_history_required,
        }
        joblib.dump(bundle, p)

    @classmethod
    def load(cls, filepath: Any) -> "MovingAverageForecaster":
        """Deserialize baseline forecaster bundle."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found at {p}")
        bundle = joblib.load(p)
        model = cls(
            window=bundle.get("window", 4),
            name=bundle.get("name", "Baseline_MovingAvg4"),
            supported_series=bundle.get("supported_series", []),
        )
        model.feature_cols = bundle.get("feature_cols", [])
        model.min_history_required = bundle.get("min_history_required", model.window)
        model.is_fitted = True
        return model


class SeasonalNaiveForecaster(BaseForecaster):
    """Baseline C: Seasonal naive forecast (52 weeks prior).

    Predicts each future week using the demand observed exactly 52 weeks earlier.
    If the 52-week lag is unavailable in history, records predicted_demand as NaN / None
    rather than fabricating zero.
    """

    def __init__(self, season_length: int = 52, name: str = "Baseline_SeasonalNaive52"):
        super().__init__(name=name)
        self.season_length = season_length

    def forecast(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        horizon: int = 4,
    ) -> pd.DataFrame:
        history_df = history_df.sort_values(by="week_start").copy()
        history_df["week_start"] = pd.to_datetime(history_df["week_start"])
        date_demand_map = dict(zip(history_df["week_start"], history_df["demand"]))

        last_date = history_df["week_start"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq="W-MON",
        )

        records = []
        for step, f_date in enumerate(future_dates, start=1):
            seasonal_date = f_date - pd.DateOffset(weeks=self.season_length)
            # Look up closest weekly date or exact date in map
            seasonal_val = date_demand_map.get(seasonal_date, np.nan)
            if not np.isnan(seasonal_val):
                pred_val = max(0.0, float(seasonal_val))
            else:
                pred_val = np.nan

            records.append(
                {
                    "series_id": series_id,
                    "week_start": f_date,
                    "horizon_step": step,
                    "predicted_demand": pred_val,
                    "model_name": self.name,
                }
            )

        return pd.DataFrame(records)
