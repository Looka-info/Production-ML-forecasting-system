"""Supervised Machine Learning Forecasting Model for DemandCast.

Implements HistGradientBoostingRegressor with recursive multi-step forecasting
and leakage-safe feature transformations.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from demandcast.baselines import BaseForecaster
from demandcast.config import FeaturesConfig, ModelConfig, load_config
from demandcast.features import build_inference_features_for_step, extract_features_dataset

logger = logging.getLogger(__name__)


class DemandForecastModel(BaseForecaster):
    """Machine Learning demand forecaster using HistGradientBoostingRegressor.

    Supports recursive multi-step forecasting across multiple product series
    without leaking future observations.
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        features_config: FeaturesConfig | None = None,
        name: str = "HistGradientBoosting",
    ):
        super().__init__(name=name)
        self.config = model_config or load_config().model
        self.features_config = features_config or load_config().features

        self.pipeline: Pipeline | None = None
        self.feature_cols: list[str] = []
        self.numeric_cols: list[str] = []
        self.supported_series: list[str] = []
        self.is_fitted: bool = False

        if self.features_config.include_seasonal_lag:
            self.min_history_required = self.features_config.seasonal_lag + 1
        else:
            self.min_history_required = max(self.features_config.lags) + 1

    def _build_pipeline(self, categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
        """Create sklearn Pipeline with categorical ordinal encoding and HistGradientBoosting."""
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    categorical_cols,
                ),
                ("num", "passthrough", numeric_cols),
            ],
            remainder="drop",
        )

        regressor = HistGradientBoostingRegressor(
            learning_rate=self.config.params.learning_rate,
            max_iter=self.config.params.max_iter,
            max_leaf_nodes=self.config.params.max_leaf_nodes,
            min_samples_leaf=self.config.params.min_samples_leaf,
            random_state=self.config.params.random_state,
            loss=self.config.params.loss,
            categorical_features=[0],  # the first column transformed is categorical
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])
        return pipeline

    def fit(self, training_weekly_df: pd.DataFrame) -> "DemandForecastModel":
        """Extract features and train the global model on historical weekly observations."""
        logger.info("Extracting training features across all series...")
        X_df, y_series, feature_cols = extract_features_dataset(
            training_weekly_df,
            features_config=self.features_config,
            drop_na=True,
        )

        self.feature_cols = feature_cols
        categorical_cols = ["series_id"]
        self.numeric_cols = [c for c in feature_cols if c not in categorical_cols]
        self.supported_series = sorted(list(training_weekly_df["series_id"].unique()))

        logger.info(
            f"Fitting model on {len(X_df):,} samples with {len(self.feature_cols)} features "
            f"across {len(self.supported_series)} products..."
        )

        self.pipeline = self._build_pipeline(categorical_cols, self.numeric_cols)
        self.pipeline.fit(X_df[self.feature_cols], y_series)
        self.is_fitted = True
        logger.info("Model fitting complete.")
        return self

    def forecast(
        self,
        series_id: str,
        history_df: pd.DataFrame,
        horizon: int = 4,
    ) -> pd.DataFrame:
        """Produce a multi-week recursive point forecast.

        At each step h:
        1. Generates features using the available history up to step h-1.
        2. Predicts step h demand.
        3. Applies nonnegative clipping (sales cannot be negative).
        4. Appends the predicted demand into the working history series for the next step.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("DemandForecastModel must be fitted before generating forecasts.")

        if series_id not in self.supported_series:
            logger.warning(
                f"Series {series_id} is not in fitted supported series: {self.supported_series}"
            )

        working_history = history_df.sort_values(by="week_start").copy()
        working_history["week_start"] = pd.to_datetime(working_history["week_start"])
        working_history["demand"] = working_history["demand"].astype(float)

        if len(working_history) < self.min_history_required:
            raise ValueError(
                f"DemandForecastModel requires at least {self.min_history_required} weeks of history, "
                f"but received {len(working_history)} weeks for series '{series_id}'."
            )

        last_observed_date = working_history["week_start"].iloc[-1]
        future_dates = pd.date_range(
            start=last_observed_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq="W-MON",
        )

        forecast_records = []
        for step, target_date in enumerate(future_dates, start=1):
            # Generate feature vector for target_date using working_history
            step_features_df = build_inference_features_for_step(
                series_id=series_id,
                history_df=working_history,
                target_week_start=target_date,
                features_config=self.features_config,
            )

            # Ensure all expected feature columns are present for the pipeline
            for col in self.feature_cols:
                if col not in step_features_df.columns:
                    raise ValueError(
                        f"Missing required feature column '{col}' for step {step}. "
                        "All features must be computed genuinely without fabricated substitution."
                    )

            # Predict point forecast
            raw_pred = self.pipeline.predict(step_features_df[self.feature_cols])[0]

            # Post-processing: Demand is non-negative
            clipped_pred = max(0.0, float(raw_pred))

            forecast_records.append({
                "series_id": series_id,
                "week_start": target_date,
                "horizon_step": step,
                "predicted_demand": clipped_pred,
                "model_name": self.name,
            })

            # Append prediction to working history for next recursive step
            new_hist_row = pd.DataFrame([{
                "series_id": series_id,
                "week_start": target_date,
                "demand": clipped_pred,
            }])
            working_history = pd.concat([working_history, new_hist_row], ignore_index=True)

        return pd.DataFrame(forecast_records)

    def save(self, filepath: Path) -> None:
        """Serialize model bundle to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": self.pipeline,
            "feature_cols": self.feature_cols,
            "numeric_cols": self.numeric_cols,
            "supported_series": self.supported_series,
            "model_config": self.config.model_dump(),
            "features_config": self.features_config.model_dump(),
            "name": self.name,
            "min_history_required": self.min_history_required,
        }
        joblib.dump(bundle, filepath)
        logger.info(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "DemandForecastModel":
        """Load serialized model bundle from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")

        bundle = joblib.load(filepath)
        model = cls(name=bundle.get("name", "HistGradientBoosting"))
        model.pipeline = bundle["pipeline"]
        model.feature_cols = bundle["feature_cols"]
        model.numeric_cols = bundle["numeric_cols"]
        model.supported_series = bundle["supported_series"]
        model.min_history_required = bundle.get("min_history_required", 13)
        model.is_fitted = True
        return model
