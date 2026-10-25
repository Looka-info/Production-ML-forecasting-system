"""VN1 Global Gradient Boosting Forecaster with 13-week Recursive Multi-Step Inference."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from demandcast.vn1.features import generate_vn1_features, get_feature_column_names

logger = logging.getLogger(__name__)

# Try LightGBM; fallback to HistGradientBoosting if not available
try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_LIGHTGBM = False


class VN1Forecaster:
    """Global Gradient Boosting Forecaster for VN1 Retail Challenge."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        include_price: bool = True,
        name: str | None = None,
    ):
        self.include_price = include_price
        self.params = params or {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 120,
            "min_child_samples": 15,
            "random_state": 42,
            "verbose": -1,
        }

        if HAS_LIGHTGBM:
            self.model = LGBMRegressor(**self.params)
            default_name = "LightGBM_PriceAware" if include_price else "LightGBM_NoPrice"
        else:
            hgb_params = {
                "learning_rate": self.params.get("learning_rate", 0.05),
                "max_iter": self.params.get("n_estimators", 120),
                "max_leaf_nodes": self.params.get("num_leaves", 31),
                "min_samples_leaf": self.params.get("min_child_samples", 15),
                "random_state": self.params.get("random_state", 42),
            }
            self.model = HistGradientBoostingRegressor(**hgb_params)
            default_name = "HistGradientBoosting_PriceAware" if include_price else "HistGradientBoosting_NoPrice"

        self.name = name or default_name
        self.feature_cols = get_feature_column_names(include_price=include_price)
        self.supported_series: list[str] = []
        self.categorical_mappings: dict[str, dict[str, int]] = {}

    def fit(self, train_df: pd.DataFrame) -> "VN1Forecaster":
        """Fit global forecaster on pre-processed training dataframe."""
        self.supported_series = sorted(train_df["series_id"].unique().tolist())

        # Build categorical mappings
        for col, cat_name in [
            ("client_id", "client_id_cat"),
            ("warehouse_id", "warehouse_id_cat"),
            ("product_id", "product_id_cat"),
        ]:
            if col in train_df.columns:
                unique_vals = sorted(train_df[col].dropna().unique().tolist())
                self.categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        feat_df = generate_vn1_features(train_df, include_price=self.include_price)
        # Drop rows where lag_13 is NaN (initial burn-in)
        valid_mask = ~feat_df["lag_13"].isna() & ~feat_df["sales"].isna()
        train_valid = feat_df[valid_mask].copy()

        X = train_valid[self.feature_cols]
        y = train_valid["sales"]

        logger.info(f"Fitting {self.name} on {len(X)} samples with {len(self.feature_cols)} features...")
        self.model.fit(X, y)
        logger.info(f"Successfully fitted {self.name}.")
        return self

    def predict_recursive(
        self,
        history_df: pd.DataFrame,
        horizon: int = 13,
        origin_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Execute 13-week recursive multi-step forecasting across series.

        Predictions for step h feed into lags for steps h+1..13.
        Future price information is NEVER leaked: price from origin is preserved.
        """
        if origin_date is not None:
            history_df = history_df[history_df["week_start"] <= origin_date].copy()

        # Work with a mutable copy of history to append predictions iteratively
        sim_df = history_df.sort_values(by=["series_id", "week_start"]).copy()
        series_last_dates = sim_df.groupby("series_id")["week_start"].max().to_dict()

        predictions = []

        for h in range(1, horizon + 1):
            # 1. Generate features up to current simulation state
            feat_df = generate_vn1_features(sim_df, include_price=self.include_price)

            # Get latest step per series
            latest_step_df = feat_df.groupby("series_id").last().reset_index()

            # Predict next step
            X_step = latest_step_df[self.feature_cols].copy()
            preds = self.model.predict(X_step)
            # Non-negative retail constraint: sales cannot be negative
            preds = np.maximum(0.0, preds)

            step_records = []
            for idx, row in latest_step_df.iterrows():
                sid = row["series_id"]
                pred_val = float(preds[idx])
                step_date = series_last_dates[sid] + pd.Timedelta(weeks=h)

                predictions.append(
                    {
                        "series_id": sid,
                        "week_start": step_date,
                        "horizon_step": h,
                        "predicted_sales": pred_val,
                        "model_name": self.name,
                    }
                )

                # Append to simulation buffer for subsequent lag steps
                # Price is carried forward from origin (no future price leakage)
                step_records.append(
                    {
                        "series_id": sid,
                        "week_start": step_date,
                        "sales": pred_val,
                        "price": row.get("price", 0.0),
                        "client_id": row.get("client_id", sid.split("_")[0]),
                        "warehouse_id": row.get("warehouse_id", sid.split("_")[1]),
                        "product_id": row.get("product_id", sid.split("_")[2]),
                        "phase": 1,
                    }
                )

            step_df = pd.DataFrame(step_records)
            sim_df = pd.concat([sim_df, step_df], ignore_index=True)

        return pd.DataFrame(predictions)

    def save(self, filepath: Path | str) -> None:
        """Serialize forecaster artifact."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "name": self.name,
            "include_price": self.include_price,
            "params": self.params,
            "feature_cols": self.feature_cols,
            "supported_series": self.supported_series,
            "categorical_mappings": self.categorical_mappings,
            "model": self.model,
        }
        joblib.dump(bundle, filepath)
        logger.info(f"Saved {self.name} artifact to {filepath}")

    @classmethod
    def load(cls, filepath: Path | str) -> "VN1Forecaster":
        """Deserialize forecaster artifact."""
        bundle = joblib.load(filepath)
        instance = cls(
            params=bundle.get("params"),
            include_price=bundle.get("include_price", True),
            name=bundle.get("name"),
        )
        instance.feature_cols = bundle["feature_cols"]
        instance.supported_series = bundle["supported_series"]
        instance.categorical_mappings = bundle.get("categorical_mappings", {})
        instance.model = bundle["model"]
        return instance
