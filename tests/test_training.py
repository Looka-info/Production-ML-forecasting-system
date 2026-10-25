"""Tests for ML forecasting model fitting, recursive forecasting, and persistence."""

import numpy as np

from demandcast.models import DemandForecastModel


def test_model_fit_and_recursive_forecast(sample_weekly_demand, tmp_path):
    """Verify DemandForecastModel fitting, non-negative clipping, and multi-step recursive forecasting."""
    model = DemandForecastModel()
    model.fit(sample_weekly_demand)

    assert model.is_fitted
    assert len(model.supported_series) == 2
    assert "SKU_ALPHA" in model.supported_series

    # Single series history
    history = sample_weekly_demand[sample_weekly_demand["series_id"] == "SKU_ALPHA"].tail(15)
    f_df = model.forecast("SKU_ALPHA", history, horizon=4)

    assert len(f_df) == 4
    assert list(f_df["horizon_step"]) == [1, 2, 3, 4]
    # Verify non-negativity postprocessing
    assert (f_df["predicted_demand"] >= 0.0).all()

    # Test serialization and loading
    model_path = tmp_path / "model.joblib"
    model.save(model_path)
    assert model_path.exists()

    loaded_model = DemandForecastModel.load(model_path)
    assert loaded_model.is_fitted
    f_df_loaded = loaded_model.forecast("SKU_ALPHA", history, horizon=4)
    np.testing.assert_array_almost_equal(
        f_df["predicted_demand"].values,
        f_df_loaded["predicted_demand"].values,
    )
