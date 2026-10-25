"""Tests for expanding-window chronological backtesting and split integrity."""

from demandcast.backtesting import ChronologicalBacktester
from demandcast.baselines import NaivePersistenceForecaster
from demandcast.config import BacktestingConfig


def test_generate_origins_and_split_integrity(sample_weekly_demand):
    """Verify origin generation, temporal progression, and zero target leakage."""
    config = BacktestingConfig(n_origins=3, horizon=4, test_horizon=4, step_size=4)
    backtester = ChronologicalBacktester(config)

    val_origins, test_origin = backtester.generate_origins(sample_weekly_demand)

    assert len(val_origins) == 3
    assert test_origin is not None

    # Verify temporal ordering
    for i in range(len(val_origins) - 1):
        assert val_origins[i].origin_date < val_origins[i + 1].origin_date

    assert val_origins[-1].origin_date < test_origin.origin_date

    # Verify that for each origin, forecast dates are strictly in the future
    for orig in val_origins:
        assert len(orig.forecast_dates) == 4
        for f_date in orig.forecast_dates:
            assert f_date > orig.origin_date


def test_run_backtest_execution(sample_weekly_demand):
    """Verify backtest execution produces matched forecast and actual pairs."""
    config = BacktestingConfig(n_origins=2, horizon=4, test_horizon=4, step_size=4)
    backtester = ChronologicalBacktester(config)
    val_origins, _ = backtester.generate_origins(sample_weekly_demand)

    forecaster = NaivePersistenceForecaster()
    merged, metrics = backtester.run_backtest_for_forecaster(
        forecaster_factory_or_instance=forecaster,
        weekly_df=sample_weekly_demand,
        origins=val_origins,
        model_name=forecaster.name,
    )

    assert "actual_demand" in merged.columns
    assert "predicted_demand" in merged.columns
    assert "overall" in metrics
    assert metrics["overall"]["mae"] is not None


def test_test_horizon_3_complete_weeks_split(sample_weekly_demand):
    """Verify that configuring test_horizon=3 produces exactly 3 complete test forecast dates

    while keeping validation origins strictly prior to the test period.
    """
    config = BacktestingConfig(n_origins=3, horizon=4, test_horizon=3, step_size=4)
    backtester = ChronologicalBacktester(config)

    val_origins, test_origin = backtester.generate_origins(sample_weekly_demand)

    assert len(val_origins) == 3
    assert len(test_origin.forecast_dates) == 3

    # All validation forecast dates must precede the test origin's forecast dates
    max_val_fc_date = max(d for orig in val_origins for d in orig.forecast_dates)
    min_test_fc_date = min(test_origin.forecast_dates)
    assert max_val_fc_date <= test_origin.origin_date
    assert test_origin.origin_date < min_test_fc_date
