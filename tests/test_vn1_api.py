"""Tests for VN1 API endpoints (/v2/forecast, dual /ready, /metadata)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demandcast.api.dependencies import get_model_container, get_vn1_model_container
from demandcast.api.main import app
from demandcast.vn1.baselines import VN1MovingAverageForecaster


@pytest.fixture
def dual_api_client(trained_dummy_model):
    """TestClient configured with active models in both UCI and VN1 containers."""
    # Setup UCI container
    uci_container = get_model_container()
    uci_container.model = trained_dummy_model
    uci_container.version = "v1.0.0-test"
    uci_container.metadata = {
        "model_type": trained_dummy_model.name,
        "checksum_sha256": "fake_sha256_uci",
        "supported_series": trained_dummy_model.supported_series,
        "feature_columns": trained_dummy_model.feature_cols,
        "training_date_range": {"min_date": "2010-01-04", "max_date": "2010-10-04"},
        "validation_metrics": {"mae": 255.94, "wape": 0.5516},
    }
    uci_container.is_ready = True
    uci_container.init_error = None

    # Setup VN1 container
    vn1_model = VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12_Official")
    vn1_model.supported_series = ["0_1_PROD1"]

    vn1_container = get_vn1_model_container()
    vn1_container.model = vn1_model
    vn1_container.version = "v1.0.0-vn1"
    vn1_container.metadata = {
        "version": "v1.0.0-vn1",
        "dataset_profile": "vn1",
        "model_type": "Baseline_MovingAvg12_Official",
        "checksum_sha256": "fake_sha256_vn1",
        "forecast_horizon": 13,
        "training_date_range": {"start_date": "2020-07-06", "end_date": "2023-10-02"},
        "validation_metrics": {"vn1_score": 0.4325, "wape": 0.4276, "mae": 207.24},
        "quality_status": "HEALTHY",
        "model_freshness": {
            "status": "stale_historical_dataset",
            "dataset_vintage": "VN1 Forecasting Accuracy Challenge (2020-07 to 2024-04)",
            "training_cutoff_date": "2023-10-02",
        },
    }
    vn1_container.is_ready = True
    vn1_container.init_error = None

    client = TestClient(app)
    return client


def test_dual_ready_endpoint(dual_api_client):
    """Verify /ready reports both UCI and VN1 active profiles."""
    response = dual_api_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["vn1_ready"] is True
    assert "uci" in data["active_profiles"]
    assert "vn1" in data["active_profiles"]


def test_dual_metadata_endpoint(dual_api_client):
    """Verify /metadata includes VN1 profile configuration alongside UCI."""
    response = dual_api_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["vn1_profile"] is not None
    assert data["vn1_profile"]["model_type"] == "Baseline_MovingAvg12_Official"
    assert data["vn1_profile"]["forecast_horizon_weeks"] == 13
    assert data["vn1_profile"]["validation_metrics"]["vn1_score"] == 0.4325


def test_v2_forecast_successful_13_weeks(dual_api_client):
    """Verify POST /v2/forecast produces 13-week multi-step forecast."""
    # Origin 2023-10-02 (Monday) with 14 weeks history
    origin = pd.to_datetime("2023-10-02")
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=13 - i)).strftime("%Y-%m-%d"),
            "sales": float(30 + (i % 7)),
            "price": 19.99,
        }
        for i in range(14)
    ]

    payload = {
        "series_id": "0_1_PROD1",
        "horizon": 13,
        "history": history,
    }

    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["dataset_profile"] == "vn1"
    assert data["series_id"] == "0_1_PROD1"
    assert data["model_version"] == "v1.0.0-vn1"
    assert data["horizon"] == 13
    assert len(data["forecasts"]) == 13
    assert data["forecast_start_date"] == "2023-10-09"
    assert data["forecast_end_date"] == "2024-01-01"

    for i, item in enumerate(data["forecasts"], start=1):
        assert item["horizon_step"] == i
        assert item["predicted_sales"] >= 0.0
        expected_date = (pd.to_datetime("2023-10-02") + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d")
        assert item["week_start"] == expected_date


def test_v2_forecast_temporal_contract_violation_lookahead(dual_api_client):
    """Verify POST /v2/forecast rejects history with origin predating model training cutoff."""
    # Cutoff is 2023-10-02; supply origin in 2022
    origin = pd.to_datetime("2022-05-02")
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=13 - i)).strftime("%Y-%m-%d"),
            "sales": 25.0,
            "price": 15.0,
        }
        for i in range(14)
    ]

    payload = {
        "series_id": "0_1_PROD1",
        "horizon": 13,
        "history": history,
    }

    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "TemporalContractViolation"
    assert "violates temporal integrity" in data["detail"]["message"]


def test_v2_forecast_rejects_non_monday(dual_api_client):
    """Verify POST /v2/forecast rejects dates that do not fall on Mondays."""
    origin = pd.to_datetime("2023-10-02")
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=13 - i)).strftime("%Y-%m-%d"),
            "sales": 20.0,
            "price": 10.0,
        }
        for i in range(14)
    ]
    # Make one date a Tuesday
    history[5]["week_start"] = "2023-08-08"

    payload = {"series_id": "0_1_PROD1", "horizon": 13, "history": history}
    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 400
    assert "Monday week-start convention" in response.json()["detail"]["message"]


def test_v2_forecast_rejects_gaps(dual_api_client):
    """Verify POST /v2/forecast rejects historical sequences with skipped weeks."""
    origin = pd.to_datetime("2023-10-02")
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=14 - i)).strftime("%Y-%m-%d"),
            "sales": 20.0,
            "price": 10.0,
        }
        for i in range(14)
    ]
    # Remove index 4 to introduce a 2-week gap
    history.pop(4)

    payload = {"series_id": "0_1_PROD1", "horizon": 13, "history": history}
    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 400
    assert "gaps or irregular intervals" in response.json()["detail"]["message"]


def test_v2_forecast_rejects_insufficient_history(dual_api_client):
    """Verify POST /v2/forecast rejects history with fewer than 12 weeks via request validation."""
    origin = pd.to_datetime("2023-10-02")
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=5 - i)).strftime("%Y-%m-%d"),
            "sales": 20.0,
            "price": 10.0,
        }
        for i in range(6)
    ]

    payload = {"series_id": "0_1_PROD1", "horizon": 13, "history": history}
    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_v2_forecast_stale_warning(dual_api_client):
    """Verify POST /v2/forecast emits StaleModelWarning when origin is >90 days past cutoff."""
    origin = pd.to_datetime("2024-02-12")  # ~133 days after 2023-10-02
    history = [
        {
            "week_start": (origin - pd.Timedelta(weeks=13 - i)).strftime("%Y-%m-%d"),
            "sales": 40.0,
            "price": 12.50,
        }
        for i in range(14)
    ]

    payload = {"series_id": "0_1_PROD1", "horizon": 13, "history": history}
    response = dual_api_client.post("/v2/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) >= 1
    assert "StaleModelWarning" in data["warnings"][0]


def test_uci_v1_forecast_remains_intact(dual_api_client):
    """Verify original /forecast endpoint continues to function without degradation."""
    base_date = pd.to_datetime("2011-01-03")
    history = [
        {
            "week_start": (base_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(20 + i),
        }
        for i in range(14)
    ]
    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = dual_api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["series_id"] == "SKU_ALPHA"
    assert len(data["forecasts"]) == 4
