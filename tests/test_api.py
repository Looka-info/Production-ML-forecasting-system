"""Tests for FastAPI endpoints, request validation, and prediction responses."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demandcast.api.dependencies import get_model_container
from demandcast.api.main import app


@pytest.fixture
def api_client(trained_dummy_model):
    """TestClient with model loaded in ModelContainer."""
    container = get_model_container()
    container.model = trained_dummy_model
    container.version = "v1.0.0-test"
    container.metadata = {
        "model_type": trained_dummy_model.name,
        "checksum_sha256": "fake_sha256",
        "supported_series": trained_dummy_model.supported_series,
        "feature_columns": trained_dummy_model.feature_cols,
        "training_date_range": {"min_date": "2010-01-04", "max_date": "2010-10-04"},
        "validation_metrics": {"mae": 2.1, "wape": 0.05},
    }
    container.is_ready = True
    container.init_error = None

    client = TestClient(app)
    return client


def test_health_endpoint(api_client):
    """Verify /health returns 200 and healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DemandCast"


def test_ready_endpoint(api_client):
    """Verify /ready returns 200 when model is loaded."""
    response = api_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_version"] == "v1.0.0-test"


def test_metadata_endpoint(api_client):
    """Verify /metadata returns complete governance information."""
    response = api_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "supported_series" in data
    assert "SKU_ALPHA" in data["supported_series"]
    assert "limitations" in data


def test_valid_forecast_request(api_client):
    """Verify /forecast produces valid 4-week forecast for supported SKU."""
    base_date = pd.to_datetime("2011-01-03")
    history = [
        {
            "week_start": (base_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(25 + (i % 4)),
        }
        for i in range(14)
    ]

    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }

    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["series_id"] == "SKU_ALPHA"
    assert len(data["forecasts"]) == 4
    assert data["horizon"] == 4
    for item in data["forecasts"]:
        assert item["predicted_demand"] >= 0.0
        assert item["horizon_step"] in [1, 2, 3, 4]


def test_forecast_unsupported_series(api_client):
    """Verify /forecast rejects unsupported product SKU with 400 Bad Request."""
    base_date = pd.to_datetime("2011-01-03")
    history = [
        {
            "week_start": (base_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": 25.0,
        }
        for i in range(14)
    ]
    payload = {
        "series_id": "UNKNOWN_PRODUCT_999",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 400
    assert "Unsupported series_id" in response.json()["detail"]["message"]


def test_forecast_insufficient_history(api_client):
    """Verify /forecast rejects history shorter than 13 weeks with 422 Unprocessable Entity."""
    history = [
        {"week_start": "2011-01-03", "demand": 25.0},
        {"week_start": "2011-01-10", "demand": 30.0},
    ]
    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_forecast_irregular_dates(api_client):
    """Verify /forecast rejects weekly dates with gaps / irregular spacing."""
    base_date = pd.to_datetime("2011-01-03")
    history = [
        {
            "week_start": (base_date + pd.Timedelta(weeks=i if i < 5 else i + 2)).strftime("%Y-%m-%d"),
            "demand": 25.0,
        }
        for i in range(14)
    ]
    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 400
    assert "Non-continuous weekly history" in response.json()["detail"]["message"]


def test_forecast_temporal_cutoff_rejection(api_client):
    """Verify /forecast rejects history ending before model training cutoff (temporal violation)."""
    # Cutoff in test fixture is 2010-10-04. Provide history ending in May 2010.
    start_date = pd.to_datetime("2010-01-04")
    past_history = [
        {
            "week_start": (start_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(20 + i),
        }
        for i in range(14)
    ]
    assert pd.to_datetime(past_history[-1]["week_start"]) < pd.to_datetime("2010-10-04")

    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": past_history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "TemporalContractViolation"
    assert "Temporal contract violation" in detail["message"]


def test_forecast_temporal_cutoff_exact_match(api_client):
    """Verify /forecast succeeds when history ends exactly on the training cutoff date."""
    cutoff_dt = pd.to_datetime("2010-10-04")
    start_dt = cutoff_dt - pd.Timedelta(weeks=13)
    history = [
        {
            "week_start": (start_dt + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(25 + i),
        }
        for i in range(14)
    ]
    assert history[-1]["week_start"] == cutoff_dt.strftime("%Y-%m-%d")

    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    assert len(response.json()["forecasts"]) == 4


def test_forecast_temporal_cutoff_after_cutoff(api_client):
    """Verify /forecast succeeds when history origin is strictly after the training cutoff date (valid out-of-time forecast)."""
    cutoff_dt = pd.to_datetime("2010-10-04")
    # History ending 2 weeks after cutoff
    origin_dt = cutoff_dt + pd.Timedelta(weeks=2)
    start_dt = origin_dt - pd.Timedelta(weeks=13)
    history = [
        {
            "week_start": (start_dt + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(30 + i),
        }
        for i in range(14)
    ]
    assert pd.to_datetime(history[-1]["week_start"]) > cutoff_dt

    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 4
    expected_start = (origin_dt + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    assert data["forecast_start_date"] == expected_start


def test_ready_endpoint_model_absent(api_client):
    """Verify /ready returns 503 Service Unavailable when model is not loaded."""
    container = get_model_container()
    container.is_ready = False
    container.init_error = "Model artifact missing"

    response = api_client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"

    # Restore container
    container.is_ready = True
    container.init_error = None


def test_ready_endpoint_reports_quality_status_and_freshness(api_client):
    """Verify /ready communicates quality status, alert details, and model freshness."""
    response = api_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "quality_status" in data
    assert "model_freshness_status" in data
    assert data["model_freshness_status"] == "stale_historical_dataset"


def test_metadata_endpoint_reports_freshness_and_limitations(api_client):
    """Verify /metadata provides structured model freshness metadata and operational caveats."""
    response = api_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_freshness" in data
    freshness = data["model_freshness"]
    assert freshness["status"] == "stale_historical_dataset"
    assert freshness["is_calibrated_for_modern_retail"] is False
    assert "refresh_requirements" in freshness
    assert "quality_status" in data


def test_forecast_stale_model_warning_when_origin_far_past_cutoff(api_client):
    """Verify that inference requests with origin > 90 days past training cutoff return StaleModelWarning."""
    # Test fixture cutoff is 2010-10-04. Provide history ending in June 2011 (>200 days past cutoff).
    base_date = pd.to_datetime("2011-03-07")
    history = [
        {
            "week_start": (base_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(35 + (i % 5)),
        }
        for i in range(14)
    ]
    # Origin is 2011-06-06, which is > 200 days after 2010-10-04
    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) > 0
    assert any("StaleModelWarning" in w for w in data["warnings"])
    assert any("not establish validity for modern retail markets" in w for w in data["warnings"])


def test_forecast_historical_cutoff_origin_has_no_stale_warning(api_client):
    """Verify that history ending exactly on model training cutoff does not trigger StaleModelWarning."""
    cutoff_dt = pd.to_datetime("2010-10-04")
    start_dt = cutoff_dt - pd.Timedelta(weeks=13)
    history = [
        {
            "week_start": (start_dt + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "demand": float(25 + i),
        }
        for i in range(14)
    ]
    payload = {
        "series_id": "SKU_ALPHA",
        "horizon": 4,
        "history": history,
    }
    response = api_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert not any("StaleModelWarning" in w for w in data["warnings"])
