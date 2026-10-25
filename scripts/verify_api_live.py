import logging

import pandas as pd
from fastapi.testclient import TestClient

from demandcast.api.main import app
from demandcast.config import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_api")


def main():
    root = get_project_root()
    data_path = root / "data/processed/weekly_demand.parquet"

    logger.info("Initializing TestClient for DemandCast API...")
    with TestClient(app) as client:
        # 1. Test /health
        res_health = client.get("/health")
        logger.info(f"GET /health -> Status: {res_health.status_code}, Body: {res_health.json()}")
        assert res_health.status_code == 200

        # 2. Test /ready
        res_ready = client.get("/ready")
        ready_data = res_ready.json()
        logger.info(f"GET /ready -> Status: {res_ready.status_code}, Body: {ready_data}")
        assert res_ready.status_code == 200
        assert ready_data["status"] == "ready"
        assert ready_data["quality_status"] == "ALERT"
        assert ready_data["model_freshness_status"] == "stale_historical_dataset"

        # 3. Test /metadata
        res_meta = client.get("/metadata")
        meta_data = res_meta.json()
        logger.info(f"GET /metadata -> Status: {res_meta.status_code}, Model Type: {meta_data.get('model_type')}")
        logger.info(f"Training Range: {meta_data.get('training_date_range')}")
        logger.info(f"Model Freshness: {meta_data.get('model_freshness')}")
        assert res_meta.status_code == 200
        assert meta_data["quality_status"] == "ALERT"
        assert meta_data["model_freshness"]["status"] == "stale_historical_dataset"
        assert meta_data["model_freshness"]["is_calibrated_for_modern_retail"] is False
        assert "refresh_requirements" in meta_data["model_freshness"]
        supported = meta_data["supported_series"]
        cutoff_date = meta_data.get("training_date_range", {}).get("max_date", "2011-11-07")

        # 4. Test Temporal Contract Rejection: Request with history ending before cutoff (e.g. April 2011)
        test_sku = supported[0]
        april_base = pd.to_datetime("2011-01-03")
        april_history = [
            {
                "week_start": (april_base + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
                "demand": float(50.0 + (i * 2)),
            }
            for i in range(14)
        ]
        res_temporal = client.post(
            "/forecast",
            json={"series_id": test_sku, "horizon": 4, "history": april_history},
        )
        logger.info(f"POST /forecast (past origin April 2011) -> Status: {res_temporal.status_code}")
        assert res_temporal.status_code == 400
        temporal_detail = res_temporal.json()["detail"]
        assert temporal_detail["error"] == "TemporalContractViolation"
        logger.info(f"Temporal contract correctly rejected lookahead request: {temporal_detail['message'][:80]}...")

        # 5. Test Temporally Valid Forecast Request: History ending exactly at model training cutoff (2011-11-07)
        # Using real historical observations from weekly_demand.parquet
        assert data_path.exists(), f"Processed data missing at {data_path}"
        weekly_df = pd.read_parquet(data_path)
        sku_weekly = weekly_df[weekly_df["series_id"] == test_sku].sort_values(by="week_start")
        valid_history_df = sku_weekly[sku_weekly["week_start"] <= pd.to_datetime(cutoff_date)].tail(14)

        valid_history = [
            {
                "week_start": row["week_start"].strftime("%Y-%m-%d"),
                "demand": float(row["demand"]),
            }
            for _, row in valid_history_df.iterrows()
        ]
        assert valid_history[-1]["week_start"] == cutoff_date, (
            f"Expected last date {cutoff_date}, got {valid_history[-1]['week_start']}"
        )

        res_fc = client.post(
            "/forecast",
            json={"series_id": test_sku, "horizon": 4, "history": valid_history},
        )
        logger.info(f"POST /forecast (aligned origin {cutoff_date}) -> Status: {res_fc.status_code}")
        assert res_fc.status_code == 200
        fc_data = res_fc.json()
        forecasts = fc_data["forecasts"]
        assert len(forecasts) == 4, f"Expected 4 forecast steps, got {len(forecasts)}"

        forecast_dates = [f["week_start"] for f in forecasts]
        expected_dates = ["2011-11-14", "2011-11-21", "2011-11-28", "2011-12-05"]
        assert forecast_dates == expected_dates, f"Expected {expected_dates}, got {forecast_dates}"
        logger.info(f"Verified 4-week forecast dates cover: {forecast_dates}")
        logger.info(f"Forecast predictions: {[f['predicted_demand'] for f in forecasts]}")

        assert not any("StaleModelWarning" in w for w in fc_data["warnings"]), (
            "Expected no stale warning for historical cutoff origin."
        )

        # 5b. Test Stale-Model Warning: Request ending > 90 days after training cutoff (e.g. April 2012)
        stale_base = pd.to_datetime("2012-01-02")
        stale_history = [
            {
                "week_start": (stale_base + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
                "demand": float(40.0 + (i % 3)),
            }
            for i in range(14)
        ]
        res_stale = client.post(
            "/forecast",
            json={"series_id": test_sku, "horizon": 4, "history": stale_history},
        )
        assert res_stale.status_code == 200
        stale_data = res_stale.json()
        assert any("StaleModelWarning" in w for w in stale_data["warnings"]), (
            "Expected StaleModelWarning for history > 90 days past cutoff."
        )
        logger.info(
            f"StaleModelWarning correctly attached for date {stale_history[-1]['week_start']}: {stale_data['warnings'][0][:80]}..."
        )

        # 6. Test unsupported SKU
        res_bad = client.post(
            "/forecast",
            json={"series_id": "UNSUPPORTED_SKU_9999", "horizon": 4, "history": valid_history},
        )
        assert res_bad.status_code == 400
        assert res_bad.json()["detail"]["error"] == "InvalidInputData"
        logger.info("Unsupported SKU correctly rejected with HTTP 400 InvalidInputData.")

    logger.info("=" * 70)
    logger.info("ALL API ENDPOINTS AND TEMPORAL BOUNDARIES VERIFIED SUCCESSFULLY!")
    logger.info("Note: This historical test on 2011 data verifies software and temporal integrity,")
    logger.info("not that the historical model predicts modern retail customer demand.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
