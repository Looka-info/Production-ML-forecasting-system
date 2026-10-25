"""Unit tests for Provider-Independent Shadow Ingestion Adapters."""

import pandas as pd

from demandcast.vn1.ingestion import (
    FileBatchIngestionAdapter,
    ProspectiveStreamIngestionAdapter,
)


def test_file_batch_adapter_long_format_success():
    adapter = FileBatchIngestionAdapter()
    assert adapter.get_status() == "CONFIGURED"

    df = pd.DataFrame(
        [
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0},
            {"series_id": "0_0_1", "week_start": "2023-01-02", "sales": 25.5},
            {"series_id": "0_0_0", "week_start": "2023-01-09", "sales": 12.0},
        ]
    )

    result = adapter.ingest(df, as_of_timestamp="2023-01-16T00:00:00Z")
    assert result.status == "SUCCESS"
    assert result.record_count == 3
    assert result.series_count == 2
    assert result.last_observed_week_start == "2023-01-09"
    assert result.snapshot_hash is not None
    assert result.snapshot_df is not None


def test_file_batch_adapter_wide_format_unpivot():
    adapter = FileBatchIngestionAdapter()
    df_wide = pd.DataFrame(
        [
            {"Client": 0, "Warehouse": 1, "Product": 100, "2023-01-02": 15.0, "2023-01-09": 20.0},
            {"Client": 1, "Warehouse": 2, "Product": 200, "2023-01-02": 5.0, "2023-01-09": 8.0},
        ]
    )

    result = adapter.ingest(df_wide, as_of_timestamp="2023-01-16T00:00:00Z")
    assert result.status == "SUCCESS"
    assert result.record_count == 4
    assert result.series_count == 2
    assert "0_1_100" in result.snapshot_df["series_id"].values


def test_file_batch_adapter_rejects_contradictory_entries():
    adapter = FileBatchIngestionAdapter()
    df_bad = pd.DataFrame(
        [
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0},
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 50.0},  # Contradictory sales
        ]
    )

    result = adapter.ingest(df_bad)
    assert result.status == "REJECTED_CONTRADICTORY"
    assert "Contradictory" in result.errors[0]


def test_file_batch_adapter_rejects_non_monday_cadence():
    adapter = FileBatchIngestionAdapter()
    df_bad = pd.DataFrame(
        [
            {"series_id": "0_0_0", "week_start": "2023-01-04", "sales": 10.0},  # Wednesday
        ]
    )

    result = adapter.ingest(df_bad)
    assert result.status == "REJECTED_SCHEMA_VIOLATION"
    assert "W-MON" in result.errors[0]


def test_stream_adapter_truthful_unconfigured_status():
    adapter = ProspectiveStreamIngestionAdapter()
    assert adapter.get_status() == "NOT_CONFIGURED"

    # Ingestion without payload returns NOT_CONFIGURED
    res = adapter.ingest(None)
    assert res.status == "NOT_CONFIGURED"
    assert "NOT_CONFIGURED" in res.errors[0]


def test_stream_adapter_schema_version_mismatch():
    adapter = ProspectiveStreamIngestionAdapter(schema_version="v1.0")
    payload = {
        "schema_version": "v2.0-unsupported",
        "records": [{"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True}],
    }
    res = adapter.ingest(payload)
    assert res.status == "REJECTED_SCHEMA_VIOLATION"
    assert "Schema version mismatch" in res.errors[0]


def test_stream_adapter_rejects_incomplete_bucket():
    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="mock_token",
    )
    assert adapter.get_status() == "HEALTHY"

    payload = {
        "schema_version": "v1.0",
        "records": [
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True},
            {"series_id": "0_0_0", "week_start": "2023-01-09", "sales": 5.0, "is_bucket_complete": False},
        ],
    }
    res = adapter.ingest(payload, as_of_timestamp="2023-01-16T00:00:00Z")
    assert res.status == "REJECTED_INCOMPLETE"
    assert "is_bucket_complete=False" in res.errors[0]


def test_stream_adapter_rejects_out_of_order_future_bucket():
    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="mock_token",
    )
    # as_of is Jan 05 (mid-week), but payload claims bucket starting Jan 02 (closed Sunday Jan 08)
    payload = {
        "schema_version": "v1.0",
        "records": [
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True},
        ],
    }
    res = adapter.ingest(payload, as_of_timestamp="2023-01-05T12:00:00Z")
    assert res.status == "REJECTED_OUT_OF_ORDER"
    assert "temporal leakage violation" in res.errors[0]


def test_stream_adapter_successful_ingestion():
    adapter = ProspectiveStreamIngestionAdapter(
        endpoint_url="https://mock-retail-feed.internal/stream",
        auth_token="mock_token",
    )
    payload = {
        "schema_version": "v1.0",
        "records": [
            {"series_id": "0_0_0", "week_start": "2023-01-02", "sales": 10.0, "is_bucket_complete": True},
            {"series_id": "0_0_1", "week_start": "2023-01-02", "sales": 20.0, "is_bucket_complete": True},
        ],
    }
    res = adapter.ingest(payload, as_of_timestamp="2023-01-09T00:00:00Z")
    assert res.status == "SUCCESS"
    assert res.record_count == 2
    assert res.series_count == 2
    assert res.snapshot_hash is not None
    assert adapter.total_ingested_batches == 1
