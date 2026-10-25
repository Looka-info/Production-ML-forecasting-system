# DemandCast VN1 — Shadow Ingestion Architecture Specification

**Document Version**: 1.0.0  
**Governing Standard**: Provider-Independent Shadow Ingestion Contract  
**Author**: DemandCast MLOps & Time-Series Engineering  
**Status**: Implemented & Contract-Verified  

---

## 1. Architectural Overview & Design Goals

To enable prospective observability without disrupting production forecasting, the DemandCast VN1 shadow pipeline establishes a strict **Provider Abstraction Boundary** between upstream data sources and downstream inference:

```
[Upstream Source] (CSV / Parquet / Live Stream)
       |
       v
[Shadow Ingestion Adapter] (BaseShadowIngestionAdapter)
  ├── 1. Schema & Version Validation (v1.0 JSON / DataFrame)
  ├── 2. W-MON Weekly Interval Cadence Check (Monday 00:00:00)
  ├── 3. Fail-Closed Boundary & Completeness Verification
  ├── 4. Contradictory & Duplicate Data Rejection
  └── 5. Deterministic Snapshot Hashing (SHA-256)
       |
       v
[IngestionResult] (Validated Snapshot DataFrame + SHA-256 Hash)
       |
       +------------------------------------+
       |                                    |
       v                                    v
[Active Production Model]         [Challenger Candidate]
(Baseline_MovingAvg12_Official)   (PreOrigin_Evidence_Gated_Calibrated)
       |                                    |
       +-----------------+------------------+
                         |
                         v
       [Shadow Store (SQLite)] (Idempotent Persistence)
                         |
                         v (Deferred Until Actuals Available)
       [Shadow Evaluator] (Paired Evaluation & Reconciliation)
```

---

## 2. Ingestion Adapter Contract Specification

All shadow ingestion providers implement the [`BaseShadowIngestionAdapter`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/src/demandcast/vn1/ingestion.py#L48-L85) abstract interface:

### 2.1 Interface Definition

| Method | Signature | Responsibility |
| :--- | :--- | :--- |
| **`get_status()`** | `() -> ProviderStatus` | Returns truthful status: `'NOT_CONFIGURED'`, `'CONFIGURED'`, `'HEALTHY'`, `'DEGRADED'`, or `'FAILED'`. |
| **`validate_schema()`** | `(payload: Any) -> tuple[bool, list[str]]` | Validates schema structure, required fields, and version compatibility. |
| **`ingest()`** | `(payload: Any, as_of_timestamp: str \| None) -> IngestionResult` | Ingests raw data into a validated point-in-time snapshot DataFrame with fail-closed integrity checks. |
| **`get_telemetry()`** | `() -> dict[str, Any]` | Reports batch counts, error history, and provider health metrics. |

### 2.2 Standard Ingestion Result Dataclass

```python
@dataclass
class IngestionResult:
    status: str  # 'SUCCESS', 'REJECTED_SCHEMA_VIOLATION', 'REJECTED_DUPLICATE',
    # 'REJECTED_CONTRADICTORY', 'REJECTED_OUT_OF_ORDER',
    # 'REJECTED_INCOMPLETE', 'NOT_CONFIGURED'
    provider_id: str
    schema_version: str
    record_count: int = 0
    series_count: int = 0
    last_observed_week_start: str | None = None
    as_of_timestamp: str | None = None
    snapshot_hash: str | None = None
    snapshot_df: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
```

---

## 3. Concrete Adapter Implementations

### 3.1 FileBatchIngestionAdapter (Production Batch)
- **Target**: Verified CSV and Parquet batch files or pre-loaded pandas DataFrames.
- **Capabilities**:
  - Automatically detects long format (`['series_id', 'week_start', 'sales']`) or wide VN1 format (`['Client', 'Warehouse', 'Product']` with date columns).
  - Unpivots wide format into canonical composite identifiers: `f"{Client}_{Warehouse}_{Product}"`.
  - Enforces W-MON Monday weekly cadence.
  - Automatically deduplicates identical records while rejecting contradictory records.
  - Generates deterministic SHA-256 snapshot checksum.
- **Status**: `CONFIGURED` / `HEALTHY`.

### 3.2 ProspectiveStreamIngestionAdapter (Streaming / REST Feeds)
- **Target**: Prospective HTTP/JSON or streaming event batches.
- **Truthful Status**: Reports `NOT_CONFIGURED` unless active endpoint and credentials are configured.
- **Fail-Closed Guarantees**:
  - Rejects incomplete weekly buckets (`is_bucket_complete == False`).
  - Rejects out-of-order records where bucket end time exceeds wall-clock `as_of_timestamp`.
  - Rejects contradictory sales records for identical `(series_id, week_start)` pairs.
  - Validates schema version against expected specification.

---

## 4. Fail-Closed Validation Rules & Temporal Integrity

| Validation Check | Trigger Condition | Status Code | System Action |
| :--- | :--- | :--- | :--- |
| **Schema Version Mismatch** | `payload.schema_version != adapter.schema_version` | `REJECTED_SCHEMA_VIOLATION` | Rejects batch, logs version mismatch error. |
| **Non-Monday Date** | `week_start.dt.day_of_week != 0` | `REJECTED_SCHEMA_VIOLATION` | Rejects batch; enforces W-MON 7-day interval boundaries. |
| **Incomplete Bucket** | `is_bucket_complete == False` | `REJECTED_INCOMPLETE` | Rejects prospective batch; prevents premature forecasting on partial sales. |
| **Contradictory Sales** | Same `(series_id, week_start)` with differing `sales` | `REJECTED_CONTRADICTORY` | Rejects batch; alerts upstream of data corruption. |
| **Temporal Leakage** | `bucket_end > as_of_timestamp` | `REJECTED_OUT_OF_ORDER` | Rejects batch; prevents future lookahead leakage. |
| **Missing Provider** | Live adapter called without active feed | `NOT_CONFIGURED` | Returns truthful unconfigured status without simulated data. |

---

## 5. Verification & Contract Testing

The ingestion architecture is verified by 9 dedicated contract tests in [`tests/test_vn1_ingestion_adapter.py`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/tests/test_vn1_ingestion_adapter.py):
- `test_file_batch_adapter_long_format_success`
- `test_file_batch_adapter_wide_format_unpivot`
- `test_file_batch_adapter_rejects_contradictory_entries`
- `test_file_batch_adapter_rejects_non_monday_cadence`
- `test_stream_adapter_truthful_unconfigured_status`
- `test_stream_adapter_schema_version_mismatch`
- `test_stream_adapter_rejects_incomplete_bucket`
- `test_stream_adapter_rejects_out_of_order_future_bucket`
- `test_stream_adapter_successful_ingestion`

All contract tests pass with 100% compliance.
