# DemandCast VN1 — Shadow Pilot Operator Runbook

**Document Version**: v2.2.7  
**Effective Date**: 2026-10-26  
**Audience**: ML Platform Engineers, MLOps Engineers, Site Reliability Engineers, and Forecasting Operators  
**System Target**: DemandCast VN1 Retail Forecasting Shadow Inference Pipeline  

---

## 1. Executive Summary & Non-Negotiable Governance

The DemandCast VN1 Shadow Pipeline allows validated challenger models to execute concurrently alongside the active production model on point-in-time retail inputs. It measures comparative performance under genuine operational conditions without risking customer orders, retail inventory replenishment, or active production forecasting.

### Non-Negotiable Governance Principles
1. **Production Immutability**:
   - **Active Production Release**: Pinned at `v1.0.0` (`Baseline_MovingAvg12_Official`) in [`models/profiles/vn1/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/profiles/vn1/current.json).
   - **Legacy UCI Release**: Pinned at `v1.0.0` (`Baseline_MovingAvg4`) in [`models/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/current.json).
   - Production forecasts remain authoritative and isolated.
2. **Experimental Challenger Status**:
   - **Candidate C** (`PreOrigin_Evidence_Gated_Calibrated`): Validated research challenger. Executes post-holiday calibration only when series meet strict pre-origin activity criteria.
   - **Candidate D** (`Baseline_MovingAvg12_PostHolidayTargeted`): Empirical Bayes reference model.
   - Neither candidate is promoted. Both remain strictly experimental.
3. **Hard Inactive Default**:
   - `ShadowPilotConfig.is_active` is hardcoded to `False`.
   - Provider status defaults to `NOT_CONFIGURED`.
   - Automatic model promotion is locked (`promotion_policy = "MANUAL_SIGN_OFF_REQUIRED"`).
4. **Preflight Rule**:
   - A successful preflight status (`READY`) verifies connectivity and compatibility, but **never automatically activates the pilot** or alters production behavior.

---

## 2. Upstream Provider Prerequisites

Before attempting prospective integration with an enterprise ERP, WMS, or POS system, verify the following prerequisites:

| Requirement | Specification | Enforcement Mechanism |
| :--- | :--- | :--- |
| **Protocol** | Strict HTTPS (TLS 1.2+) | `ProspectiveStreamIngestionAdapter` rejects plain `http://` with `REJECTED_INVALID_ENDPOINT` |
| **Authentication** | Bearer Token or `X-API-Key` | Constant-time token verification (`hmac.compare_digest`); redacted from logs |
| **Payload Format** | JSON Schema Version `v1.0` | Validated by `validate_schema()`; rejects malformed structures with `REJECTED_SCHEMA_VIOLATION` |
| **Time Cadence** | W-MON (Monday weekly start) | Day-of-week validation (`dt.day_of_week == 0`); rejects non-Mondays |
| **Accounting State** | Closed weekly financial buckets | Requires `is_bucket_complete == True` on all records |
| **Temporal Alignment** | Historical point-in-time | Bucket end ($\text{week\_start} + 6\text{d } 23\text{h } 59\text{m } 59\text{s}$) $\le \text{as\_of\_timestamp}$ |

---

## 3. Configuration & Environment Variables

Credentials must be injected via environment variables or an authorized secret manager vault (e.g., AWS Secrets Manager, HashiCorp Vault). **Never commit raw credentials to source repositories or configuration files.**

### Supported Environment Variables
```bash
# Required: Target HTTPS API endpoint of the retail sales feed
export DEMANDCAST_STREAM_ENDPOINT_URL="https://erp.corporate.internal/api/v1/sales-feed"

# Required: Shared secret authentication token or API key
export DEMANDCAST_STREAM_AUTH_TOKEN="sec_live_retail_token_sample_string"

# Optional: HTTP request timeout in seconds (Default: 10.0)
export DEMANDCAST_STREAM_TIMEOUT_SECONDS="15.0"

# Optional: Local testing only - allow plain HTTP for localhost / mock endpoints
# DO NOT ENABLE IN PRODUCTION ENVIRONMENTS
export DEMANDCAST_STREAM_ALLOW_INSECURE_HTTP="false"
```

---

## 4. Expected JSON Schema Contract (`v1.0`)

The upstream provider must emit a JSON object conforming to the following schema:

```json
{
  "schema_version": "v1.0",
  "records": [
    {
      "series_id": "0_1_100",
      "week_start": "2026-10-19",
      "sales": 42.50,
      "price": 12.99,
      "is_bucket_complete": true
    },
    {
      "series_id": "0_1_200",
      "week_start": "2026-10-19",
      "sales": 110.00,
      "price": 18.50,
      "is_bucket_complete": true
    }
  ]
}
```

### Field Definitions
- `schema_version` (*string, required*): Must be `"v1.0"`.
- `records` (*array of objects, required*): Non-empty array of weekly sales observations.
  - `series_id` (*string, required*): Composite identifier formatted as `"{Client}_{Warehouse}_{Product}"`.
  - `week_start` (*string, required*): Monday date of the weekly bucket (`"YYYY-MM-DD"`).
  - `sales` (*float, required*): Non-negative units sold during the closed weekly bucket.
  - `price` (*float, optional*): Unit sales price during the weekly bucket.
  - `is_bucket_complete` (*boolean, required*): Must be `true`. If `false`, the batch is rejected with `REJECTED_INCOMPLETE` to prevent temporal leakage from incomplete sales accounting.

---

## 5. Operator Preflight Diagnostic Procedure

Before starting any scheduled shadow inference run, the operator must execute the preflight check:

```bash
python scripts/run_vn1_shadow.py preflight
```

### Preflight Status Diagnostic Matrix

| Preflight Status | Meaning | Root Cause | Operator Remediation Action |
| :--- | :--- | :--- | :--- |
| `NOT_CONFIGURED` | No provider configured | `DEMANDCAST_STREAM_ENDPOINT_URL` or `DEMANDCAST_STREAM_AUTH_TOKEN` is unset | Export required environment variables or configure secret vault. |
| `INVALID_CONFIGURATION` | Malformed endpoint or protocol violation | Plain HTTP used for remote URL, invalid URL scheme, or missing token | Ensure URL uses strict `https://` with valid hostname and token is present. |
| `AUTHENTICATION_FAILED` | Upstream rejected credentials | Remote API returned HTTP 401 or 403 | Verify token validity and permissions with ERP/WMS administrator. |
| `PROVIDER_UNAVAILABLE` | Network or remote server error | Connection timeout, DNS resolution failure, network partition, or HTTP 5xx | Check network connectivity, corporate VPN routing, and provider health. |
| `SCHEMA_INVALID` | Payload schema violation | Missing required keys, invalid types, or non-Monday dates | Ensure upstream emits Schema v1.0 and weekly Monday dates (`W-MON`). |
| `INCOMPLETE_INPUT` | Accounting bucket incomplete | Record has `is_bucket_complete=false` or bucket end exceeds current time | Wait for financial weekly close (Sunday 23:59:59) before publishing. |
| `READY` | All preflight checks passed | Endpoint, credentials, network, schema, completeness, and SQLite store verified | Pilot is verified. Manual operator authorization required to activate. |

### JSON Diagnostic Output for Automation
For CI/CD health checks and monitoring agents, pass the `--json` flag:
```bash
python scripts/run_vn1_shadow.py preflight --json
```

---

## 6. Execution Workflow: Historical Replay vs. Prospective Live

The shadow pipeline supports two distinct operational modes:

### Mode 1: Historical Replay (Testing & Verification)
Runs point-in-time shadow inference across historical holdout records:
```bash
python scripts/run_vn1_shadow.py forecast \
  --mode historical_replay \
  --origin-date 2023-01-02 \
  --candidates candidate_c candidate_d \
  --series-subset 250
```

### Mode 2: Prospective Live (Approved Operator Activation Only)
When an upstream live provider has passed preflight and formal operator sign-off is granted, prospective live execution requires passing all 6 activation gates. The operator can authorize activation via:

1. **CLI Flag & Operator Sign-Off**:
   ```bash
   python scripts/run_vn1_shadow.py forecast \
     --mode prospective_live \
     --candidates candidate_c candidate_d \
     --authorize-pilot \
     --operator-sign-off "lead_mlops_engineer"
   ```
2. **Environment Variable Configuration**:
   ```bash
   export DEMANDCAST_SHADOW_PILOT_ACTIVE="true"
   export DEMANDCAST_OPERATOR_SIGN_OFF_BY="lead_mlops_engineer"
   ```
3. **Operator Sign-Off Configuration File**:
   Specify `--pilot-config configs/vn1/pilot_activation.yaml` containing `is_active: true` and sign-off details.

*Important Governance Safeguard: Even when operator authorization is provided, if any of the remaining 5 gates fail (e.g., provider offline, incomplete weekly buckets, store error), the pipeline terminates immediately fail-closed.*

### Partial Failure Isolation
The pipeline isolates model failures independently:
- If Candidate C fails (e.g., memory exhaustion in eligibility gate), Production and Candidate D still execute and persist successfully.
- If Candidate D fails, Production and Candidate C still execute and persist successfully.
- The run is recorded as `PARTIAL_FAILURE` in `shadow_runs`, and production forecasts remain 100% available.

---

## 7. Actuals Reconciliation & Evaluation Workflow

Shadow forecasts remain in an unverified, pending state until real actual sales are recorded for the future target weeks.

### Step 1: Reconcile Observed Actuals
When the future accounting weeks have completed and actual sales are published:
```bash
python scripts/run_vn1_shadow.py reconcile \
  --actuals-path data/vn1/raw/Phase_1_Sales.csv \
  --source corporate_erp_actuals
```

### Step 2: Generate Evaluation & Parity Report
Evaluate comparative accuracy on matched forecast–actual pairs:
```bash
python scripts/run_vn1_shadow.py report --output-dir reports/vn1/shadow/evaluations/
```

---

## 8. Monitoring & Telemetry Inspection

Inspect shadow pipeline health and database volume via CLI:
```bash
python scripts/run_vn1_shadow.py telemetry
```

### Key Operational Metrics
- `total_runs`: Total execution runs performed.
- `total_forecasts`: Total individual weekly forecasts persisted.
- `matched_forecasts`: Forecasts reconciled against ground-truth actuals.
- `awaiting_actuals_forecasts`: Prospective forecasts pending future observation.
- `auth_failures_count`: Cumulative security rejects (must alert if $> 3$).
- `total_rejected_batches`: Cumulative malformed batches rejected by fail-closed guards.

---

## 9. Emergency Rollback & Shutdown Procedure

In the event of upstream data corruption, network flapping, or system maintenance:

1. **Immediate Pipeline Deactivation**:
   Unset or revoke the stream environment variables:
   ```bash
   unset DEMANDCAST_STREAM_ENDPOINT_URL
   unset DEMANDCAST_STREAM_AUTH_TOKEN
   ```
   The preflight check will immediately revert to `NOT_CONFIGURED`, halting prospective ingestion.

2. **Verify Production Isolation**:
   Confirm active production pointers remain unchanged:
   - [`models/profiles/vn1/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/profiles/vn1/current.json) must point to `v1.0.0` (`Baseline_MovingAvg12_Official`).
   - [`models/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/current.json) must point to `v1.0.0` (`Baseline_MovingAvg4`).

3. **Database Quarantine (If Needed)**:
   If corrupt batches were ingested in SQLite:
   ```bash
   # Isolate SQLite database for forensic audit
   mv data/shadow/shadow_store.db data/shadow/shadow_store_quarantine_$(date +%Y%m%d).db
   ```
   A new, clean database will be automatically initialized on the next run.
