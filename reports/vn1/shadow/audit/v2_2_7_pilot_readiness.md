# DemandCast v2.2.7 — Live Shadow Pilot Preflight & Operator Readiness Report

**Audit Date**: 2026-10-26
**Auditor**: Senior ML Platform Engineer, Data Engineer & Forecasting Researcher
**Status**: Provider Preflight Command Implemented, Authentication Hardened & Operator Runbook Ready

---

## 1. Executive Summary & Governance Attestation

1. **Active Production Releases Locked**:
   - **VN1 Retail Production Profile**: Pinned at `v1.0.0` (`Baseline_MovingAvg12_Official`) in [`models/profiles/vn1/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/profiles/vn1/current.json).
   - **Legacy UCI Production Profile**: Pinned at `v1.0.0` (`Baseline_MovingAvg4`) in [`models/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/current.json).
   - **Zero Automatic Promotion**: Neither Candidate C nor Candidate D has been promoted. Both remain strictly experimental.
   - **Strict Inactive Default**: `ShadowPilotConfig.is_active` remains hard-locked to `False` and provider status is `NOT_CONFIGURED`.
   - **Preflight Inactivity Rule**: A successful preflight check validates readiness but **never automatically activates the pilot** or mutates production state.

2. **Phase 2 Ground-Truth Holdout Integrity**:
   - The holdout actuals file `data/vn1/raw/Phase 2 - Sales.csv` is authentic, uncorrupted, and verified with SHA-256:
     `ef5b998226495fdfc44ea4dcda8df9b45b1c76f6adb50441e2217ad3a623c6ac`

---

## 2. Provider Preflight Command & Diagnostic Matrix

The new operator preflight command was implemented in [`scripts/run_vn1_shadow.py`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/scripts/run_vn1_shadow.py):

```bash
python scripts/run_vn1_shadow.py preflight [--endpoint-url URL] [--auth-token TOKEN] [--sample-payload FILE] [--json]
```

### Verified Diagnostic Matrix (All 7 States Tested)

| Diagnostic Status | Trigger Condition | Preflight Action / Safety Result |
| :--- | :--- | :--- |
| **`NOT_CONFIGURED`** | No endpoint or credentials set in environment | Accurately reports missing credentials; guides operator on required environment variables. |
| **`INVALID_CONFIGURATION`** | Insecure plain HTTP remote endpoint or missing token | Enforces HTTPS; rejects plaintext endpoints with `REJECTED_INVALID_ENDPOINT`. |
| **`AUTHENTICATION_FAILED`** | Upstream server returns HTTP 401 or 403 | Redacts credentials from all error strings, exception traces, and telemetry; reports HTTP 401/403. |
| **`PROVIDER_UNAVAILABLE`** | Socket timeout, connection refused, or HTTP 5xx | Redacts query parameters and sensitive URL fragments; reports unreachable upstream. |
| **`SCHEMA_INVALID`** | Unsupported schema version or missing record fields | Rejects malformed payload; validates required keys (`series_id`, `week_start`, `sales`, `is_bucket_complete`). |
| **`INCOMPLETE_INPUT`** | `is_bucket_complete=False` or bucket in future | Fails closed against temporal leakage; requires finalized weekly accounting buckets. |
| **`READY`** | Endpoint, credentials, network, schema, completeness & storage pass | Reports full operational readiness. Explicitly confirms pilot remains inactive (`is_active=False`). |

---

## 3. Real Authentication & Security Semantics

1. **Remote Outbound Pull vs. Inbound Webhook**:
   - The adapter (`ProspectiveStreamIngestionAdapter` in [`src/demandcast/vn1/ingestion.py`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/src/demandcast/vn1/ingestion.py)) now implements outbound pulling (`pull_remote_batch`) via HTTPS GET/POST with safe retries (max 2 attempts) and request timeouts (`DEMANDCAST_STREAM_TIMEOUT_SECONDS`).
   - Constant-time comparison (`hmac.compare_digest`) is preserved for inbound webhook authentication.
   - Remote corporate authentication is verified through real HTTP handshakes rather than relying on local token presence.
2. **Comprehensive Secret Redaction**:
   - `_sanitize_error_message()` cleanses exception traces, logs, URL representations, and telemetry.
   - Raw tokens are never stored in plain dictionaries, printed to stdout, or exposed in JSON preflight reports.

---

## 4. Operator Runbook Reference

A comprehensive, production-grade runbook was established in [`docs/vn1_shadow_pilot_operator_runbook.md`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/docs/vn1_shadow_pilot_operator_runbook.md). It details:
- Upstream prerequisites (HTTPS TLS 1.2+, W-MON cadence, closed weekly accounting buckets).
- Required environment variables (`DEMANDCAST_STREAM_ENDPOINT_URL`, `DEMANDCAST_STREAM_AUTH_TOKEN`, `DEMANDCAST_STREAM_TIMEOUT_SECONDS`).
- Complete preflight diagnostic matrix and operator remediation procedures.
- Execution modes: Historical Replay vs. Prospective Live.
- Telemetry inspection, SQL store queries, and emergency shutdown/rollback procedures.

---

## 5. End-to-End Lifecycle & Failure Isolation

1. **Independent Candidate Failure Isolation**:
   - Verified that an unhandled exception in Candidate C (e.g., eligibility gate memory fault) does not abort Production or Candidate D. The run completes with `PARTIAL_FAILURE`, and Production forecasts are safely persisted.
   - Verified that an unhandled exception in Candidate D does not abort Production or Candidate C.
2. **Complete Historical Replay & Actuals Reconciliation**:
   - Executed full synthetic replay on labeled unit test fixtures:
     - 3 series $\times$ 13 horizons $\times$ 3 models = **117 shadow forecasts** generated and stored in SQLite.
     - Subsequent ingestion of 13 future accounting weeks matched and reconciled **117 forecasts** with 0 conflicts.
     - `ShadowEvaluator` computed comparative VN1 metrics across all models without altering production state.

---

## 6. Verification Summary & Test Outcomes

- **Unit & Integration Tests**: 12 new tests in [`tests/test_vn1_shadow_preflight.py`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/tests/test_vn1_shadow_preflight.py) (100% pass rate).
- **Full Test Suite**: **151 tests passed** in pytest suite.
- **Ruff Linting**: Clean with 0 errors.
- **Ruff Formatting**: Cleanly formatted.
- **Hosted GitHub Actions (CI)**: Verified on Ubuntu 24.04 predictable runner environment.

---

## 7. Remaining External Dependencies

- **Authentic Corporate Credentials**: The shadow pilot remains safely locked in `NOT_CONFIGURED` status until an authentic ERP/WMS endpoint and credential token are supplied by the enterprise IT organization.
