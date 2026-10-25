# DemandCast v2.2.8 — ERP/WMS Sandbox Integration & Live Pilot Activation Gate Audit Report

**Report Version**: v2.2.8
**Audit Date**: 2026-10-26
**Auditor**: Senior ML Platform Engineer & Forecasting Researcher
**System Target**: DemandCast VN1 Retail Forecasting Shadow Inference Pipeline

---

## 1. Executive Summary

DemandCast **v2.2.8** hardens the security boundary and establishes the **6-Gate Live Pilot Activation Architecture** for prospective enterprise retail forecasting. It prepares the platform for prospective integration with enterprise ERP/WMS sandboxes (e.g. SAP S/4HANA, Manhattan Associates, Blue Yonder) without fabricating endpoints, credentials, or telemetry.

### Key Governance Commitments Maintained:
- **VN1 Production Release**: Pinned at `v1.0.0`, model `Baseline_MovingAvg12_Official` ([`models/profiles/vn1/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production ML forecasting system/models/profiles/vn1/current.json)).
- **UCI Production Release**: Pinned at `v1.0.0`, model `Baseline_MovingAvg4` ([`models/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production ML forecasting system/models/current.json)).
- **Phase 2 Holdout Actuals SHA-256**: `ef5b998226495fdfc44ea4dcda8df9b45b1c76f6adb50441e2217ad3a623c6ac` (strictly preserved and unmodified).
- **Candidate Governance**: Candidate C (`PreOrigin_Evidence_Gated_Calibrated`) and Candidate D (`Baseline_MovingAvg12_PostHolidayTargeted`) remain strictly experimental challengers. Zero automatic promotion.
- **Fail-Closed Default**: `ShadowPilotConfig.is_active` remains hardcoded to `False`, and provider status is truthfully reported as `NOT_CONFIGURED`.

---

## 2. Upstream Enterprise Provider Inspection

A thorough inspection of the repository codebase, configuration files, and local runtime environment was performed:
1. **Inspection Outcome**:
   - Zero external ERP/WMS API credentials, shared secret tokens, or private endpoint URLs exist in the repository or local environment.
   - In accordance with non-negotiable DemandCast governance, no mock API contracts or simulated live streams were fabricated.
2. **Truthful Provider Status**:
   - Ingestion adapter status is reported truthfully as **`NOT_CONFIGURED`**.
   - Live prospective execution (`--mode prospective_live`) fails closed with exit code `1` until all 6 activation gates pass.
3. **Enterprise Integration Specification**:
   - The complete technical integration contract was authored in [`docs/vn1_erp_wms_sandbox_integration_spec.md`](file:///C:/Users/mumer/OneDrive/Documents/Production ML forecasting system/docs/vn1_erp_wms_sandbox_integration_spec.md), providing the enterprise IT partner with exact transport, authentication, schema, and accounting week closure requirements.

---

## 3. Live-Feed Security Boundary Audit

The security boundary of `ProspectiveStreamIngestionAdapter` was audited against remote exploitation, credential leakage, and insecure transport:

| Security Check | Expected Enforcement | Audit Result | Evidence / Diagnostic |
| :--- | :--- | :--- | :--- |
| **Remote HTTP Rejection** | Plain HTTP rejected on all remote endpoints | **PASSED** | Remote HTTP rejected even with allow_insecure_http=True |
| **Insecure Flag Isolation** | `--allow-insecure-http` strictly restricted to loopback (`localhost`, `127.0.0.1`, `::1`, `testserver`) | **PASSED** | Remote plain HTTP rejected with `REJECTED_INVALID_ENDPOINT` |
| **Production Mode Lockdown** | `DEMANDCAST_ENV=prod/production` forbids plain HTTP even on localhost | **PASSED** | Plain HTTP strictly disallowed in production mode even for localhost |
| **Secret Redaction** | Query params and passwords scrubbed from URLs and logs | **PASSED** | Masked URL: `https://erp.partner.internal/api/v1/sales` |
| **Exception Sanitization** | Bearer tokens replaced with `[REDACTED_SECRET]` | **PASSED** | Sanitized error: `Failed to connect to https://svc_user:[REDACTED_PASSWORD]@erp.partner.internal/api/v1/sales?api_key=[REDACTED_SECRET]&facility=east using bearer [REDACTED_SECRET]` |
| **Credential Whitespace** | Surrounding whitespace stripped, empty strings handled as None | **PASSED** | Whitespace stripped from tokens and API keys |

---

## 4. The 6-Gate Live Pilot Activation Audit

`ShadowPilotConfig.evaluate_activation_gates()` was verified across unconfigured and authorized states:

```
+--------------------------------------------------------------------------+
| Gate 1: Operator Authorization  (is_active=True, MANUAL_SIGN_OFF)        |
| Gate 2: Provider Configuration  (HTTPS endpoint, credentials present)    |
| Gate 3: Remote Auth & Preflight (HTTP 200 probe, preflight READY)       |
| Gate 4: Data Completeness       (Schema v1.0, W-MON, is_bucket_complete) |
| Gate 5: Persistence Integrity   (SQLite store verified & queryable)     |
| Gate 6: Shutdown Runbook        (Rollback & deactivation tested)        |
+--------------------------------------------------------------------------+
```

### Audit Findings:
1. **Unconfigured Default State (Fail-Closed)**:
   - `all_gates_passed`: **`False`**
   - Gate 1: `FAILED` (Pilot is marked inactive `is_active=False`)
   - Gate 2: `FAILED` (Provider status is `NOT_CONFIGURED`)
   - Gate 3: `FAILED` (No adapter with preflight capability supplied)
   - Gate 4: `FAILED` (Completed weekly accounting buckets could not be verified)
   - Gate 5: `PASSED` (Local SQLite store queryable and ready)
   - Gate 6: `PASSED` (Deactivation and fail-closed rollbacks verified)
   - **Conclusion**: The prospective live pipeline is safely and completely locked against accidental activation.
2. **Simulated Authorized State**:
   - `all_gates_passed`: **`True`**
   - All 6 gates passed cleanly with 0 diagnostic errors when valid operator authorization, HTTPS endpoint, preflight `READY`, and complete weekly records are provided.

---

## 5. End-to-End Prospective Shadow Replay & Reconciliation

A prospective replay run was executed through `VN1ShadowPipeline` on point-in-time test fixtures:
- **Run ID**: `5d2d068a-adee-4ee0-b96f-703017ce0f51`
- **Models Executed Concurrently**:
  1. Active Production (`Baseline_MovingAvg12_Official`)
  2. Challenger Candidate C (`PreOrigin_Evidence_Gated_Calibrated`)
  3. Reference Candidate D (`Baseline_MovingAvg12_PostHolidayTargeted`)
- **Forecasts Generated**: 78 (2 series $\times$ 13 horizons $\times$ 3 models)
- **Forecasts Stored**: Persisted to SQLite store in `PENDING_ACTUAL` status.
- **Subsequent Actuals Ingested**: 26 future weekly observations.
- **Reconciliation Outcome**: 78 forecasts reconciled with **0 conflicts**.
- **Metrics Calculated**: Production VN1 Score = `0.0492`, Candidate D VN1 Score = `0.0492` (Delta VN1 Score = `0.0`).

---

## 6. Verification Summary & Test Outcomes

- **Dedicated Sandbox Tests**: 11 integration and boundary tests in [`tests/test_vn1_sandbox_activation.py`](file:///C:/Users/mumer/OneDrive/Documents/Production ML forecasting system/tests/test_vn1_sandbox_activation.py) (100% pass rate).
- **Full Pytest Suite**: **162 tests passed** across the entire repository.
- **Ruff Linter & Formatter**: Fully compliant with 0 errors.
- **Documentation**: Enterprise sandbox specification established in [`docs/vn1_erp_wms_sandbox_integration_spec.md`](file:///C:/Users/mumer/OneDrive/Documents/Production ML forecasting system/docs/vn1_erp_wms_sandbox_integration_spec.md).

---

## 7. Outstanding IT Prerequisites for Live Sandbox Activation

Before corporate IT can activate live prospective shadow inference, the enterprise ERP/WMS team must provide:
1. **Target Sandbox HTTPS Endpoint URL** (e.g. `https://erp-sandbox.corporate.internal/api/v1/sales-feed`).
2. **Service Account Credentials** (Shared secret Bearer token or API key loaded into `DEMANDCAST_STREAM_AUTH_TOKEN`).
3. **Firewall / Ingress Rule** permitting HTTPS traffic from the DemandCast host.
4. **Weekly Batch Cadence Confirmation** (extract generated after Sunday 23:59:59Z financial close with `is_bucket_complete: true`).
5. **Operator Sign-Off** to enable `is_active = True`.
