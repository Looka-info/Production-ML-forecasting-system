# DemandCast VN1 — Enterprise ERP/WMS Sandbox Integration Specification

**Document Version**: v2.2.8  
**Effective Date**: 2026-10-26  
**Audience**: Enterprise Systems Architects, ERP/WMS Integration Engineers, ML Platform Engineers, Security Administrators  
**System Target**: DemandCast VN1 Retail Forecasting Shadow Inference Pipeline  

---

## 1. Executive Overview & Governance Boundary

This specification defines the security, protocol, schema, and activation contracts required to integrate an enterprise Enterprise Resource Planning (ERP), Warehouse Management System (WMS), or Point of Sale (POS) sandbox environment with DemandCast VN1.

### Governance Constraints
1. **Production Isolation**:
   - The authoritative production forecasting pipeline remains pinned to `v1.0.0` (`Baseline_MovingAvg12_Official`) in [`models/profiles/vn1/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/profiles/vn1/current.json).
   - Legacy UCI production remains pinned to `v1.0.0` (`Baseline_MovingAvg4`) in [`models/current.json`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/models/current.json).
   - Live shadow forecasting runs strictly in parallel without altering customer replenishment schedules, ERP order proposals, or warehouse dispatch instructions.
2. **Strict Inactive Default**:
   - `ShadowPilotConfig.is_active` defaults to `False`.
   - Provider state defaults to `NOT_CONFIGURED`.
   - Preflight success (`READY`) does **not** automatically activate the pilot.
3. **No Fabricated Infrastructure**:
   - In accordance with DemandCast engineering standards, no external provider endpoint, credential, or mock telemetry is fabricated.
   - Until genuine enterprise sandbox credentials are provisioned by corporate IT, the system reports status truthfully as `NOT_CONFIGURED`.

---

## 2. Security & Network Boundary Contract

Integration between the upstream ERP/WMS sandbox and DemandCast follows a zero-trust ingest architecture.

```
+-----------------------------------+               +--------------------------------------+
| Enterprise ERP / WMS Sandbox      |               | DemandCast Ingestion Boundary        |
| (SAP S/4HANA, Manhattan, BlueYonder)|               | (ProspectiveStreamIngestionAdapter)   |
|                                   |  HTTPS (TLS)  |                                      |
|   Weekly Sales Batch / Ledger     | ------------> | 1. TLS 1.2+ & Hostname Verification  |
|   (Closed Sunday 23:59:59Z)       |  Bearer Auth  | 2. Secret Redaction & Sanitization   |
|                                   |               | 3. Schema & Completeness Gate (v1.0) |
+-----------------------------------+               | 4. Deterministic Hash Deduplication  |
                                                    +--------------------------------------+
                                                                       |
                                                                       v
                                                    +--------------------------------------+
                                                    | SQLite Shadow Store & Evaluator      |
                                                    | (v2.2.8 6-Gate Activation Guard)    |
                                                    +--------------------------------------+
```

### 2.1 Transport Layer Security (TLS)
- **Mandatory HTTPS**: All remote endpoints must communicate over HTTPS using TLS 1.2 or TLS 1.3 with forward secrecy ciphers.
- **Strict Prohibition of Insecure HTTP**: Plain HTTP (`http://`) is strictly rejected for all remote hosts.
- **Loopback Exemption Policy**: The `--allow-insecure-http` flag is strictly restricted to local loopback test harnesses (`localhost`, `127.0.0.1`, `::1`, `testserver`). Remote URLs supplied with this flag are immediately rejected with `REJECTED_INVALID_ENDPOINT`.
- **Production Environment Lockdown**: If `DEMANDCAST_ENV` is set to `prod` or `production`, plain HTTP is unconditionally forbidden, rejecting even loopback addresses.

### 2.2 Authentication & Secret Management
- **Supported Schemes**:
  1. **HTTP Bearer Token**: Standard `Authorization: Bearer <token>` header.
  2. **API Key Header**: `X-API-Key: <key>` header.
  3. **OAuth2 Client Credentials Grant**: Upstream API gateway issues short-lived JWT access tokens via mTLS or client credentials flow.
- **Credential Storage & Loading**:
  - Tokens and API keys must be loaded strictly from environment variables (`DEMANDCAST_STREAM_AUTH_TOKEN`, `DEMANDCAST_STREAM_API_KEY`) or an authorized secret manager client (`secret_getter` callable).
  - Whitespace, surrounding quotes, and empty strings are stripped upon loading.
  - No secret values may be committed to version control, logged in application logs, or echoed in CLI error messages.
- **Secret Redaction**:
  - All exception messages, stack traces, and HTTP error payloads are scrubbed via `_sanitize_error_message()`.
  - Endpoint URLs with embedded credentials (`user:password@`) or query string parameters are scrubbed via `_mask_url()` prior to telemetry persistence or stdout display.

---

## 3. Upstream Enterprise Data Mapping to Canonical v1.0

Enterprise ERP and WMS databases use domain-specific table structures. The integration layer or upstream enterprise extract must transform source records into the DemandCast canonical `v1.0` JSON payload.

### 3.1 Field Mapping Matrix

| DemandCast Canonical v1.0 Field | Data Type | SAP S/4HANA (SD/CO-PA) | Manhattan Associates WMS | Blue Yonder (JDA) | Requirement & Validation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `series_id` | `string` | Composite: `KUNNR`_`MATNR` (Customer_Material) | `FACILITY_ID`_`ITEM_ID` | `LOC`_`ITEM` | **Mandatory**. Matches catalog series key (e.g. `0_1_100`). |
| `week_start` | `string` (YYYY-MM-DD) | Calculated W-MON from `VBRP-FKDAT` (Billing Date) | W-MON from `ORDER_LINE.SHIP_DATE` | W-MON from `HIST.STARTDATE` | **Mandatory**. Must fall strictly on Monday (`00:00:00Z`). |
| `sales` | `float` | `VBRP-FKIMG` (Billed Quantity) | `SUM(ACTUAL_QTY)` | `SUM(QTY)` | **Mandatory**. Non-negative float. Intermittent zeros permitted. |
| `price` | `float` | `VBRP-NETWR / VBRP-FKIMG` | Contract Unit Price | Unit Net Price | **Optional**. Default `0.0`. Validated if present. |
| `is_bucket_complete` | `boolean` | Ledger Close Status flag (`CO-PA` finalized) | Shift/Batch closure confirmation | Demand period finalized flag | **Mandatory for Live Pilot**. Must be `True` for all records. |
| `data_available_at` | `string` (ISO 8601) | Timestamp of data release / export run | Export generation timestamp | Extract publication timestamp | **Optional**. Evaluated against `as_of_timestamp` to prevent leakage. |

### 3.2 Canonical JSON Payload Example
```json
{
  "schema_version": "v1.0",
  "provider_id": "erp_sap_sandbox_us_east",
  "extraction_timestamp": "2023-01-09T02:15:00Z",
  "records": [
    {
      "series_id": "0_1_100",
      "week_start": "2023-01-02",
      "sales": 48.0,
      "price": 12.50,
      "is_bucket_complete": true,
      "data_available_at": "2023-01-09T01:00:00Z"
    },
    {
      "series_id": "0_1_200",
      "week_start": "2023-01-02",
      "sales": 0.0,
      "price": 8.00,
      "is_bucket_complete": true,
      "data_available_at": "2023-01-09T01:00:00Z"
    }
  ]
}
```

---

## 4. Accounting Week Closure & Temporal Contract

The mathematical integrity of DemandCast's calibration gate (Candidate C) and baseline models depends strictly on point-in-time temporal alignment.

```
       Weekly Accounting Bucket (Week of Jan 2 - Jan 8)             Execution As-Of (Jan 9)
  |---------------------------------------------------------|             |
Monday Jan 2                                           Sunday Jan 8   Monday Jan 9 02:00Z
00:00:00 UTC                                           23:59:59 UTC        (as_of)
  ^                                                         ^             ^
  |                                                         |             |
week_start                                             bucket_end   as_of_timestamp
```

1. **Monday Cadence (W-MON)**:
   - Each weekly record must start on Monday (`week_start` timestamp `YYYY-MM-DD`). Non-Monday records trigger an immediate `REJECTED_SCHEMA_VIOLATION`.
2. **Weekly Accounting Closure**:
   - Retail sales accumulate through Sunday at 23:59:59 UTC.
   - Batches published before the closure of Sunday financial reconciliation contain partial-week counts.
   - Any record with `is_bucket_complete == False` triggers an immediate `REJECTED_INCOMPLETE` result.
3. **Temporal Leakage Guard**:
   - The weekly bucket end ($\text{week\_start} + 6\text{d } 23\text{h } 59\text{m } 59\text{s}$) must be less than or equal to the forecast execution `as_of_timestamp`.
   - Records with bucket end times exceeding `as_of_timestamp` are rejected with `REJECTED_OUT_OF_ORDER`.

---

## 5. Live Pilot Activation Gates (The 6-Gate Rule)

Before the prospective live shadow pipeline will execute (`--mode prospective_live`), `ShadowPilotConfig.evaluate_activation_gates()` evaluates six non-negotiable gates:

```
[ Gate 1: Operator Authorization ]  --> is_active == True & promotion_policy == "MANUAL_SIGN_OFF_REQUIRED"
              |
              v
[ Gate 2: Provider Configuration ]  --> HTTPS endpoint configured, credentials present, status == "HEALTHY"
              |
              v
[ Gate 3: Remote Auth & Preflight]  --> Remote connectivity verified, HTTP 200 returned, preflight == "READY"
              |
              v
[ Gate 4: Data Completeness      ]  --> Schema v1.0 validated, W-MON cadence verified, is_bucket_complete == True
              |
              v
[ Gate 5: Persistence Integrity  ]  --> SQLite ShadowStore schema verified, read/write/telemetry operational
              |
              v
[ Gate 6: Shutdown Runbook       ]  --> Deactivation procedure verified, fail-closed rollbacks tested
              |
              v
[ ACTIVATION PERMITTED: Run 3-Model Prospective Shadow Inference ]
```

If **any** single gate fails, prospective execution terminates with exit code `1` without modifying the shadow database or executing inference.

---

## 6. Preflight Diagnostic & Readiness Verification

Operators and enterprise IT administrators can verify sandbox integration readiness using the CLI without activating inference:

```powershell
python scripts/run_vn1_shadow.py preflight
```

### Preflight Diagnostic Stages
1. **Stage 1 (Configuration)**: Validates endpoint format, HTTPS scheme, and presence of authentication credentials.
2. **Stage 2 (Network & Authentication)**: Sends authenticated HTTP probe to upstream endpoint; tests SSL handshake and token validity (checks for HTTP 401/403/500).
3. **Stage 3 (Schema & Version)**: Samples remote payload or fixture; validates structure against schema `v1.0`.
4. **Stage 4 (Completeness & Cadence)**: Confirms all records have `is_bucket_complete=True` and start on Monday.
5. **Stage 5 (Persistence)**: Inspects local SQLite database tables (`shadow_forecasts`, `shadow_actuals`, `shadow_runs`).
6. **Stage 6 (Governance Status)**: Verifies that even if all technical checks pass (`status="READY"`), `is_active` remains `False` until explicit operator sign-off.

---

## 7. IT Prerequisites & Hand-off Checklist for Enterprise ERP/WMS Team

For the standalone executive checklist, refer directly to [`docs/vn1_enterprise_it_handoff_checklist.md`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/docs/vn1_enterprise_it_handoff_checklist.md).

To activate prospective shadow forecasting with an enterprise partner, the enterprise IT team must provide and verify the following items:

- [ ] **1. Sandbox Endpoint Provisioning**:
  - Dedicated HTTPS endpoint (e.g., `https://sandbox-erp.partner.internal/api/v1/demandcast/weekly-feed`).
  - TLS 1.2 or 1.3 enforced.
  - Network firewall / VPN routing allowing ingress from DemandCast host.
- [ ] **2. Service Account & Authentication**:
  - Dedicated service principal with read-only access to sales and inventory ledgers.
  - Shared secret Bearer token or API key provisioned into secure vault.
  - Token expiration and rotation schedule agreed upon (minimum 90-day validity for pilot).
- [ ] **3. Extract Cadence & Batch Closure**:
  - Weekly batch generation scheduled for Monday at 02:00 UTC (following Sunday 23:59:59 UTC ledger close).
  - Explicit confirmation that `is_bucket_complete: true` is only asserted after financial batch closure.
- [ ] **4. Test Fixture Delivery**:
  - Delivery of one 100-series sample payload in canonical `v1.0` JSON format for integration preflight validation.
- [ ] **5. Runbook Sign-Off**:
  - Mutual agreement on the deactivation procedure documented in [`docs/vn1_shadow_pilot_operator_runbook.md`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/docs/vn1_shadow_pilot_operator_runbook.md).
