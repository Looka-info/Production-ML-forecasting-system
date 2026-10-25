# DemandCast VN1 — Enterprise IT Sandbox Handoff Checklist

**Document Version**: v2.2.9  
**Effective Date**: 2026-10-26  
**Audience**: Enterprise Systems Architects, ERP/WMS Integration Engineers, Information Security Officers, ML Platform Engineers  
**System Target**: DemandCast VN1 Retail Forecasting Shadow Inference Pipeline  

---

## 1. Executive Summary & Non-Negotiable Governance

This checklist defines the technical and operational requirements for connecting an enterprise Enterprise Resource Planning (ERP) or Warehouse Management System (WMS) sandbox (e.g., SAP S/4HANA, Manhattan Associates WMS, Blue Yonder) to DemandCast VN1.

### Zero-Impact Shadow Governance
1. **Read-Only Shadow Testing**: DemandCast executes strictly in observational shadow mode. It generates candidate forecasts in parallel with the production model (`Baseline_MovingAvg12_Official`, pinned at `v1.0.0`) without altering replenishment proposals, inventory reorders, or customer dispatches.
2. **Zero Insecure Protocol Allowance**: Plain HTTP is strictly forbidden for all remote connections. Strict HTTPS (TLS 1.2+) is enforced at the network boundary.
3. **No Credential Sharing via Chat or Git**: In accordance with enterprise security standards, credentials must never be requested in chat, shared over email, or committed to Git. All secrets are loaded strictly from secure environment variables or a certified secret vault client.
4. **Fail-Closed Pilot Lock**: Even with a valid endpoint and token, the shadow pilot remains inactive (`is_active = False`) until an authorized operator explicitly signs off and all six activation gates pass.

---

## 2. Enterprise IT Handoff Checklist

Please review and confirm each prerequisite prior to scheduled sandbox onboarding:

| Area | Requirement | Technical Specification | Verification Status |
| :--- | :--- | :--- | :--- |
| **1. Endpoint** | Dedicated HTTPS Sandbox URL | FQDN with valid commercial or corporate CA SSL certificate. Minimum TLS 1.2 (TLS 1.3 preferred). Port 443. Plain HTTP (`http://`) is rejected fail-closed. | [ ] Pending / [ ] Provided |
| **2. Authentication** | Read-Only Service Account Token | HTTP Bearer Token (`Authorization: Bearer <token>`) or API Key (`X-API-Key: <key>`). Minimum 90-day validity. Injected via `DEMANDCAST_STREAM_AUTH_TOKEN`. | [ ] Pending / [ ] Provided |
| **3. Permissions** | Least-Privilege Scope | Read-only access to sales history / billed quantities. **Strictly no write, update, delete, or execution privileges.** | [ ] Verified Read-Only |
| **4. Schema** | Canonical `v1.0` JSON Payload | Matches DemandCast JSON Schema `v1.0` (`series_id`, `week_start`, `sales`, `price`, `is_bucket_complete`). | [ ] Verified Compatible |
| **5. Week Cadence** | W-MON Monday Boundary | All record `week_start` values must fall on Monday (`00:00:00Z`). Non-Monday records trigger schema rejection. | [ ] Verified W-MON |
| **6. Accounting State** | Closed Weekly Financial Ledger | Records published after Sunday 23:59:59Z financial close with `is_bucket_complete = true`. Incomplete batches fail closed. | [ ] Verified Complete |
| **7. Network** | Egress / Ingress Allowlisting | Firewall or VPN routing allowing HTTPS egress from DemandCast host to sandbox API gateway. | [ ] Firewall Configured |
| **8. Diagnostics** | Self-Service Preflight Sign-Off | Upstream connectivity, authentication, and payload schema verified using `python scripts/run_vn1_shadow.py preflight`. | [ ] Preflight READY |

---

## 3. Canonical Schema Contract (v1.0)

The upstream enterprise batch extract must conform to the canonical `v1.0` JSON payload specification:

```json
{
  "schema_version": "v1.0",
  "provider_id": "erp_enterprise_sandbox",
  "extraction_timestamp": "2023-01-09T02:00:00Z",
  "records": [
    {
      "series_id": "0_1_100",
      "week_start": "2023-01-02",
      "sales": 48.0,
      "price": 12.50,
      "is_bucket_complete": true,
      "data_available_at": "2023-01-09T01:30:00Z"
    },
    {
      "series_id": "0_1_200",
      "week_start": "2023-01-02",
      "sales": 72.0,
      "price": 8.00,
      "is_bucket_complete": true,
      "data_available_at": "2023-01-09T01:30:00Z"
    }
  ]
}
```

### Field Definitions
- `schema_version` (`string`, required): Must be strictly `"v1.0"`.
- `provider_id` (`string`, required): Unique identifier for upstream system (e.g., `"sap_s4hana_sandbox_us"`).
- `records` (`list[dict]`, required): Array of weekly sales records.
  - `series_id` (`string`, required): Composite identifier for series (e.g., `Client_Warehouse_Product`).
  - `week_start` (`string`, required): `YYYY-MM-DD` formatted date strictly on Monday.
  - `sales` (`float`, required): Non-negative units sold / dispatched during closed accounting week.
  - `price` (`float`, optional): Average selling price during week (default: `0.0`).
  - `is_bucket_complete` (`boolean`, required): Must be `true` for prospective inference. Batches with `false` fail closed.
  - `data_available_at` (`string`, optional): ISO 8601 timestamp of data publication.

---

## 4. Secure Credential Injection

Never send credentials via messaging, tickets, or email. Inject credentials securely into the runtime environment:

```bash
# Target Sandbox HTTPS Endpoint (TLS 1.2+ required)
export DEMANDCAST_STREAM_ENDPOINT_URL="https://sandbox-erp.partner.internal/api/v1/demandcast/weekly-feed"

# Secret Service Account Bearer Token
export DEMANDCAST_STREAM_AUTH_TOKEN="sec_live_retail_token_sample_string"

# Optional: HTTP Request Timeout in seconds (Default: 10.0)
export DEMANDCAST_STREAM_TIMEOUT_SECONDS="15.0"
```

*Note: DemandCast automatically masks URLs in telemetry and replaces secret tokens with `[REDACTED_SECRET]` in all logs, exception traces, and evaluation reports.*

---

## 5. Self-Service Preflight Command

Once environment variables are set, the enterprise integration engineer can verify connectivity without executing forecasts:

```powershell
python scripts/run_vn1_shadow.py preflight
```

### Expected Output for Healthy Preflight
- Overall Preflight Status: `READY`
- Provider Configuration: `PASSED`
- Remote Network & Authentication: `PASSED` (HTTP 200)
- Schema Validation: `PASSED` (`v1.0`)
- Accounting Completeness: `PASSED` (`is_bucket_complete = true`)
- Governance State: `is_active = False` (Pilot remains safely locked inactive)

---

## 6. Incident Escalation & Support Contacts

For integration defects, SSL handshake errors, or schema validation inquiries, contact the DemandCast engineering team:

- **ML Platform & Forecasting Engineering**: `mlops-support@demandcast.internal`
- **Security & Infrastructure Boundary**: `security-review@demandcast.internal`
- **Escalation Runbook**: [`docs/vn1_shadow_pilot_operator_runbook.md`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/docs/vn1_shadow_pilot_operator_runbook.md)
- **Detailed Integration Spec**: [`docs/vn1_erp_wms_sandbox_integration_spec.md`](file:///C:/Users/mumer/OneDrive/Documents/Production%20ML%20forecasting%20system/docs/vn1_erp_wms_sandbox_integration_spec.md)
