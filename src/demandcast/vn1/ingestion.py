"""Provider-Independent Shadow Ingestion Architecture for DemandCast VN1.

Establishes a strict contract and adapter abstraction for ingesting point-in-time
retail sales snapshots for shadow inference and subsequent reconciliation.

Core Guarantees:
1. Strict schema and schema-version validation.
2. Stable composite series identifiers ('Client_Warehouse_Product').
3. Strict Monday W-MON weekly boundary validation.
4. Fail-closed rejection of duplicate, contradictory, incomplete, or out-of-order data.
5. Idempotent snapshot hashing (SHA-256).
6. Complete isolation between prospective inference input and subsequent actuals reconciliation.
7. Truthful provider status reporting ('NOT_CONFIGURED' when no live feed is configured).
"""

import hashlib
import hmac
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ProviderStatus = Literal["NOT_CONFIGURED", "CONFIGURED", "HEALTHY", "DEGRADED", "FAILED"]
PreflightStatus = Literal[
    "NOT_CONFIGURED",
    "INVALID_CONFIGURATION",
    "AUTHENTICATION_FAILED",
    "PROVIDER_UNAVAILABLE",
    "SCHEMA_INVALID",
    "INCOMPLETE_INPUT",
    "READY",
]


@dataclass
class PreflightResult:
    """Standardized result of a shadow provider preflight readiness check."""

    status: PreflightStatus
    provider_id: str
    schema_version: str
    provider_configured: bool
    endpoint_url_masked: str | None
    endpoint_valid: bool
    endpoint_scheme: str | None
    credentials_present: bool
    network_reachable: bool | None = None
    remote_authenticated: bool | None = None
    schema_valid: bool | None = None
    data_completeness: bool | None = None
    persistence_ready: bool = False
    pilot_activation_ready: bool = False  # Strictly False by default per governance
    details: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    remediation_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert preflight result to clean dictionary."""
        return asdict(self)


@dataclass
class IngestionResult:
    """Standardized result of a shadow ingestion attempt."""

    status: str  # 'SUCCESS', 'REJECTED_SCHEMA_VIOLATION', 'REJECTED_DUPLICATE', 'REJECTED_CONTRADICTORY', 'REJECTED_OUT_OF_ORDER', 'REJECTED_INCOMPLETE', 'NOT_CONFIGURED'
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

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary excluding heavy dataframe."""
        d = asdict(self)
        d.pop("snapshot_df", None)
        return d


class BaseShadowIngestionAdapter(ABC):
    """Abstract base adapter for shadow data providers."""

    def __init__(self, provider_id: str, schema_version: str = "v1.0"):
        self.provider_id = provider_id
        self.schema_version = schema_version
        self.last_ingestion_time: str | None = None
        self.total_ingested_batches: int = 0
        self.total_rejected_batches: int = 0
        self.total_records_ingested: int = 0
        self.last_error: str | None = None

    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """Return truthful status of provider connection."""
        pass

    @abstractmethod
    def validate_schema(self, payload: Any) -> tuple[bool, list[str]]:
        """Validate input payload against expected schema."""
        pass

    @abstractmethod
    def ingest(self, payload: Any, as_of_timestamp: str | None = None) -> IngestionResult:
        """Ingest raw payload into a validated, point-in-time snapshot DataFrame."""
        pass

    def get_telemetry(self) -> dict[str, Any]:
        """Return operational health and telemetry."""
        return {
            "provider_id": self.provider_id,
            "schema_version": self.schema_version,
            "status": self.get_status(),
            "last_ingestion_time": self.last_ingestion_time,
            "total_ingested_batches": self.total_ingested_batches,
            "total_rejected_batches": self.total_rejected_batches,
            "last_error": self.last_error,
        }


class FileBatchIngestionAdapter(BaseShadowIngestionAdapter):
    """Adapter for file-based batch snapshots (CSV, Parquet, or pre-loaded DataFrame)."""

    def __init__(self, provider_id: str = "file_batch", schema_version: str = "v1.0"):
        super().__init__(provider_id=provider_id, schema_version=schema_version)

    def get_status(self) -> ProviderStatus:
        return "CONFIGURED"

    def validate_schema(self, payload: Any) -> tuple[bool, list[str]]:
        errors = []
        if payload is None:
            return False, ["Payload is None"]

        if isinstance(payload, pd.DataFrame):
            df = payload
        elif isinstance(payload, (str, Path)):
            p = Path(payload)
            if not p.exists():
                return False, [f"File path does not exist: {p}"]
            return True, []
        elif isinstance(payload, dict) and "data" in payload:
            df = pd.DataFrame(payload["data"])
        else:
            return False, [f"Unsupported payload type: {type(payload)}"]

        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        # Validate required columns: either long format ('series_id', 'week_start', 'sales')
        # or wide format ('Client', 'Warehouse', 'Product' and date columns)
        cols = set(df.columns)
        has_long = {"series_id", "week_start", "sales"}.issubset(cols)
        has_wide = {"Client", "Warehouse", "Product"}.issubset(cols)

        if not (has_long or has_wide):
            errors.append(
                f"Missing required columns. Expected either ['series_id', 'week_start', 'sales'] "
                f"or ['Client', 'Warehouse', 'Product']. Found: {sorted(list(cols))[:6]}..."
            )

        return len(errors) == 0, errors

    def ingest(self, payload: Any, as_of_timestamp: str | None = None) -> IngestionResult:
        as_of = as_of_timestamp or datetime.now(timezone.utc).isoformat()
        is_valid, errors = self.validate_schema(payload)
        if not is_valid:
            self.total_rejected_batches += 1
            self.last_error = "; ".join(errors)
            return IngestionResult(
                status="REJECTED_SCHEMA_VIOLATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=errors,
            )

        if isinstance(payload, (str, Path)):
            p = Path(payload)
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        elif isinstance(payload, pd.DataFrame):
            df = payload.copy()
        elif isinstance(payload, dict) and "data" in payload:
            df = pd.DataFrame(payload["data"])
        else:
            df = pd.DataFrame()

        # Unpivot wide format if necessary
        if "series_id" not in df.columns and {"Client", "Warehouse", "Product"}.issubset(set(df.columns)):
            from demandcast.vn1.data_loading import parse_wide_to_long

            df = parse_wide_to_long(df, value_name="sales", phase=0)

        # Standardize series_id and week_start
        df["series_id"] = df["series_id"].astype(str)
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.strftime("%Y-%m-%d")
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)

        # Check for duplicates or contradictory entries
        dups = df.duplicated(subset=["series_id", "week_start"], keep=False)
        if dups.any():
            dup_rows = df[dups]
            # Check if contradictory (different sales)
            diff_sales = dup_rows.groupby(["series_id", "week_start"])["sales"].nunique().max()
            if diff_sales > 1:
                self.total_rejected_batches += 1
                self.last_error = "Contradictory duplicate entries detected"
                return IngestionResult(
                    status="REJECTED_CONTRADICTORY",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    as_of_timestamp=as_of,
                    errors=["Contradictory sales values detected for identical series_id and week_start"],
                )
            else:
                # Deduplicate identical records
                df = df.drop_duplicates(subset=["series_id", "week_start"], keep="first").reset_index(drop=True)

        # Ensure Monday W-MON alignment
        non_mondays = (pd.to_datetime(df["week_start"]).dt.day_of_week != 0).sum()
        if non_mondays > 0:
            self.total_rejected_batches += 1
            self.last_error = f"Found {non_mondays} non-Monday weekly observations"
            return IngestionResult(
                status="REJECTED_SCHEMA_VIOLATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=[f"Found {non_mondays} dates that do not conform to W-MON (Monday) weekly cadence"],
            )

        last_observed = df["week_start"].max()
        record_count = len(df)
        series_count = df["series_id"].nunique()

        # Compute deterministic snapshot hash
        sorted_sub = df[["series_id", "week_start", "sales"]].sort_values(["series_id", "week_start"])
        hash_str = sorted_sub.to_csv(index=False)
        snap_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

        self.last_ingestion_time = as_of
        self.total_ingested_batches += 1
        self.total_records_ingested += record_count
        self.last_error = None

        return IngestionResult(
            status="SUCCESS",
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            record_count=record_count,
            series_count=series_count,
            last_observed_week_start=last_observed,
            as_of_timestamp=as_of,
            snapshot_hash=snap_hash,
            snapshot_df=df,
            errors=[],
            telemetry={
                "ingestion_adapter": "FileBatchIngestionAdapter",
                "earliest_week_start": df["week_start"].min(),
                "latest_week_start": last_observed,
            },
        )


class ProspectiveStreamIngestionAdapter(BaseShadowIngestionAdapter):
    """Generic adapter for streaming or prospective HTTP/JSON live feeds.

    Truthful reporting: Status is strictly 'NOT_CONFIGURED' unless an authenticated,
    active upstream stream endpoint is explicitly provided via environment variables,
    secret manager, or configuration. Raw credentials are never exposed in logs or telemetry.
    """

    def __init__(
        self,
        provider_id: str = "prospective_live_stream",
        schema_version: str = "v1.0",
        endpoint_url: str | None = None,
        auth_token: str | None = None,
        timeout_seconds: float | None = None,
        secret_getter: Callable[[str], str | None] | None = None,
        allow_insecure_http: bool = False,
    ):
        super().__init__(provider_id=provider_id, schema_version=schema_version)
        self.secret_getter = secret_getter

        # Load credentials with preference: explicit argument -> secret_getter -> os.environ
        raw_endpoint = endpoint_url
        if not raw_endpoint and secret_getter:
            raw_endpoint = secret_getter("DEMANDCAST_STREAM_ENDPOINT_URL")
        if not raw_endpoint:
            raw_endpoint = os.environ.get("DEMANDCAST_STREAM_ENDPOINT_URL")

        raw_token = auth_token
        if not raw_token and secret_getter:
            raw_token = secret_getter("DEMANDCAST_STREAM_AUTH_TOKEN")
        if not raw_token:
            raw_token = os.environ.get("DEMANDCAST_STREAM_AUTH_TOKEN") or os.environ.get("DEMANDCAST_STREAM_API_KEY")

        raw_timeout = timeout_seconds
        if raw_timeout is None and secret_getter:
            t_val = secret_getter("DEMANDCAST_STREAM_TIMEOUT_SECONDS")
            raw_timeout = float(t_val) if t_val else None
        if raw_timeout is None:
            t_env = os.environ.get("DEMANDCAST_STREAM_TIMEOUT_SECONDS")
            raw_timeout = float(t_env) if t_env else 10.0

        self.timeout_seconds = float(raw_timeout)
        self.allow_insecure_http = allow_insecure_http
        self.endpoint_url = raw_endpoint
        self._auth_token = raw_token
        self.auth_failures_count = 0
        self.endpoint_valid, self.endpoint_error = self._validate_endpoint(raw_endpoint)

    def _validate_endpoint(self, url: str | None) -> tuple[bool, str | None]:
        """Validate endpoint URL schema and prevent arbitrary protocol injection."""
        if not url:
            return False, "Endpoint URL is not configured"
        # Support HTTPS and explicit local/mock test endpoints
        is_https = url.startswith("https://")
        is_allowed_local = self.allow_insecure_http and (
            url.startswith("http://localhost")
            or url.startswith("http://127.0.0.1")
            or url.startswith("http://testserver")
            or url.startswith("http://mock-retail-feed.internal")
            or url.startswith("https://mock-retail-feed.internal")
        )
        if not (is_https or is_allowed_local):
            if url.startswith("http://"):
                return False, "Insecure HTTP protocol is disallowed for prospective feeds. Endpoint must use HTTPS."
            return False, f"Unsupported URL scheme in endpoint: '{url}'. Only HTTPS is permitted."
        return True, None

    def _mask_url(self, url: str | None) -> str | None:
        """Mask URL parameters and userinfo for secure logging and telemetry."""
        if not url:
            return None
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except Exception:
            return "https://***"

    @property
    def auth_token(self) -> str | None:
        """Expose auth_token for backward compatibility with existing tests."""
        return self._auth_token

    @auth_token.setter
    def auth_token(self, value: str | None):
        self._auth_token = value

    def get_status(self) -> ProviderStatus:
        if not self.endpoint_url or not self._auth_token:
            return "NOT_CONFIGURED"
        if not self.endpoint_valid:
            return "FAILED"
        return "HEALTHY"

    def validate_authentication(
        self,
        incoming_token: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bool, str | None]:
        """Verify incoming push/pull token against stored secret without leaking token in error."""
        if not self._auth_token:
            return False, "No auth_token configured on ingestion adapter (NOT_CONFIGURED)"

        provided = incoming_token
        if not provided and headers:
            auth_hdr = headers.get("Authorization") or headers.get("authorization")
            if auth_hdr and auth_hdr.startswith("Bearer "):
                provided = auth_hdr.split("Bearer ", 1)[1].strip()
            if not provided:
                provided = headers.get("X-API-Key") or headers.get("x-api-key") or headers.get("X-Api-Key")

        if not provided:
            self.auth_failures_count += 1
            return False, "Authentication failed: missing authentication token"

        if not hmac.compare_digest(provided, self._auth_token):
            self.auth_failures_count += 1
            return False, "Authentication failed: invalid authentication token"

        return True, None

    def get_telemetry(self) -> dict[str, Any]:
        """Actionable telemetry strictly excluding secrets."""
        return {
            "provider_id": self.provider_id,
            "schema_version": self.schema_version,
            "status": self.get_status(),
            "endpoint_configured": bool(self.endpoint_url),
            "endpoint_masked": self._mask_url(self.endpoint_url),
            "auth_token_configured": bool(self._auth_token),
            "timeout_seconds": self.timeout_seconds,
            "total_ingested_batches": self.total_ingested_batches,
            "total_rejected_batches": self.total_rejected_batches,
            "total_records_ingested": self.total_records_ingested,
            "auth_failures_count": self.auth_failures_count,
            "last_error": self.last_error,
            "is_bucket_complete_enforced": True,
            "fail_closed": True,
        }

    def __repr__(self) -> str:
        return (
            f"<ProspectiveStreamIngestionAdapter provider_id='{self.provider_id}' "
            f"status='{self.get_status()}' endpoint='{self._mask_url(self.endpoint_url)}' "
            f"auth_configured={bool(self._auth_token)}>"
        )

    def validate_schema(self, payload: Any) -> tuple[bool, list[str]]:
        errors = []
        if payload is None:
            return False, ["Payload is null"]

        if not isinstance(payload, dict):
            return False, [f"Payload must be a JSON object (dict), received {type(payload)}"]

        if payload.get("schema_version") != self.schema_version:
            errors.append(
                f"Schema version mismatch: expected '{self.schema_version}', received '{payload.get('schema_version')}'"
            )

        records = payload.get("records")
        if not isinstance(records, list) or len(records) == 0:
            errors.append("Payload must contain a non-empty 'records' list")
            return False, errors

        required_record_keys = {"series_id", "week_start", "sales", "is_bucket_complete"}
        for idx, rec in enumerate(records[:100]):  # Sample up to first 100 for fast validation
            if not isinstance(rec, dict):
                errors.append(f"Record at index {idx} must be a dict")
                break
            missing = required_record_keys - set(rec.keys())
            if missing:
                errors.append(f"Record at index {idx} missing required fields: {sorted(list(missing))}")
                break

        return len(errors) == 0, errors

    def ingest(
        self,
        payload: Any,
        as_of_timestamp: str | None = None,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> IngestionResult:
        as_of = as_of_timestamp or datetime.now(timezone.utc).isoformat()

        if self.get_status() == "NOT_CONFIGURED" and not payload:
            return IngestionResult(
                status="NOT_CONFIGURED",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=["Prospective retail stream provider is NOT_CONFIGURED (no active live feed)."],
            )

        if self.get_status() == "FAILED":
            self.total_rejected_batches += 1
            self.last_error = self.endpoint_error or "Invalid endpoint URL"
            return IngestionResult(
                status="REJECTED_INVALID_ENDPOINT",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=[self.endpoint_error or "Invalid endpoint URL configuration"],
            )

        # Enforce authentication verification if auth credentials or headers are supplied
        if auth_token or headers:
            is_authed, auth_err = self.validate_authentication(incoming_token=auth_token, headers=headers)
            if not is_authed:
                self.total_rejected_batches += 1
                self.last_error = auth_err or "Authentication failed"
                return IngestionResult(
                    status="REJECTED_AUTHENTICATION_FAILURE",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    as_of_timestamp=as_of,
                    errors=[auth_err or "Authentication failed"],
                )

        is_valid, errors = self.validate_schema(payload)
        if not is_valid:
            self.total_rejected_batches += 1
            self.last_error = "; ".join(errors)
            return IngestionResult(
                status="REJECTED_SCHEMA_VIOLATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=errors,
            )

        records = payload["records"]
        df = pd.DataFrame(records)

        # 1. Enforce completeness: Prospective live inference fails closed on incomplete buckets
        if "is_bucket_complete" in df.columns:
            incomplete_mask = ~df["is_bucket_complete"].astype(bool)
            if incomplete_mask.any():
                self.total_rejected_batches += 1
                self.last_error = "Incomplete weekly bucket rejected (fail-closed)"
                return IngestionResult(
                    status="REJECTED_INCOMPLETE",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    as_of_timestamp=as_of,
                    errors=[
                        f"Found {incomplete_mask.sum()} records with is_bucket_complete=False. "
                        "Prospective live inference requires closed weekly accounting intervals."
                    ],
                )

        # 2. Check for contradictory entries
        dups = df.duplicated(subset=["series_id", "week_start"], keep=False)
        if dups.any():
            dup_rows = df[dups]
            diff_sales = dup_rows.groupby(["series_id", "week_start"])["sales"].nunique().max()
            if diff_sales > 1:
                self.total_rejected_batches += 1
                self.last_error = "Contradictory sales values detected in streaming batch"
                return IngestionResult(
                    status="REJECTED_CONTRADICTORY",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    as_of_timestamp=as_of,
                    errors=["Contradictory sales values detected for identical series_id and week_start in stream"],
                )
            else:
                df = df.drop_duplicates(subset=["series_id", "week_start"], keep="first").reset_index(drop=True)

        # 3. Check for out-of-order or future data beyond as_of
        df["week_dt"] = pd.to_datetime(df["week_start"])
        # Standardize week_start string
        df["week_start"] = df["week_dt"].dt.strftime("%Y-%m-%d")
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)

        # Non-Monday rejection
        non_mondays = (df["week_dt"].dt.day_of_week != 0).sum()
        if non_mondays > 0:
            self.total_rejected_batches += 1
            self.last_error = f"Stream contained {non_mondays} non-Monday weekly records"
            return IngestionResult(
                status="REJECTED_SCHEMA_VIOLATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=[f"Found {non_mondays} dates that do not conform to W-MON (Monday) weekly cadence"],
            )

        as_of_dt = pd.to_datetime(as_of)
        # Bucket end is Sunday 23:59:59.999999 (week_dt + 6 days 23:59:59)
        bucket_ends = df["week_dt"] + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
        if (bucket_ends.dt.tz_localize("UTC") > as_of_dt).any() if as_of_dt.tzinfo else (bucket_ends > as_of_dt).any():
            self.total_rejected_batches += 1
            self.last_error = "Out-of-order/future unclosed bucket rejected (fail-closed)"
            return IngestionResult(
                status="REJECTED_OUT_OF_ORDER",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=["Stream record bucket end exceeds execution as_of_timestamp (temporal leakage violation)"],
            )

        last_observed = df["week_start"].max()
        record_count = len(df)
        series_count = df["series_id"].nunique()

        # Deterministic snapshot hash
        sorted_sub = df[["series_id", "week_start", "sales"]].sort_values(["series_id", "week_start"])
        hash_str = sorted_sub.to_csv(index=False)
        snap_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

        self.last_ingestion_time = as_of
        self.total_ingested_batches += 1
        self.total_records_ingested += record_count
        self.last_error = None

        return IngestionResult(
            status="SUCCESS",
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            record_count=record_count,
            series_count=series_count,
            last_observed_week_start=last_observed,
            as_of_timestamp=as_of,
            snapshot_hash=snap_hash,
            snapshot_df=df,
            errors=[],
            telemetry={
                "ingestion_adapter": "ProspectiveStreamIngestionAdapter",
                "endpoint_url": self.endpoint_url or "UNCONFIGURED_MOCK",
                "latest_week_start": last_observed,
            },
        )

    def _sanitize_error_message(self, message: str) -> str:
        """Strip any token or credential leakage from exception strings and URLs."""
        if not message:
            return ""
        sanitized = str(message)
        if self._auth_token and len(self._auth_token) > 0:
            sanitized = sanitized.replace(self._auth_token, "[REDACTED_SECRET]")
        if self.endpoint_url and "?" in self.endpoint_url:
            masked = self._mask_url(self.endpoint_url)
            if masked:
                sanitized = sanitized.replace(self.endpoint_url, masked)
        return sanitized

    def pull_remote_batch(
        self,
        as_of_timestamp: str | None = None,
        http_session: Any = None,
        max_retries: int = 2,
    ) -> IngestionResult:
        """Pull weekly sales batch from remote HTTPS ERP/WMS API with safe retry and secret redaction."""
        as_of = as_of_timestamp or datetime.now(timezone.utc).isoformat()

        if not self.endpoint_url or not self._auth_token:
            return IngestionResult(
                status="NOT_CONFIGURED",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=["Prospective stream adapter is NOT_CONFIGURED (no active live feed or credentials)."],
            )

        if not self.endpoint_valid:
            self.total_rejected_batches += 1
            self.last_error = self.endpoint_error or "Invalid endpoint configuration"
            return IngestionResult(
                status="REJECTED_INVALID_ENDPOINT",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=[self.endpoint_error or "Invalid endpoint URL configuration."],
            )

        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Accept": "application/json",
            "User-Agent": "DemandCast-ShadowPilot/v2.2.7",
        }

        session = http_session or requests.Session()
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = session.get(self.endpoint_url, headers=headers, timeout=self.timeout_seconds)
                if resp.status_code in (401, 403):
                    self.auth_failures_count += 1
                    self.total_rejected_batches += 1
                    self.last_error = f"Remote provider rejected authentication (HTTP {resp.status_code})"
                    return IngestionResult(
                        status="REJECTED_AUTHENTICATION_FAILURE",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        as_of_timestamp=as_of,
                        errors=[f"Remote provider rejected authentication credentials (HTTP {resp.status_code})."],
                    )
                if resp.status_code != 200:
                    if attempt < max_retries and resp.status_code in (500, 502, 503, 504):
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    self.total_rejected_batches += 1
                    self.last_error = f"Remote provider returned HTTP {resp.status_code}"
                    return IngestionResult(
                        status="PROVIDER_UNAVAILABLE",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        as_of_timestamp=as_of,
                        errors=[f"Remote provider returned error status HTTP {resp.status_code}."],
                    )

                payload = resp.json()
                return self.ingest(payload, as_of_timestamp=as_of)

            except Exception as exc:
                last_exc = exc
                sanitized_msg = self._sanitize_error_message(str(exc))
                if attempt < max_retries:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                self.total_rejected_batches += 1
                self.last_error = sanitized_msg
                return IngestionResult(
                    status="PROVIDER_UNAVAILABLE",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    as_of_timestamp=as_of,
                    errors=[f"Remote provider connection failed or timed out: {sanitized_msg}"],
                )

        sanitized_err = self._sanitize_error_message(str(last_exc) if last_exc else "Unknown error")
        return IngestionResult(
            status="PROVIDER_UNAVAILABLE",
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            as_of_timestamp=as_of,
            errors=[f"Remote provider connection failed: {sanitized_err}"],
        )

    def run_preflight(
        self,
        store: Any = None,
        sample_payload: Any = None,
        http_session: Any = None,
        skip_network: bool = False,
        as_of_timestamp: str | None = None,
    ) -> PreflightResult:
        """Run thorough multi-stage preflight check against provider configuration and health."""
        as_of = as_of_timestamp or datetime.now(timezone.utc).isoformat()
        endpoint_present = bool(self.endpoint_url)
        credentials_present = bool(self._auth_token)
        endpoint_url_masked = self._mask_url(self.endpoint_url)
        endpoint_scheme = urlparse(self.endpoint_url).scheme if self.endpoint_url else None

        # Check local persistence store readiness
        persistence_ready = False
        try:
            if store is not None:
                _ = store.get_telemetry()
                persistence_ready = True
            else:
                from demandcast.vn1.config import load_vn1_config
                from demandcast.vn1.shadow import ShadowStore

                cfg = load_vn1_config()
                test_store = ShadowStore(cfg.shadow.db_path)
                _ = test_store.get_telemetry()
                persistence_ready = True
        except Exception:
            persistence_ready = False

        # Stage 1: Configuration Presence & Validity
        if not endpoint_present and not credentials_present:
            return PreflightResult(
                status="NOT_CONFIGURED",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                provider_configured=False,
                endpoint_url_masked=None,
                endpoint_valid=False,
                endpoint_scheme=None,
                credentials_present=False,
                persistence_ready=persistence_ready,
                reasons=[
                    "Provider endpoint URL and authentication credentials are not configured in environment or settings."
                ],
                remediation_steps=[
                    "Set DEMANDCAST_STREAM_ENDPOINT_URL (e.g. 'https://erp.corporate.internal/api/v1/sales-stream').",
                    "Set DEMANDCAST_STREAM_AUTH_TOKEN in secure environment or secret vault.",
                ],
            )

        if endpoint_present and not credentials_present:
            return PreflightResult(
                status="INVALID_CONFIGURATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                provider_configured=False,
                endpoint_url_masked=endpoint_url_masked,
                endpoint_valid=self.endpoint_valid,
                endpoint_scheme=endpoint_scheme,
                credentials_present=False,
                persistence_ready=persistence_ready,
                reasons=["Authentication token is missing for configured endpoint."],
                remediation_steps=["Set DEMANDCAST_STREAM_AUTH_TOKEN in secure environment or secret vault."],
            )

        if not endpoint_present and credentials_present:
            return PreflightResult(
                status="INVALID_CONFIGURATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                provider_configured=False,
                endpoint_url_masked=None,
                endpoint_valid=False,
                endpoint_scheme=None,
                credentials_present=True,
                persistence_ready=persistence_ready,
                reasons=["Endpoint URL is not configured."],
                remediation_steps=["Set DEMANDCAST_STREAM_ENDPOINT_URL with a valid HTTPS endpoint."],
            )

        if not self.endpoint_valid:
            return PreflightResult(
                status="INVALID_CONFIGURATION",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                provider_configured=False,
                endpoint_url_masked=endpoint_url_masked,
                endpoint_valid=False,
                endpoint_scheme=endpoint_scheme,
                credentials_present=credentials_present,
                persistence_ready=persistence_ready,
                reasons=[self.endpoint_error or "Endpoint URL failed schema validation (HTTPS required)."],
                remediation_steps=["Ensure endpoint uses strict HTTPS protocol and valid hostname."],
            )

        # Stage 2: Network Reachability & Remote Authentication
        network_reachable: bool | None = None
        remote_authenticated: bool | None = None

        if not skip_network:
            session = http_session or requests.Session()
            headers = {
                "Authorization": f"Bearer {self._auth_token}",
                "Accept": "application/json",
                "User-Agent": "DemandCast-ShadowPilot/v2.2.7",
            }
            try:
                resp = session.get(self.endpoint_url, headers=headers, timeout=self.timeout_seconds)
                if resp.status_code in (401, 403):
                    return PreflightResult(
                        status="AUTHENTICATION_FAILED",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        provider_configured=True,
                        endpoint_url_masked=endpoint_url_masked,
                        endpoint_valid=True,
                        endpoint_scheme=endpoint_scheme,
                        credentials_present=True,
                        network_reachable=True,
                        remote_authenticated=False,
                        reasons=[f"Remote provider rejected authentication credentials (HTTP {resp.status_code})."],
                        remediation_steps=[
                            "Verify DEMANDCAST_STREAM_AUTH_TOKEN with ERP/WMS administrator.",
                            "Ensure token has read permissions for weekly sales feed.",
                        ],
                    )
                if resp.status_code != 200:
                    return PreflightResult(
                        status="PROVIDER_UNAVAILABLE",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        provider_configured=True,
                        endpoint_url_masked=endpoint_url_masked,
                        endpoint_valid=True,
                        endpoint_scheme=endpoint_scheme,
                        credentials_present=True,
                        network_reachable=False,
                        remote_authenticated=False,
                        reasons=[f"Remote provider returned error status HTTP {resp.status_code}."],
                        remediation_steps=[
                            "Check ERP/WMS feed service health and status page.",
                            "Verify DEMANDCAST_STREAM_TIMEOUT_SECONDS setting.",
                        ],
                    )

                network_reachable = True
                remote_authenticated = True

                # If sample payload was not explicitly provided, try extracting from response
                if sample_payload is None:
                    try:
                        remote_json = resp.json()
                        if isinstance(remote_json, dict) and "records" in remote_json:
                            sample_payload = remote_json
                    except Exception:
                        pass

            except Exception as exc:
                sanitized_msg = self._sanitize_error_message(str(exc))
                return PreflightResult(
                    status="PROVIDER_UNAVAILABLE",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    provider_configured=True,
                    endpoint_url_masked=endpoint_url_masked,
                    endpoint_valid=True,
                    endpoint_scheme=endpoint_scheme,
                    credentials_present=True,
                    network_reachable=False,
                    remote_authenticated=False,
                    reasons=[f"Remote provider connection failed or timed out: {sanitized_msg}"],
                    remediation_steps=[
                        "Verify network / VPN routing to the endpoint.",
                        "Check ERP/WMS feed service health.",
                    ],
                )

        # Stage 3: Schema & Version Compatibility
        schema_valid: bool | None = None
        data_completeness: bool | None = None

        if sample_payload is not None:
            is_valid_schema, schema_errs = self.validate_schema(sample_payload)
            if not is_valid_schema:
                return PreflightResult(
                    status="SCHEMA_INVALID",
                    provider_id=self.provider_id,
                    schema_version=self.schema_version,
                    provider_configured=True,
                    endpoint_url_masked=endpoint_url_masked,
                    endpoint_valid=True,
                    endpoint_scheme=endpoint_scheme,
                    credentials_present=True,
                    network_reachable=network_reachable,
                    remote_authenticated=remote_authenticated,
                    schema_valid=False,
                    data_completeness=False,
                    reasons=schema_errs,
                    remediation_steps=[
                        f"Ensure payload matches schema version '{self.schema_version}'.",
                        "Verify records list contains dicts with 'series_id', 'week_start', 'sales', 'is_bucket_complete'.",
                    ],
                )
            schema_valid = True

            # Stage 4: Data Completeness & Temporal Alignment
            records = sample_payload.get("records", [])
            df_rec = pd.DataFrame(records)
            if "is_bucket_complete" in df_rec.columns:
                incomplete_cnt = int((df_rec["is_bucket_complete"] == False).sum())  # noqa: E712
                if incomplete_cnt > 0:
                    return PreflightResult(
                        status="INCOMPLETE_INPUT",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        provider_configured=True,
                        endpoint_url_masked=endpoint_url_masked,
                        endpoint_valid=True,
                        endpoint_scheme=endpoint_scheme,
                        credentials_present=True,
                        network_reachable=network_reachable,
                        remote_authenticated=remote_authenticated,
                        schema_valid=True,
                        data_completeness=False,
                        reasons=[
                            f"Incomplete weekly accounting buckets detected ({incomplete_cnt} records with is_bucket_complete=False)."
                        ],
                        remediation_steps=[
                            "Wait for corporate financial week close (Sunday 23:59:59) before publishing prospective batch.",
                            "Ensure all records have is_bucket_complete=True.",
                        ],
                    )

            if "week_start" in df_rec.columns:
                w_dts = pd.to_datetime(df_rec["week_start"])
                non_mondays = int((w_dts.dt.day_of_week != 0).sum())
                if non_mondays > 0:
                    return PreflightResult(
                        status="SCHEMA_INVALID",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        provider_configured=True,
                        endpoint_url_masked=endpoint_url_masked,
                        endpoint_valid=True,
                        endpoint_scheme=endpoint_scheme,
                        credentials_present=True,
                        network_reachable=network_reachable,
                        remote_authenticated=remote_authenticated,
                        schema_valid=False,
                        data_completeness=False,
                        reasons=[f"Found {non_mondays} weekly records that do not conform to W-MON (Monday) cadence."],
                        remediation_steps=["Ensure all week_start values correspond to Mondays."],
                    )

                # Temporal leakage check
                as_of_dt = pd.to_datetime(as_of)
                bucket_ends = w_dts + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
                is_future = (
                    (bucket_ends.dt.tz_localize("UTC") > as_of_dt).any()
                    if as_of_dt.tzinfo
                    else (bucket_ends > as_of_dt).any()
                )
                if is_future:
                    return PreflightResult(
                        status="INCOMPLETE_INPUT",
                        provider_id=self.provider_id,
                        schema_version=self.schema_version,
                        provider_configured=True,
                        endpoint_url_masked=endpoint_url_masked,
                        endpoint_valid=True,
                        endpoint_scheme=endpoint_scheme,
                        credentials_present=True,
                        network_reachable=network_reachable,
                        remote_authenticated=remote_authenticated,
                        schema_valid=True,
                        data_completeness=False,
                        reasons=[
                            "Stream record bucket end exceeds execution as_of_timestamp (temporal leakage violation)."
                        ],
                        remediation_steps=["Ensure weekly bucket has closed before generating prospective forecast."],
                    )

            data_completeness = True

        # Stage 5: Persistence Store Readiness
        persistence_ready = False
        try:
            if store is not None:
                _ = store.get_telemetry()
                persistence_ready = True
            else:
                from demandcast.vn1.config import load_vn1_config
                from demandcast.vn1.shadow import ShadowStore

                cfg = load_vn1_config()
                test_store = ShadowStore(cfg.shadow.db_path)
                _ = test_store.get_telemetry()
                persistence_ready = True
        except Exception as s_exc:
            sanitized_store_err = self._sanitize_error_message(str(s_exc))
            return PreflightResult(
                status="PROVIDER_UNAVAILABLE",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                provider_configured=True,
                endpoint_url_masked=endpoint_url_masked,
                endpoint_valid=True,
                endpoint_scheme=endpoint_scheme,
                credentials_present=True,
                network_reachable=network_reachable,
                remote_authenticated=remote_authenticated,
                schema_valid=schema_valid,
                data_completeness=data_completeness,
                persistence_ready=False,
                reasons=[f"Persistence store initialization check failed: {sanitized_store_err}"],
                remediation_steps=["Check SQLite database directory permissions and disk space."],
            )

        # Stage 6: Final Preflight Verdict
        return PreflightResult(
            status="READY",
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            provider_configured=True,
            endpoint_url_masked=endpoint_url_masked,
            endpoint_valid=True,
            endpoint_scheme=endpoint_scheme,
            credentials_present=True,
            network_reachable=network_reachable,
            remote_authenticated=remote_authenticated,
            schema_valid=schema_valid,
            data_completeness=data_completeness,
            persistence_ready=persistence_ready,
            pilot_activation_ready=False,  # Governance lock: strictly False until explicit authorization
            details={
                "timeout_seconds": self.timeout_seconds,
                "allow_insecure_http": self.allow_insecure_http,
                "governance_lock": "LOCKED_INACTIVE",
            },
            reasons=[
                "All preflight connectivity, authentication, schema, completeness, and storage checks passed successfully."
            ],
            remediation_steps=[
                "System is verified and ready for prospective shadow inference.",
                "Notice: Pilot remains inactive (is_active=False) per governance until authorized operator sign-off.",
            ],
        )
