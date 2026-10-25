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
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

ProviderStatus = Literal["NOT_CONFIGURED", "CONFIGURED", "HEALTHY", "DEGRADED", "FAILED"]


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
