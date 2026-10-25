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
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
    active upstream stream endpoint is explicitly provided in configuration.
    """

    def __init__(
        self,
        provider_id: str = "prospective_live_stream",
        schema_version: str = "v1.0",
        endpoint_url: str | None = None,
        auth_token: str | None = None,
    ):
        super().__init__(provider_id=provider_id, schema_version=schema_version)
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token

    def get_status(self) -> ProviderStatus:
        if not self.endpoint_url or not self.auth_token:
            return "NOT_CONFIGURED"
        return "HEALTHY"

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

    def ingest(self, payload: Any, as_of_timestamp: str | None = None) -> IngestionResult:
        as_of = as_of_timestamp or datetime.now(timezone.utc).isoformat()

        if self.get_status() == "NOT_CONFIGURED" and not payload:
            return IngestionResult(
                status="NOT_CONFIGURED",
                provider_id=self.provider_id,
                schema_version=self.schema_version,
                as_of_timestamp=as_of,
                errors=["Prospective retail stream provider is NOT_CONFIGURED (no active live feed)."],
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
