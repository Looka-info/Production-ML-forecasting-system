"""DemandCast VN1 Forecast-Origin Time Contract and Availability Semantics.

Establishes strict boundaries between:
- last_observed_week_start: The start date (Monday 00:00:00) of the latest completed 7-day historical bucket.
- as_of_timestamp: The point-in-time timestamp (UTC) at which the forecast is evaluated and generated.
- target_week_start: The start date (Monday 00:00:00) of each forward forecast bucket (h = 1..H).
- data_available_at: The timestamp when a historical bucket was finalized and made available.

Enforces fail-closed validation for prospective live feeds and deterministic period cutoffs for historical replay.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastTimeContract:
    """Explicit point-in-time forecast contract."""

    last_observed_week_start: pd.Timestamp
    as_of_timestamp: str  # ISO 8601 UTC
    horizon: int = 13
    mode: Literal["historical_replay", "prospective_live"] = "historical_replay"
    min_history_required: int = 12

    def __post_init__(self):
        if self.last_observed_week_start.day_of_week != 0:
            raise ValueError(
                f"last_observed_week_start must fall strictly on a Monday, got {self.last_observed_week_start} "
                f"(day of week {self.last_observed_week_start.day_of_week})."
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}.")

    @property
    def target_weeks(self) -> list[pd.Timestamp]:
        """Compute the ordered target forecast Mondays: last_observed_week_start + h weeks."""
        return [self.last_observed_week_start + pd.Timedelta(weeks=h) for h in range(1, self.horizon + 1)]

    @property
    def target_week_strings(self) -> list[str]:
        """ISO date strings for all forward target weeks."""
        return [dt.strftime("%Y-%m-%d") for dt in self.target_weeks]

    def validate_snapshot(
        self,
        snapshot_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate input snapshot against the explicit time contract.

        Enforces:
        1. Required columns presence.
        2. Monday date convention for all week_start values.
        3. Strict uniqueness: duplicate (series_id, week_start) pairs with conflicting values are rejected.
        4. Information Cutoff:
           - In prospective_live mode: fail closed if data completeness cannot be verified
             or if data_available_at > as_of_timestamp.
           - In historical_replay mode: observations after last_observed_week_start are purged;
             all observations <= last_observed_week_start are completed historical buckets.
        5. Zero-leakage guarantee: Target week actuals cannot enter features.
        """
        if snapshot_df is None or snapshot_df.empty:
            raise ValueError("Cannot validate empty or null snapshot.")

        df = snapshot_df.copy()

        # Check required columns
        req_cols = {"series_id", "week_start", "sales"}
        if not req_cols.issubset(set(df.columns)):
            missing = req_cols - set(df.columns)
            raise ValueError(f"Snapshot missing required time-contract columns: {missing}")

        # Parse and validate dates
        df["week_start"] = pd.to_datetime(df["week_start"])
        non_mondays = df[df["week_start"].dt.day_of_week != 0]
        if not non_mondays.empty:
            bad_examples = non_mondays["week_start"].dt.strftime("%Y-%m-%d").unique()[:3]
            raise ValueError(
                f"Time contract violation: {len(non_mondays)} observations do not fall on Monday. "
                f"Examples: {bad_examples}"
            )

        # Check duplicate (series_id, week_start)
        duplicates = df[df.duplicated(subset=["series_id", "week_start"], keep=False)]
        if not duplicates.empty:
            # Check if duplicates have conflicting sales
            conflicts = duplicates.groupby(["series_id", "week_start"])["sales"].nunique()
            has_conflicts = (conflicts > 1).any()
            if has_conflicts:
                raise ValueError("Duplicate week labels with conflicting sales values detected in snapshot.")
            else:
                # Deduplicate exact identical duplicates with warning
                logger.warning(f"Dropping {len(duplicates) // 2} identical duplicate observations in snapshot.")
                df = df.drop_duplicates(subset=["series_id", "week_start"]).copy()

        # Evaluate information cutoff & availability
        audit_metadata: dict[str, Any] = {
            "mode": self.mode,
            "last_observed_week_start": self.last_observed_week_start.strftime("%Y-%m-%d"),
            "as_of_timestamp": self.as_of_timestamp,
            "horizon": self.horizon,
            "target_weeks": self.target_week_strings,
            "rows_raw": len(df),
            "rows_excluded_post_cutoff": 0,
            "rows_excluded_incomplete": 0,
            "rows_eligible": 0,
            "status": "VALID",
        }

        # Check explicit data_available_at if present
        if "data_available_at" in df.columns:
            df["data_available_at"] = pd.to_datetime(df["data_available_at"], utc=True)
            as_of_dt = pd.to_datetime(self.as_of_timestamp, utc=True)
            future_available = df[df["data_available_at"] > as_of_dt]

            if self.mode == "prospective_live" and not future_available.empty:
                raise ValueError(
                    f"Prospective live time contract violation: {len(future_available)} rows have "
                    f"data_available_at > as_of_timestamp ({self.as_of_timestamp})."
                )
            # In historical replay or filtering, drop rows not yet available
            df = df[df["data_available_at"] <= as_of_dt].copy()

        # Check explicit is_bucket_complete if present
        if "is_bucket_complete" in df.columns:
            incomplete_buckets = df[~df["is_bucket_complete"].astype(bool)]
            if not incomplete_buckets.empty:
                if self.mode == "prospective_live":
                    raise ValueError(
                        f"Incomplete trailing weekly bucket detected in prospective live mode: "
                        f"{len(incomplete_buckets)} observations."
                    )
                # In replay mode, drop incomplete buckets
                audit_metadata["rows_excluded_incomplete"] = len(incomplete_buckets)
                df = df[df["is_bucket_complete"].astype(bool)].copy()

        # Prospective live mode fail-closed rule: if live and no availability info, require verification
        if (
            self.mode == "prospective_live"
            and "data_available_at" not in df.columns
            and "is_bucket_complete" not in df.columns
        ):
            # Check if trailing week is complete by checking if current time is past the week's end (Sunday 23:59:59)
            as_of_dt = pd.to_datetime(self.as_of_timestamp, utc=True)
            latest_bucket = df["week_start"].max()
            latest_bucket_end = latest_bucket + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
            latest_bucket_end_utc = (
                latest_bucket_end.tz_localize("UTC") if latest_bucket_end.tzinfo is None else latest_bucket_end
            )

            if as_of_dt < latest_bucket_end_utc:
                raise ValueError(
                    f"Prospective live time contract fail-closed: Latest bucket {latest_bucket.strftime('%Y-%m-%d')} "
                    f"ends at {latest_bucket_end_utc.isoformat()}, which is after as_of_timestamp {as_of_dt.isoformat()}. "
                    "Unclosed trailing bucket cannot be verified as complete."
                )

        # Enforce temporal cutoff: rows strictly <= last_observed_week_start
        rows_after_cutoff = (df["week_start"] > self.last_observed_week_start).sum()
        audit_metadata["rows_excluded_post_cutoff"] = int(rows_after_cutoff)
        eligible_df = df[df["week_start"] <= self.last_observed_week_start].copy()

        if eligible_df.empty:
            raise ValueError(
                f"No eligible observations remain on or before last_observed_week_start "
                f"{self.last_observed_week_start.strftime('%Y-%m-%d')}."
            )

        # Check minimum history requirement for evaluated series
        series_history_counts = eligible_df.groupby("series_id")["week_start"].nunique()
        insufficient_series = series_history_counts[series_history_counts < self.min_history_required]
        audit_metadata["series_total"] = int(eligible_df["series_id"].nunique())
        audit_metadata["series_with_sufficient_history"] = int(
            (series_history_counts >= self.min_history_required).sum()
        )
        audit_metadata["series_with_insufficient_history"] = len(insufficient_series)
        audit_metadata["rows_eligible"] = len(eligible_df)

        # Compute deterministic snapshot hash of eligible data
        sorted_bytes = (
            eligible_df[["series_id", "week_start", "sales"]]
            .sort_values(["series_id", "week_start"])
            .to_csv(index=False)
            .encode("utf-8")
        )
        audit_metadata["eligible_snapshot_hash"] = hashlib.sha256(sorted_bytes).hexdigest()

        return eligible_df, audit_metadata


def create_time_contract(
    origin_date: pd.Timestamp | str | None,
    as_of_timestamp: str | datetime | None = None,
    horizon: int = 13,
    mode: Literal["historical_replay", "prospective_live"] = "historical_replay",
) -> ForecastTimeContract:
    """Factory function for building a validated ForecastTimeContract."""
    if origin_date is None:
        raise ValueError("origin_date (last_observed_week_start) must be provided.")

    origin_ts = pd.to_datetime(origin_date)

    if as_of_timestamp is None:
        if mode == "historical_replay":
            # In historical replay, as_of_timestamp defaults to the end of the origin bucket
            as_of_str = (origin_ts + pd.Timedelta(days=7)).isoformat()
        else:
            as_of_str = datetime.now(timezone.utc).isoformat()
    elif isinstance(as_of_timestamp, datetime):
        as_of_str = as_of_timestamp.isoformat()
    else:
        as_of_str = str(as_of_timestamp)

    return ForecastTimeContract(
        last_observed_week_start=origin_ts,
        as_of_timestamp=as_of_str,
        horizon=horizon,
        mode=mode,
    )
