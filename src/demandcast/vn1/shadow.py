"""DemandCast v2.2 — Production-Safe Shadow Forecasting Pipeline.

Executes active production model and experimental Candidate D on the same point-in-time
input snapshot, persists forecasts idempotently to a dedicated SQLite shadow store,
isolates candidate failures from production, and reconciles delayed actuals without
overwriting historical predictions.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from demandcast.config import get_project_root
from demandcast.vn1.baselines import VN1MovingAverageForecaster
from demandcast.vn1.config import VN1Config, load_vn1_config
from demandcast.vn1.metrics import compute_all_vn1_metrics
from demandcast.vn1.registry import VN1ModelRegistry
from demandcast.vn1.seasonal_calibration import (
    CalibratedForecaster,
    SeasonalCalibrationConfig,
)
from demandcast.vn1.time_contract import create_time_contract

logger = logging.getLogger(__name__)

ShadowRunStatus = Literal[
    "NOT_CONFIGURED",
    "INCOMPLETE_INPUT_REJECTED",
    "READY",
    "RUNNING",
    "SUCCEEDED",
    "PARTIAL_FAILURE",
    "FAILED",
]


@dataclass
class ShadowForecastRecord:
    """Immutable structured record representing an individual series-horizon forecast."""

    record_id: str
    run_id: str
    dataset_provider: str
    snapshot_hash: str
    series_id: str
    client_id: str | None
    warehouse_id: str | None
    product_id: str | None
    origin_date: str
    target_week: str
    horizon_step: int
    model_identifier: str  # e.g., 'production_v1.0.0' or 'candidate_d_v2.1'
    model_name: str  # e.g., 'Baseline_MovingAvg12_Official'
    model_version: str  # e.g., 'v1.0.0' or 'v2.1-experimental'
    is_candidate: bool
    predicted_sales: float
    raw_predicted_sales: float | None
    calibration_factor: float | None
    calibration_metadata: str | None  # JSON string
    execution_timestamp: str
    forecast_status: str  # 'SUCCESS' or 'FAILED'
    error_message: str | None = None
    actual_sales: float | None = None
    actual_observed_at: str | None = None
    actual_source: str | None = None
    reconciliation_timestamp: str | None = None
    reconciliation_status: str = "PENDING_ACTUAL"  # 'PENDING_ACTUAL' | 'MATCHED' | 'CONFLICT_REJECTED'
    as_of_timestamp: str | None = None
    last_observed_week_start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        d = asdict(self)
        d["is_candidate"] = 1 if self.is_candidate else 0
        return d


class ShadowStore:
    """Thread-safe SQLite store for persisting shadow forecasts, actuals, and run metadata."""

    def __init__(self, db_path: Path | str | None = None):
        root = get_project_root()
        if db_path is None:
            self.db_path = root / "data" / "shadow" / "shadow_store.db"
        else:
            self.db_path = Path(db_path) if Path(db_path).is_absolute() else root / db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and unique indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Shadow Runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shadow_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset_provider TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    origin_date TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    series_count INTEGER NOT NULL,
                    production_status TEXT NOT NULL,
                    candidate_status TEXT NOT NULL,
                    production_version TEXT NOT NULL,
                    candidate_version TEXT NOT NULL,
                    forecasts_count INTEGER NOT NULL,
                    awaiting_actuals_count INTEGER NOT NULL,
                    error_message TEXT
                )
            """)

            # 2. Shadow Forecasts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shadow_forecasts (
                    record_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    dataset_provider TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    client_id TEXT,
                    warehouse_id TEXT,
                    product_id TEXT,
                    origin_date TEXT NOT NULL,
                    target_week TEXT NOT NULL,
                    horizon_step INTEGER NOT NULL,
                    model_identifier TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    is_candidate INTEGER NOT NULL,
                    predicted_sales REAL NOT NULL,
                    raw_predicted_sales REAL,
                    calibration_factor REAL,
                    calibration_metadata TEXT,
                    execution_timestamp TEXT NOT NULL,
                    forecast_status TEXT NOT NULL,
                    error_message TEXT,
                    actual_sales REAL,
                    actual_observed_at TEXT,
                    actual_source TEXT,
                    reconciliation_timestamp TEXT,
                    reconciliation_status TEXT NOT NULL DEFAULT 'PENDING_ACTUAL',
                    UNIQUE(snapshot_hash, origin_date, target_week, series_id, model_identifier)
                )
            """)

            # 3. Shadow Actuals
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shadow_actuals (
                    actual_id TEXT PRIMARY KEY,
                    series_id TEXT NOT NULL,
                    week_start TEXT NOT NULL,
                    sales REAL NOT NULL,
                    source TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    snapshot_hash TEXT,
                    UNIQUE(series_id, week_start, source)
                )
            """)

            # Column migrations for time-contract transparency
            for tbl, col_def in [
                ("shadow_runs", "as_of_timestamp TEXT"),
                ("shadow_runs", "last_observed_week_start TEXT"),
                ("shadow_forecasts", "as_of_timestamp TEXT"),
                ("shadow_forecasts", "last_observed_week_start TEXT"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col_def}")
                except Exception:
                    pass

            # Indexes for fast lookup and reconciliation
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON shadow_forecasts (series_id, target_week, reconciliation_status)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_run ON shadow_forecasts (run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actuals_lookup ON shadow_actuals (series_id, week_start)")
            conn.commit()

    def record_run(self, run_dict: dict[str, Any]) -> None:
        """Insert or replace shadow run metadata."""
        cols = [
            "run_id",
            "created_at",
            "status",
            "dataset_provider",
            "mode",
            "snapshot_hash",
            "origin_date",
            "horizon",
            "series_count",
            "production_status",
            "candidate_status",
            "production_version",
            "candidate_version",
            "forecasts_count",
            "awaiting_actuals_count",
            "error_message",
            "as_of_timestamp",
            "last_observed_week_start",
        ]
        placeholders = ", ".join(["?"] * len(cols))
        vals = [run_dict.get(c) for c in cols]
        sql = f"INSERT OR REPLACE INTO shadow_runs ({', '.join(cols)}) VALUES ({placeholders})"

        with self._get_connection() as conn:
            conn.execute(sql, vals)
            conn.commit()

    def save_forecasts(self, records: list[ShadowForecastRecord]) -> dict[str, int]:
        """Persist forecast records idempotently. Rerunning same snapshot skips duplicates."""
        if not records:
            return {"inserted": 0, "skipped_duplicates": 0}

        cols = [
            "record_id",
            "run_id",
            "dataset_provider",
            "snapshot_hash",
            "series_id",
            "client_id",
            "warehouse_id",
            "product_id",
            "origin_date",
            "target_week",
            "horizon_step",
            "model_identifier",
            "model_name",
            "model_version",
            "is_candidate",
            "predicted_sales",
            "raw_predicted_sales",
            "calibration_factor",
            "calibration_metadata",
            "execution_timestamp",
            "forecast_status",
            "error_message",
            "actual_sales",
            "actual_observed_at",
            "actual_source",
            "reconciliation_timestamp",
            "reconciliation_status",
            "as_of_timestamp",
            "last_observed_week_start",
        ]
        placeholders = ", ".join(["?"] * len(cols))
        # INSERT OR IGNORE respects the UNIQUE constraint on (snapshot_hash, origin_date, target_week, series_id, model_identifier)
        sql = f"INSERT OR IGNORE INTO shadow_forecasts ({', '.join(cols)}) VALUES ({placeholders})"

        rows = []
        for r in records:
            d = r.to_dict()
            rows.append([d.get(c) for c in cols])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, rows)
            inserted = cursor.rowcount
            conn.commit()

        skipped = len(records) - inserted
        logger.info(f"Persisted shadow forecasts: {inserted} inserted, {skipped} duplicate records skipped.")
        return {"inserted": inserted, "skipped_duplicates": skipped}

    def query_forecasts(
        self,
        run_id: str | None = None,
        origin_date: str | None = None,
        target_week: str | None = None,
        is_candidate: bool | None = None,
        reconciliation_status: str | None = None,
    ) -> pd.DataFrame:
        """Query stored shadow forecasts into a pandas DataFrame."""
        query = "SELECT * FROM shadow_forecasts WHERE 1=1"
        params: list[Any] = []

        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if origin_date is not None:
            query += " AND origin_date = ?"
            params.append(origin_date)
        if target_week is not None:
            query += " AND target_week = ?"
            params.append(target_week)
        if is_candidate is not None:
            query += " AND is_candidate = ?"
            params.append(1 if is_candidate else 0)
        if reconciliation_status is not None:
            query += " AND reconciliation_status = ?"
            params.append(reconciliation_status)

        query += " ORDER BY target_week, series_id, is_candidate"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def ingest_actuals(self, actuals_df: pd.DataFrame, source: str = "local_file") -> dict[str, int]:
        """Ingest observed actuals into shadow_actuals table with validation."""
        df_clean = actuals_df.copy()
        if "series_id" not in df_clean.columns and {"Client", "Warehouse", "Product"}.issubset(set(df_clean.columns)):
            from demandcast.vn1.data_loading import parse_wide_to_long

            df_clean = parse_wide_to_long(df_clean, value_name="sales", phase=1)

        required_cols = {"series_id", "week_start", "sales"}
        if not required_cols.issubset(set(df_clean.columns)):
            raise ValueError(f"Actuals DataFrame missing required columns: {required_cols - set(df_clean.columns)}")

        df_clean["week_start"] = pd.to_datetime(df_clean["week_start"])
        # Validate Monday start
        if not (df_clean["week_start"].dt.day_of_week == 0).all():
            raise ValueError("All actuals week_start dates must conform strictly to Monday week-start convention.")

        df_clean["sales"] = pd.to_numeric(df_clean["sales"], errors="coerce")
        if df_clean["sales"].isna().any() or (df_clean["sales"] < 0).any():
            raise ValueError("Actuals contain invalid, negative, or NaN sales values.")

        ingested_at = datetime.now(timezone.utc).isoformat()
        df_clean["actual_id"] = [str(uuid.uuid4()) for _ in range(len(df_clean))]
        df_clean["week_start_str"] = df_clean["week_start"].dt.strftime("%Y-%m-%d")
        df_clean["source_col"] = source
        df_clean["ingested_at_col"] = ingested_at
        df_clean["snapshot_hash_col"] = None

        cols_tuple = [
            "actual_id",
            "series_id",
            "week_start_str",
            "sales",
            "source_col",
            "ingested_at_col",
            "snapshot_hash_col",
        ]
        rows = list(df_clean[cols_tuple].itertuples(index=False, name=None))

        sql = """
            INSERT OR IGNORE INTO shadow_actuals
            (actual_id, series_id, week_start, sales, source, ingested_at, snapshot_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, rows)
            inserted = cursor.rowcount
            conn.commit()

        skipped = len(rows) - inserted
        return {"received": len(rows), "inserted": inserted, "skipped_duplicates": skipped}

    def reconcile_actuals(self, target_week: str | None = None) -> dict[str, Any]:
        """Reconcile pending forecasts with observed actuals without overwriting predictions."""
        reconciled_at = datetime.now(timezone.utc).isoformat()

        # Join pending forecasts with actuals by series_id and target_week == week_start
        select_sql = """
            SELECT
                f.record_id,
                f.series_id,
                f.target_week,
                f.actual_sales AS existing_actual,
                a.sales AS new_actual,
                a.source AS actual_source,
                a.ingested_at AS actual_observed_at
            FROM shadow_forecasts f
            INNER JOIN shadow_actuals a
                ON f.series_id = a.series_id AND f.target_week = a.week_start
            WHERE f.reconciliation_status = 'PENDING_ACTUAL'
        """
        params: list[Any] = []
        if target_week is not None:
            select_sql += " AND f.target_week = ?"
            params.append(target_week)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            matches = cursor.execute(select_sql, params).fetchall()

            matched_count = 0
            conflicts_count = 0
            conflicted_rec_ids: set[str] = set()
            update_rows: dict[str, tuple[Any, ...]] = {}

            for row in matches:
                rec_id = row["record_id"]
                existing_act = row["existing_actual"]
                new_act = row["new_actual"]
                source = row["actual_source"]
                obs_at = row["actual_observed_at"]

                # If an actual was already recorded and differs, mark conflict rather than overwriting
                if existing_act is not None and abs(existing_act - new_act) > 1e-5:
                    logger.warning(
                        f"Conflict detected for forecast {rec_id} (series {row['series_id']}, week {row['target_week']}): "
                        f"Existing={existing_act}, Incoming={new_act}. Rejecting conflicting overwrite."
                    )
                    conflicts_count += 1
                    conflicted_rec_ids.add(rec_id)
                    if rec_id in update_rows:
                        del update_rows[rec_id]
                    cursor.execute(
                        "UPDATE shadow_forecasts SET reconciliation_status = 'CONFLICT_REJECTED' WHERE record_id = ?",
                        (rec_id,),
                    )
                elif rec_id not in conflicted_rec_ids:
                    update_rows[rec_id] = (new_act, obs_at, source, reconciled_at, rec_id)

            valid_updates = list(update_rows.values())
            matched_count = len(valid_updates)

            if valid_updates:
                cursor.executemany(
                    """
                    UPDATE shadow_forecasts
                    SET actual_sales = ?,
                        actual_observed_at = ?,
                        actual_source = ?,
                        reconciliation_timestamp = ?,
                        reconciliation_status = 'MATCHED'
                    WHERE record_id = ?
                    """,
                    valid_updates,
                )

            # Update run-level awaiting counts
            cursor.execute("""
                UPDATE shadow_runs
                SET awaiting_actuals_count = (
                    SELECT COUNT(*) FROM shadow_forecasts
                    WHERE shadow_forecasts.run_id = shadow_runs.run_id
                    AND shadow_forecasts.reconciliation_status = 'PENDING_ACTUAL'
                )
            """)

            conn.commit()

            # Query overall totals
            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts WHERE reconciliation_status = 'MATCHED'")
            total_matched = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts WHERE reconciliation_status = 'PENDING_ACTUAL'")
            total_pending = cursor.fetchone()[0]

        return {
            "reconciled_in_batch": matched_count,
            "conflicts_rejected": conflicts_count,
            "total_matched_all_time": total_matched,
            "total_pending_all_time": total_pending,
            "timestamp": reconciled_at,
        }

    def get_telemetry(self) -> dict[str, Any]:
        """Aggregate operational health, execution counts, and status indicators."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM shadow_runs")
            total_runs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts")
            total_forecasts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts WHERE reconciliation_status = 'MATCHED'")
            matched_forecasts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM shadow_forecasts WHERE reconciliation_status = 'PENDING_ACTUAL'")
            pending_forecasts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM shadow_actuals")
            total_actuals = cursor.fetchone()[0]

            cursor.execute("SELECT * FROM shadow_runs ORDER BY created_at DESC LIMIT 1")
            last_run_row = cursor.fetchone()

        last_run = dict(last_run_row) if last_run_row is not None else None

        if total_runs == 0:
            overall_status = "NOT_CONFIGURED"
        elif last_run and last_run.get("status") == "FAILED":
            overall_status = "FAILED"
        elif last_run and last_run.get("status") == "PARTIAL_FAILURE":
            overall_status = "PARTIAL_FAILURE"
        else:
            overall_status = "READY"

        return {
            "status": overall_status,
            "database_path": str(self.db_path),
            "total_runs": total_runs,
            "total_forecasts": total_forecasts,
            "matched_forecasts": matched_forecasts,
            "awaiting_actuals_forecasts": pending_forecasts,
            "total_actuals_ingested": total_actuals,
            "last_run": last_run,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class VN1ShadowPipeline:
    """Orchestrates production-safe shadow inference alongside the active VN1 model."""

    def __init__(
        self,
        config: VN1Config | None = None,
        store: ShadowStore | None = None,
        registry: VN1ModelRegistry | None = None,
    ):
        self.config = config or load_vn1_config()
        self.store = store or ShadowStore(self.config.shadow.db_path)
        self.registry = registry or VN1ModelRegistry(self.config)

    def run_shadow_forecast(
        self,
        snapshot_df: pd.DataFrame | Path | str | None,
        origin_date: pd.Timestamp | str | None = None,
        horizon: int = 13,
        series_subset: int | None = None,
        provider: str = "file_batch",
        mode: Literal["historical_replay", "prospective_live"] = "historical_replay",
        as_of_timestamp: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Execute shadow inference across production model and experimental Candidate D.

        Points-in-time correctness:
        - Evaluates both models strictly on history <= origin_date under the time contract.
        - Hashes eligible snapshot for reproducibility and idempotency.
        - Completely isolates candidate failures from production forecasts.
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # 1. Check for valid input
        if snapshot_df is None:
            logger.warning("Shadow forecast called with no input snapshot; returning NOT_CONFIGURED.")
            return {
                "run_id": run_id,
                "status": "NOT_CONFIGURED",
                "message": "No input snapshot provided or configured.",
                "forecasts_count": 0,
            }

        if isinstance(snapshot_df, (str, Path)):
            p = Path(snapshot_df)
            if not p.exists():
                return {
                    "run_id": run_id,
                    "status": "NOT_CONFIGURED",
                    "message": f"Specified input snapshot path does not exist: {p}",
                    "forecasts_count": 0,
                }
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        else:
            df = snapshot_df.copy()

        if df.empty:
            return {
                "run_id": run_id,
                "status": "NOT_CONFIGURED",
                "message": "Supplied input snapshot is empty.",
                "forecasts_count": 0,
            }

        # Automatically unpivot wide raw VN1 file if needed
        if "series_id" not in df.columns and {"Client", "Warehouse", "Product"}.issubset(set(df.columns)):
            from demandcast.vn1.data_loading import parse_wide_to_long

            df = parse_wide_to_long(df, value_name="sales", phase=0)

        # Determine origin date (last_observed_week_start)
        if origin_date is None:
            if "week_start" in df.columns:
                origin_date = pd.to_datetime(df["week_start"]).max()
            else:
                origin_date = pd.to_datetime("2023-01-02")
        else:
            origin_date = pd.to_datetime(origin_date)

        # Build and validate Time Contract
        contract = create_time_contract(
            origin_date=origin_date,
            as_of_timestamp=as_of_timestamp,
            horizon=horizon,
            mode=mode,
        )

        try:
            history_df, time_audit = contract.validate_snapshot(df)
        except ValueError as e:
            if mode == "prospective_live" and (
                "fail-closed" in str(e).lower() or "incomplete" in str(e).lower() or "violation" in str(e).lower()
            ):
                logger.warning(f"Time contract validation rejected live snapshot: {e}")
                return {
                    "run_id": run_id,
                    "status": "INCOMPLETE_INPUT_REJECTED",
                    "message": str(e),
                    "forecasts_count": 0,
                    "time_contract": {
                        "mode": mode,
                        "origin_date": origin_date.strftime("%Y-%m-%d"),
                        "as_of_timestamp": contract.as_of_timestamp,
                    },
                }
            raise

        snapshot_hash = time_audit["eligible_snapshot_hash"]
        as_of_str = contract.as_of_timestamp
        orig_str = contract.last_observed_week_start.strftime("%Y-%m-%d")

        # Optional subset filtering for fast execution
        if series_subset is not None and series_subset > 0:
            top_series = (
                history_df.groupby("series_id")["sales"].sum().sort_values(ascending=False).head(series_subset).index
            )
            history_df = history_df[history_df["series_id"].isin(top_series)].copy()

        series_count = history_df["series_id"].nunique()

        logger.info(
            f"Starting VN1 shadow run {run_id} [Mode: {mode}, Provider: {provider}] "
            f"Origin (Last Observed): {orig_str}, As-Of: {as_of_str}, Series: {series_count}, Horizon: {horizon}"
        )

        # 2. Resolve Active Production Model
        prod_status = "SUCCESS"
        prod_error = None
        prod_records: list[ShadowForecastRecord] = []
        prod_version = "UNKNOWN"
        prod_model_name = "ProductionModel"

        try:
            prod_model, prod_meta = self.registry.load_current_model()
            prod_version = prod_meta.get("version", "v1.0.0")
            prod_model_name = prod_meta.get("model_type", getattr(prod_model, "name", "Baseline_MovingAvg12_Official"))

            prod_preds = prod_model.predict(history_df, horizon=horizon, origin_date=origin_date)

            for _, row in prod_preds.iterrows():
                sid = str(row["series_id"])
                parts = sid.split("_")
                cid = parts[0] if len(parts) > 0 else None
                wid = parts[1] if len(parts) > 1 else None
                pid = parts[2] if len(parts) > 2 else sid
                t_week = (
                    row["week_start"].strftime("%Y-%m-%d")
                    if hasattr(row["week_start"], "strftime")
                    else str(row["week_start"])
                )
                h_step = int(row["horizon_step"])
                p_val = float(max(0.0, row["predicted_sales"]))

                rec = ShadowForecastRecord(
                    record_id=str(uuid.uuid4()),
                    run_id=run_id,
                    dataset_provider=provider,
                    snapshot_hash=snapshot_hash,
                    series_id=sid,
                    client_id=cid,
                    warehouse_id=wid,
                    product_id=pid,
                    origin_date=orig_str,
                    target_week=t_week,
                    horizon_step=h_step,
                    model_identifier=f"production_{prod_version}",
                    model_name=prod_model_name,
                    model_version=prod_version,
                    is_candidate=False,
                    predicted_sales=p_val,
                    raw_predicted_sales=p_val,
                    calibration_factor=None,
                    calibration_metadata=None,
                    execution_timestamp=created_at,
                    forecast_status="SUCCESS",
                    as_of_timestamp=as_of_str,
                    last_observed_week_start=orig_str,
                )
                prod_records.append(rec)

        except Exception as e:
            logger.error(f"Production model inference failed in shadow runner: {e}", exc_info=True)
            prod_status = "FAILED"
            prod_error = str(e)

        # 3. Resolve and Run Candidate D (Experimental Post-Holiday Calibrated Forecaster)
        cand_status = "SUCCESS"
        cand_error = None
        cand_records: list[ShadowForecastRecord] = []
        cand_version = "v2.1-candidate_d"
        cand_model_name = "Baseline_MovingAvg12_PostHolidayTargeted"

        try:
            base_12w = VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12")
            cand_config = SeasonalCalibrationConfig(
                method="post_holiday_targeted",
                shrinkage_k=self.config.shadow.shrinkage_k,
                post_holiday_start_week=1,
                post_holiday_end_week=13,
                min_factor_bound=0.50,
                max_factor_bound=1.25,
            )
            candidate_d = CalibratedForecaster(
                base_forecaster=base_12w,
                calibration_config=cand_config,
                name=cand_model_name,
            )

            cand_preds = candidate_d.predict(history_df, horizon=horizon, origin_date=origin_date)
            cand_telemetry_json = json.dumps(candidate_d.last_applied_telemetry)

            for _, row in cand_preds.iterrows():
                sid = str(row["series_id"])
                parts = sid.split("_")
                cid = parts[0] if len(parts) > 0 else None
                wid = parts[1] if len(parts) > 1 else None
                pid = parts[2] if len(parts) > 2 else sid
                t_week = (
                    row["week_start"].strftime("%Y-%m-%d")
                    if hasattr(row["week_start"], "strftime")
                    else str(row["week_start"])
                )
                h_step = int(row["horizon_step"])
                p_val = float(max(0.0, row["predicted_sales"]))
                raw_val = float(row.get("raw_predicted_sales", p_val))
                factor_val = float(row.get("calibration_factor", 1.0))

                rec = ShadowForecastRecord(
                    record_id=str(uuid.uuid4()),
                    run_id=run_id,
                    dataset_provider=provider,
                    snapshot_hash=snapshot_hash,
                    series_id=sid,
                    client_id=cid,
                    warehouse_id=wid,
                    product_id=pid,
                    origin_date=orig_str,
                    target_week=t_week,
                    horizon_step=h_step,
                    model_identifier=f"candidate_d_{cand_version}",
                    model_name=cand_model_name,
                    model_version=cand_version,
                    is_candidate=True,
                    predicted_sales=p_val,
                    raw_predicted_sales=raw_val,
                    calibration_factor=factor_val,
                    calibration_metadata=cand_telemetry_json,
                    execution_timestamp=created_at,
                    forecast_status="SUCCESS",
                    as_of_timestamp=as_of_str,
                    last_observed_week_start=orig_str,
                )
                cand_records.append(rec)

        except Exception as e:
            logger.error(f"Candidate D inference failed in shadow runner: {e}", exc_info=True)
            cand_status = "FAILED"
            cand_error = str(e)

        # 4. Determine overall run status
        if prod_status == "SUCCESS" and cand_status == "SUCCESS":
            overall_status = "SUCCEEDED"
        elif prod_status == "SUCCESS" or cand_status == "SUCCESS":
            overall_status = "PARTIAL_FAILURE"
        else:
            overall_status = "FAILED"

        all_records = prod_records + cand_records

        # 5. Persist forecasts to SQLite Store
        save_res = self.store.save_forecasts(all_records)

        # 6. Record Run Metadata
        run_meta = {
            "run_id": run_id,
            "created_at": created_at,
            "status": overall_status,
            "dataset_provider": provider,
            "mode": mode,
            "snapshot_hash": snapshot_hash,
            "origin_date": orig_str,
            "horizon": horizon,
            "series_count": series_count,
            "production_status": prod_status,
            "candidate_status": cand_status,
            "production_version": prod_version,
            "candidate_version": cand_version,
            "forecasts_count": len(all_records),
            "awaiting_actuals_count": len(all_records),
            "error_message": prod_error or cand_error,
            "as_of_timestamp": as_of_str,
            "last_observed_week_start": orig_str,
        }
        self.store.record_run(run_meta)

        return {
            "run_id": run_id,
            "status": overall_status,
            "mode": mode,
            "dataset_provider": provider,
            "snapshot_hash": snapshot_hash,
            "origin_date": orig_str,
            "series_count": series_count,
            "horizon": horizon,
            "production": {
                "status": prod_status,
                "version": prod_version,
                "model_name": prod_model_name,
                "forecasts_count": len(prod_records),
                "error": prod_error,
            },
            "candidate_d": {
                "status": cand_status,
                "version": cand_version,
                "model_name": cand_model_name,
                "forecasts_count": len(cand_records),
                "error": cand_error,
            },
            "persistence": save_res,
            "timestamp": created_at,
            "time_contract": time_audit,
        }


class ShadowEvaluator:
    """Evaluates reconciled shadow forecasts using official VN1 competition metrics."""

    def __init__(self, store: ShadowStore):
        self.store = store

    def evaluate_matched_forecasts(
        self,
        run_id: str | None = None,
        origin_date: str | None = None,
    ) -> dict[str, Any]:
        """Compute comparison metrics strictly for matched forecasts with valid observed actuals."""
        # Query matched forecasts
        df = self.store.query_forecasts(
            run_id=run_id,
            origin_date=origin_date,
            reconciliation_status="MATCHED",
        )

        if df.empty or df["actual_sales"].isna().all():
            return {
                "status": "NO_MATCHED_ACTUALS",
                "message": (
                    "No matched actuals currently available for evaluation. "
                    "Forecast quality cannot be scored until ground-truth demand is observed and reconciled."
                ),
                "sample_count": 0,
            }

        # Ensure numeric
        df["actual_sales"] = pd.to_numeric(df["actual_sales"], errors="coerce")
        df["predicted_sales"] = pd.to_numeric(df["predicted_sales"], errors="coerce")
        df = df.dropna(subset=["actual_sales", "predicted_sales"]).copy()

        # Group by model
        prod_df = df[df["is_candidate"] == 0].copy()
        cand_df = df[df["is_candidate"] == 1].copy()

        # Inner join to ensure strict pairing across same series and target week
        paired = pd.merge(
            prod_df[
                ["series_id", "origin_date", "target_week", "horizon_step", "actual_sales", "predicted_sales"]
            ].rename(columns={"predicted_sales": "prod_predicted_sales"}),
            cand_df[["series_id", "origin_date", "target_week", "predicted_sales"]].rename(
                columns={"predicted_sales": "cand_predicted_sales"}
            ),
            on=["series_id", "origin_date", "target_week"],
            how="inner",
        )

        if paired.empty:
            return {
                "status": "INSUFFICIENT_PAIRED_DATA",
                "message": "Matched records exist but could not be strictly paired between production and candidate.",
                "sample_count": 0,
            }

        n_samples = len(paired)
        y_true = paired["actual_sales"]
        y_prod = paired["prod_predicted_sales"]
        y_cand = paired["cand_predicted_sales"]

        prod_metrics = compute_all_vn1_metrics(y_true, y_prod)
        cand_metrics = compute_all_vn1_metrics(y_true, y_cand)

        delta_vn1 = round(float(cand_metrics["vn1_score"] - prod_metrics["vn1_score"]), 4)
        delta_wape = round(float(cand_metrics["wape"] - prod_metrics["wape"]), 4)
        delta_bias = round(float(cand_metrics["relative_bias"] - prod_metrics["relative_bias"]), 4)

        # Multi-horizon breakdown
        horizon_records = []
        for h_step, grp in paired.groupby("horizon_step"):
            p_m = compute_all_vn1_metrics(grp["actual_sales"], grp["prod_predicted_sales"])
            c_m = compute_all_vn1_metrics(grp["actual_sales"], grp["cand_predicted_sales"])
            horizon_records.append(
                {
                    "horizon_step": int(h_step),
                    "sample_count": len(grp),
                    "production_vn1_score": p_m["vn1_score"],
                    "candidate_vn1_score": c_m["vn1_score"],
                    "delta_vn1_score": round(float(c_m["vn1_score"] - p_m["vn1_score"]), 4),
                    "production_wape": p_m["wape"],
                    "candidate_wape": c_m["wape"],
                    "production_bias": p_m["relative_bias"],
                    "candidate_bias": c_m["relative_bias"],
                }
            )

        # Post-holiday vs. Ordinary Calendar segmentation
        paired["target_date"] = pd.to_datetime(paired["target_week"])
        paired["is_post_holiday"] = paired["target_date"].dt.isocalendar().week.between(1, 13)

        post_hol = paired[paired["is_post_holiday"]]
        ordinary = paired[~paired["is_post_holiday"]]

        post_hol_summary = None
        if not post_hol.empty:
            p_ph = compute_all_vn1_metrics(post_hol["actual_sales"], post_hol["prod_predicted_sales"])
            c_ph = compute_all_vn1_metrics(post_hol["actual_sales"], post_hol["cand_predicted_sales"])
            post_hol_summary = {
                "sample_count": len(post_hol),
                "production_vn1_score": p_ph["vn1_score"],
                "candidate_vn1_score": c_ph["vn1_score"],
                "delta_vn1_score": round(float(c_ph["vn1_score"] - p_ph["vn1_score"]), 4),
                "production_bias": p_ph["relative_bias"],
                "candidate_bias": c_ph["relative_bias"],
            }

        ordinary_summary = None
        if not ordinary.empty:
            p_ord = compute_all_vn1_metrics(ordinary["actual_sales"], ordinary["prod_predicted_sales"])
            c_ord = compute_all_vn1_metrics(ordinary["actual_sales"], ordinary["cand_predicted_sales"])
            ordinary_summary = {
                "sample_count": len(ordinary),
                "production_vn1_score": p_ord["vn1_score"],
                "candidate_vn1_score": c_ord["vn1_score"],
                "delta_vn1_score": round(float(c_ord["vn1_score"] - p_ord["vn1_score"]), 4),
                "production_bias": p_ord["relative_bias"],
                "candidate_bias": c_ord["relative_bias"],
            }

        return {
            "status": "EVALUATED",
            "sample_count": n_samples,
            "series_count": paired["series_id"].nunique(),
            "target_weeks_count": paired["target_week"].nunique(),
            "production_model": {
                "model_identifier": prod_df["model_identifier"].iloc[0] if not prod_df.empty else "production",
                "metrics": prod_metrics,
            },
            "candidate_d": {
                "model_identifier": cand_df["model_identifier"].iloc[0] if not cand_df.empty else "candidate_d",
                "metrics": cand_metrics,
            },
            "comparison_delta": {
                "delta_vn1_score": delta_vn1,  # negative means candidate improved
                "delta_wape": delta_wape,
                "delta_relative_bias": delta_bias,
                "interpretation": (
                    "Candidate D improved over Production" if delta_vn1 < 0 else "Production outperformed Candidate D"
                ),
            },
            "by_horizon": horizon_records,
            "post_holiday_segment": post_hol_summary,
            "ordinary_segment": ordinary_summary,
            "governance_note": (
                "Shadow metrics represent prospective observability only. "
                "Candidate D is an experimental candidate and is NOT automatically promoted."
            ),
        }
