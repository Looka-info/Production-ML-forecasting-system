"""CLI Runner for DemandCast v2.2 Production-Safe Shadow Forecasting Pipeline.

Commands:
  forecast   - Generate shadow forecasts for Production and Candidate D on an input snapshot.
  reconcile  - Ingest observed actuals and reconcile them against pending shadow forecasts.
  report     - Generate evaluation and comparison report on matched forecasts.
  telemetry  - Display current operational status and shadow pipeline telemetry.

Usage examples:
  python scripts/run_vn1_shadow.py forecast --series-subset 250 --mode historical_replay
  python scripts/run_vn1_shadow.py reconcile --actuals-path data/vn1/raw/Phase_1_Sales.csv
  python scripts/run_vn1_shadow.py report
  python scripts/run_vn1_shadow.py telemetry
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from demandcast.config import get_project_root
from demandcast.vn1.config import load_vn1_config
from demandcast.vn1.shadow import (
    ShadowEvaluator,
    ShadowStore,
    VN1ShadowPipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_vn1_shadow")


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run shadow forecasting on a validated snapshot."""
    root = get_project_root()
    config = load_vn1_config()
    db_path = args.db_path or config.shadow.db_path
    store = ShadowStore(db_path)
    pipeline = VN1ShadowPipeline(config=config, store=store)

    snapshot_path = None
    if args.snapshot_path:
        snapshot_path = Path(args.snapshot_path)
    else:
        # Check standard paths in order of preference
        candidate_paths = [
            root / config.data.processed_sales_path,
            root / config.data.phase0_sales_path,
        ]
        for p in candidate_paths:
            if p.exists():
                snapshot_path = p
                break

    if snapshot_path is None or not snapshot_path.exists():
        print("\n" + "=" * 80)
        print("DEMANDCAST v2.2 SHADOW PIPELINE: STATUS = NOT_CONFIGURED")
        print("=" * 80)
        print("No valid input snapshot file was provided or found at default locations.")
        print("To run in file batch mode, supply: --snapshot-path <path_to_sales_file.csv>")
        print("=" * 80 + "\n")
        return 0

    logger.info(f"Using input snapshot: {snapshot_path}")

    # Load snapshot
    if snapshot_path.suffix == ".parquet":
        df = pd.read_parquet(snapshot_path)
    else:
        df = pd.read_csv(snapshot_path)

    res = pipeline.run_shadow_forecast(
        snapshot_df=df,
        origin_date=args.origin_date,
        horizon=args.horizon,
        series_subset=args.series_subset,
        provider=args.provider,
        mode=args.mode,
        candidates=args.candidates,
    )

    print("\n" + "=" * 85)
    print("DEMANDCAST v2.2 SHADOW FORECAST EXECUTION SUMMARY")
    print("=" * 85)
    print(f"Run ID:            {res['run_id']}")
    print(f"Overall Status:    {res['status']}")
    print(f"Execution Mode:    {res['mode']} (Prospective Live: {res['mode'] == 'prospective_live'})")
    print(f"Dataset Provider:  {res['dataset_provider']}")
    print(f"Snapshot Hash:     {res['snapshot_hash'][:16]}...")
    print(f"Forecast Origin:   {res['origin_date']}")
    print(f"Series Evaluated:  {res['series_count']}")
    print(f"Forecast Horizon:  {res['horizon']} weeks")
    print("-" * 85)
    print(f"Production Model:  {res['production']['model_name']} ({res['production']['version']})")
    print(f"  Status:          {res['production']['status']}")
    print(f"  Forecasts:       {res['production']['forecasts_count']}")
    if "candidate_c" in res and res["candidate_c"]["status"] == "SUCCESS":
        print(f"Candidate C Model: {res['candidate_c']['model_name']} ({res['candidate_c']['version']})")
        print(f"  Status:          {res['candidate_c']['status']}")
        print(f"  Forecasts:       {res['candidate_c']['forecasts_count']}")
    if "candidate_d" in res and res["candidate_d"]["status"] == "SUCCESS":
        print(f"Candidate D Model: {res['candidate_d']['model_name']} ({res['candidate_d']['version']})")
        print(f"  Status:          {res['candidate_d']['status']}")
        print(f"  Forecasts:       {res['candidate_d']['forecasts_count']}")
    print("-" * 85)
    print(
        f"Persistence:       {res['persistence']['inserted']} inserted, {res['persistence']['skipped_duplicates']} duplicate records skipped"
    )
    print("=" * 85 + "\n")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    """Ingest delayed actuals and reconcile against pending forecasts."""
    config = load_vn1_config()
    db_path = args.db_path or config.shadow.db_path
    store = ShadowStore(db_path)

    actuals_path = Path(args.actuals_path)
    if not actuals_path.exists():
        logger.error(f"Specified actuals file does not exist: {actuals_path}")
        return 1

    logger.info(f"Reading ground-truth actuals from: {actuals_path}")
    if actuals_path.suffix == ".parquet":
        actuals_df = pd.read_parquet(actuals_path)
    else:
        actuals_df = pd.read_csv(actuals_path)

    # If raw wide format, optimize unpivot to target weeks awaiting reconciliation
    if {"Client", "Warehouse", "Product"}.issubset(set(actuals_df.columns)):
        with store._get_connection() as conn:
            cur = conn.cursor()
            pending_weeks = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT target_week FROM shadow_forecasts WHERE reconciliation_status = 'PENDING_ACTUAL'"
                ).fetchall()
            ]
            pending_series = set(
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT series_id FROM shadow_forecasts WHERE reconciliation_status = 'PENDING_ACTUAL'"
                ).fetchall()
            )

        date_cols = [c for c in actuals_df.columns if c not in ["Client", "Warehouse", "Product"]]
        if pending_weeks:
            date_cols = [c for c in date_cols if c.strip() in pending_weeks]

        logger.info(f"Unpivoting wide actuals for {len(date_cols)} relevant forecast weeks...")
        actuals_df = pd.melt(
            actuals_df,
            id_vars=["Client", "Warehouse", "Product"],
            value_vars=date_cols,
            var_name="week_start_str",
            value_name="sales",
        )
        actuals_df["week_start"] = pd.to_datetime(actuals_df["week_start_str"].str.strip())
        actuals_df["series_id"] = (
            actuals_df["Client"].astype(str)
            + "_"
            + actuals_df["Warehouse"].astype(str)
            + "_"
            + actuals_df["Product"].astype(str)
        )
        if pending_series:
            actuals_df = actuals_df[actuals_df["series_id"].isin(pending_series)].copy()

    # Ingest actuals
    ingest_res = store.ingest_actuals(actuals_df, source=args.source)
    logger.info(
        f"Ingested actuals: {ingest_res['inserted']} inserted, {ingest_res['skipped_duplicates']} duplicates skipped."
    )

    # Reconcile against pending forecasts
    recon_res = store.reconcile_actuals(target_week=args.target_week)

    print("\n" + "=" * 80)
    print("DEMANDCAST v2.2 DELAYED ACTUALS RECONCILIATION SUMMARY")
    print("=" * 80)
    print(f"Actuals Source:          {args.source}")
    print(f"Actuals Records Ingested: {ingest_res['inserted']} (received: {ingest_res['received']})")
    print(f"Forecasts Reconciled:    {recon_res['reconciled_in_batch']}")
    print(f"Conflicting Overwrites:  {recon_res['conflicts_rejected']} rejected")
    print(f"Total Matched (All-Time):{recon_res['total_matched_all_time']}")
    print(f"Total Pending Actuals:   {recon_res['total_pending_all_time']}")
    print(f"Reconciliation Time:     {recon_res['timestamp']}")
    print("=" * 80 + "\n")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Evaluate matched forecasts and export summary comparison reports."""
    root = get_project_root()
    config = load_vn1_config()
    db_path = args.db_path or config.shadow.db_path
    output_dir = Path(args.output_dir or root / config.shadow.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = ShadowStore(db_path)
    evaluator = ShadowEvaluator(store)

    report = evaluator.evaluate_matched_forecasts(run_id=args.run_id)

    if report["status"] != "EVALUATED":
        print("\n" + "=" * 80)
        print(f"DEMANDCAST v2.2 SHADOW EVALUATION: STATUS = {report['status']}")
        print("=" * 80)
        print(report.get("message", "No evaluation metrics could be calculated."))
        print("=" * 80 + "\n")
        return 0

    # Save summary JSON
    summary_path = output_dir / "shadow_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save horizon breakdown CSV if available
    if report.get("by_horizon"):
        h_df = pd.DataFrame(report["by_horizon"])
        h_path = output_dir / "shadow_by_horizon.csv"
        h_df.to_csv(h_path, index=False)

    print("\n" + "=" * 85)
    print("DEMANDCAST v2.2 SHADOW EVALUATION REPORT (OFFICIAL VN1 METRICS)")
    print("=" * 85)
    print(f"Sample Size:            {report['sample_count']} matched observation pairs")
    print(f"Distinct Series:        {report['series_count']}")
    print(f"Distinct Target Weeks:  {report['target_weeks_count']}")
    print("-" * 85)
    p_m = report["production_model"]["metrics"]
    c_m = report["candidate_d"]["metrics"]
    d_m = report["comparison_delta"]
    print(f"{'Metric':<20} | {'Production (v1.0.0)':<22} | {'Candidate D (v2.1)':<22} | {'Delta (Cand - Prod)':<20}")
    print("-" * 85)
    print(
        f"{'VN1 Score (min)':<20} | {p_m['vn1_score']:<22.4f} | {c_m['vn1_score']:<22.4f} | {d_m['delta_vn1_score']:<+20.4f}"
    )
    print(f"{'WAPE (min)':<20} | {p_m['wape']:<22.4f} | {c_m['wape']:<22.4f} | {d_m['delta_wape']:<+20.4f}")
    print(f"{'MAE (min)':<20} | {p_m['mae']:<22.4f} | {c_m['mae']:<22.4f} | {c_m['mae'] - p_m['mae']:<+20.4f}")
    print(f"{'RMSE (min)':<20} | {p_m['rmse']:<22.4f} | {c_m['rmse']:<22.4f} | {c_m['rmse'] - p_m['rmse']:<+20.4f}")
    print(
        f"{'Relative Bias':<20} | {p_m['relative_bias']:<22.4f} | {c_m['relative_bias']:<22.4f} | {d_m['delta_relative_bias']:<+20.4f}"
    )
    print("-" * 85)
    print(f"Delta Interpretation:   {d_m['interpretation']}")
    if report.get("post_holiday_segment"):
        ph = report["post_holiday_segment"]
        print(
            f"Post-Holiday VN1 Score: Prod = {ph['production_vn1_score']:.4f}, Cand = {ph['candidate_vn1_score']:.4f} (Delta: {ph['delta_vn1_score']:+.4f}, n={ph['sample_count']})"
        )
    if report.get("ordinary_segment"):
        ord_s = report["ordinary_segment"]
        print(
            f"Ordinary VN1 Score:     Prod = {ord_s['production_vn1_score']:.4f}, Cand = {ord_s['candidate_vn1_score']:.4f} (Delta: {ord_s['delta_vn1_score']:+.4f}, n={ord_s['sample_count']})"
        )
    print("=" * 85)
    print(f"Saved evaluation artifacts to: {output_dir}")
    print("=" * 85 + "\n")
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    """Display store telemetry and health status."""
    config = load_vn1_config()
    db_path = args.db_path or config.shadow.db_path
    store = ShadowStore(db_path)
    telem = store.get_telemetry()

    if args.json:
        print(json.dumps(telem, indent=2))
        return 0

    print("\n" + "=" * 80)
    print("DEMANDCAST v2.2 SHADOW PIPELINE TELEMETRY")
    print("=" * 80)
    print(f"Pipeline Status:          {telem['status']}")
    print(f"Database Location:        {telem['database_path']}")
    print(f"Total Shadow Runs:        {telem['total_runs']}")
    print(f"Total Forecasts Stored:   {telem['total_forecasts']}")
    print(f"  Matched with Actuals:   {telem['matched_forecasts']}")
    print(f"  Awaiting Actuals:       {telem['awaiting_actuals_forecasts']}")
    print(f"Total Actuals Ingested:   {telem['total_actuals_ingested']}")
    if telem.get("last_run"):
        lr = telem["last_run"]
        print("-" * 80)
        print("Last Run Details:")
        print(f"  Run ID:                 {lr.get('run_id')}")
        print(f"  Created At:             {lr.get('created_at')}")
        print(f"  Status:                 {lr.get('status')}")
        print(f"  Provider:               {lr.get('dataset_provider')}")
        print(f"  Origin Date:            {lr.get('origin_date')}")
        print(f"  Series Count:           {lr.get('series_count')}")
        print(f"  Production Status:      {lr.get('production_status')}")
        print(f"  Candidate Status:       {lr.get('candidate_status')}")
    print("=" * 80 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="DemandCast v2.2 Production-Safe Shadow Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Shadow subcommands")

    # 1. Forecast
    p_fc = subparsers.add_parser("forecast", help="Run shadow forecast on snapshot")
    p_fc.add_argument("--snapshot-path", type=str, default=None, help="Path to input snapshot CSV/parquet")
    p_fc.add_argument("--origin-date", type=str, default=None, help="Forecast origin YYYY-MM-DD (Monday)")
    p_fc.add_argument("--horizon", type=int, default=13, help="Forecast horizon (default: 13)")
    p_fc.add_argument("--series-subset", type=int, default=None, help="Limit series count for fast execution")
    p_fc.add_argument("--provider", type=str, default="file_batch", help="Dataset provider identifier")
    p_fc.add_argument(
        "--mode",
        choices=["historical_replay", "prospective_live"],
        default="historical_replay",
        help="Inference mode (clearly distinguishes retrospective replay from live)",
    )
    p_fc.add_argument(
        "--candidates",
        nargs="+",
        default=["candidate_d"],
        choices=["candidate_c", "candidate_d"],
        help="Candidate models to execute in shadow mode alongside production",
    )
    p_fc.add_argument("--db-path", type=str, default=None, help="Custom SQLite db path")

    # 2. Reconcile
    p_rec = subparsers.add_parser("reconcile", help="Ingest actuals and reconcile against pending forecasts")
    p_rec.add_argument("--actuals-path", type=str, required=True, help="Path to actual sales CSV/parquet")
    p_rec.add_argument("--source", type=str, default="local_actuals", help="Source identifier of incoming actuals")
    p_rec.add_argument("--target-week", type=str, default=None, help="Filter reconciliation to specific target week")
    p_rec.add_argument("--db-path", type=str, default=None, help="Custom SQLite db path")

    # 3. Report
    p_rep = subparsers.add_parser("report", help="Evaluate matched forecasts and export comparison reports")
    p_rep.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    p_rep.add_argument("--run-id", type=str, default=None, help="Filter evaluation to specific shadow run ID")
    p_rep.add_argument("--db-path", type=str, default=None, help="Custom SQLite db path")

    # 4. Telemetry
    p_tel = subparsers.add_parser("telemetry", help="Display shadow pipeline health and store telemetry")
    p_tel.add_argument("--json", action="store_true", help="Output telemetry as JSON")
    p_tel.add_argument("--db-path", type=str, default=None, help="Custom SQLite db path")

    args = parser.parse_args()

    if args.command == "forecast":
        sys.exit(cmd_forecast(args))
    elif args.command == "reconcile":
        sys.exit(cmd_reconcile(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))
    elif args.command == "telemetry":
        sys.exit(cmd_telemetry(args))


if __name__ == "__main__":
    main()
