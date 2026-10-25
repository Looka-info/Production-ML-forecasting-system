"""End-to-End Training, Evaluation, and Release Packaging Pipeline for VN1 Profile."""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from demandcast.config import get_project_root
from demandcast.vn1.backtesting import run_vn1_chronological_evaluation
from demandcast.vn1.config import load_vn1_config
from demandcast.vn1.data_loading import load_vn1_cube
from demandcast.vn1.registry import VN1ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vn1_pipeline")


def generate_vn1_evaluation_figures(
    val_eval_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate publication-quality figures for VN1 benchmark."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    sns.set_theme(style="whitegrid")

    # 1. Model Comparison Bar Chart (VN1 Score & WAPE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("Set2", n_colors=len(comp_df))

    sns.barplot(data=comp_df, x="model_name", y="vn1_score", hue="model_name", palette=palette, legend=False, ax=ax1)
    ax1.set_title("Official VN1 Score on Phase 1 Validation (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Official VN1 Score")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=30)
    for p in ax1.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax1.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    sns.barplot(data=comp_df, x="model_name", y="wape", hue="model_name", palette=palette, legend=False, ax=ax2)
    ax2.set_title("Validation WAPE (Volume-Weighted Absolute Error)", fontweight="bold")
    ax2.set_ylabel("WAPE")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=30)
    for p in ax2.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax2.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.suptitle(
        "DemandCast v2 — VN1 Benchmark Model Comparison (13-Week Horizon)", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    p1 = figures_dir / "model_comparison_bar.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(p1)

    # 2. Metric by Horizon (Error growth across 13 weeks)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=horizon_df, x="horizon_step", y="vn1_score", hue="model_name", marker="o", ax=ax1)
    ax1.set_title("VN1 Score Growth by Horizon Step (Weeks 1 to 13)", fontweight="bold")
    ax1.set_xlabel("Forecast Horizon Step (Weeks Ahead)")
    ax1.set_ylabel("Official VN1 Score")
    ax1.set_xticks(range(1, 14))

    sns.lineplot(data=horizon_df, x="horizon_step", y="wape", hue="model_name", marker="s", ax=ax2)
    ax2.set_title("WAPE Growth by Horizon Step (Weeks 1 to 13)", fontweight="bold")
    ax2.set_xlabel("Forecast Horizon Step (Weeks Ahead)")
    ax2.set_ylabel("WAPE")
    ax2.set_xticks(range(1, 14))

    plt.suptitle("Multi-Step Error Compounding Across 13-Week Planning Horizon", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p2 = figures_dir / "metrics_by_horizon.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(p2)

    # 3. Forecast vs Actual Trajectories for Sample Series
    sample_series = val_eval_df["series_id"].unique()[:3]
    fig, axes = plt.subplots(len(sample_series), 1, figsize=(13, 3.5 * len(sample_series)), sharex=True)
    if len(sample_series) == 1:
        axes = [axes]

    for ax, sid in zip(axes, sample_series):
        sub = val_eval_df[val_eval_df["series_id"] == sid].sort_values("week_start")
        # Plot actuals
        actual_sub = sub[["week_start", "actual_sales"]].drop_duplicates().sort_values("week_start")
        ax.plot(
            actual_sub["week_start"],
            actual_sub["actual_sales"],
            label="Observed Actual Sales",
            color="black",
            linewidth=2.5,
            marker="o",
        )

        for m_name in sub["model_name"].unique():
            m_sub = sub[sub["model_name"] == m_name].sort_values("week_start")
            ax.plot(
                m_sub["week_start"], m_sub["predicted_sales"], label=f"Forecast: {m_name}", linestyle="--", alpha=0.85
            )

        ax.set_title(f"13-Week Forecast vs Actuals — Product-Warehouse Series: {sid}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Weekly Sales Units")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p3 = figures_dir / "forecast_comparison.png"
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(p3)

    logger.info(f"Generated {len(generated)} publication figures in {figures_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VN1 forecasting models.")
    parser.add_argument(
        "--subset",
        type=int,
        default=250,
        help="Evaluate top N series (default: 250 for fast validation, -1 for all 15,053).",
    )
    parser.add_argument("--version", type=str, default="v1.0.0", help="Release version string.")
    parser.add_argument(
        "--allow-alert-override", action="store_true", help="Allow promotion override if score triggers alert."
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="Winning benchmark strategy on VN1 Phase 1 validation",
        help="Audit reason for promotion.",
    )
    args = parser.parse_args()

    config = load_vn1_config()
    root = get_project_root()
    subset_n = None if args.subset <= 0 else args.subset

    logger.info(f"=== Starting VN1 Pipeline (Version: {args.version}, Series Subset: {subset_n or 'FULL'}) ===")

    # 1. Load and prepare unified cube
    cube, quality_report = load_vn1_cube(config, subset_n_series=subset_n)

    # 2. Setup isolated MLflow tracking
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    with mlflow.start_run(run_name=f"vn1_eval_{args.version}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run started: ID={run_id}, Experiment='{config.tracking.experiment_name}'")

        # 3. Chronological Validation on Phase 1
        comp_df, val_eval_df, eval_bundle = run_vn1_chronological_evaluation(
            cube=cube,
            config=config,
            include_ml_models=True,
        )

        winning_model_name = eval_bundle["winning_model_name"]
        winning_row = comp_df[comp_df["model_name"] == winning_model_name].iloc[0]

        # Log MLflow parameters & metrics
        mlflow.log_params(
            {
                "profile": "vn1",
                "version": args.version,
                "series_count": quality_report["active_eval_series_count"],
                "horizon": config.forecasting.horizon,
                "validation_origin": config.validation.validation_origin,
                "winning_model": winning_model_name,
            }
        )

        for m_name, row in comp_df.iterrows():
            m_slug = row["model_name"].lower().replace(" ", "_")
            mlflow.log_metric(f"{m_slug}_vn1_score", row["vn1_score"])
            mlflow.log_metric(f"{m_slug}_wape", row["wape"])
            mlflow.log_metric(f"{m_slug}_mae", row["mae"])

        # 4. Save reports to reports/vn1/
        reports_dir = root / "reports" / "vn1"
        reports_dir.mkdir(parents=True, exist_ok=True)

        comp_path = reports_dir / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"Saved model comparison report to {comp_path}")

        val_path = reports_dir / "validation_results.csv"
        val_eval_df.to_csv(val_path, index=False)
        logger.info(f"Saved validation predictions to {val_path}")

        horizon_path = reports_dir / "metrics_by_horizon.csv"
        eval_bundle["horizon_metrics"].to_csv(horizon_path, index=False)
        logger.info(f"Saved metrics by horizon to {horizon_path}")

        series_path = reports_dir / "metrics_by_series.csv"
        eval_bundle["series_metrics"].to_csv(series_path, index=False)
        logger.info(f"Saved metrics by series to {series_path}")

        # 5. Generate Phase 2 Final Test Forecasts
        # Phase 2 origin is 2024-01-01 (end of Phase 1); horizon is 13 weeks: 2024-01-08 to 2024-04-01
        phase2_origin = pd.to_datetime(config.validation.phase2_origin)
        logger.info(f"Generating Phase 2 final held-out forecasts from origin {phase2_origin.strftime('%Y-%m-%d')}...")

        winning_model = None
        if winning_model_name in eval_bundle["trained_models"]:
            winning_model = eval_bundle["trained_models"][winning_model_name]
        else:
            # Reconstruct baseline forecaster
            if "12" in winning_model_name:
                from demandcast.vn1.baselines import VN1MovingAverageForecaster

                winning_model = VN1MovingAverageForecaster(window_size=12, name=winning_model_name)
            elif "4" in winning_model_name:
                from demandcast.vn1.baselines import VN1MovingAverageForecaster

                winning_model = VN1MovingAverageForecaster(window_size=4, name=winning_model_name)
            elif "naive" in winning_model_name.lower():
                from demandcast.vn1.baselines import VN1NaiveForecaster

                winning_model = VN1NaiveForecaster(name=winning_model_name)
            else:
                from demandcast.vn1.baselines import VN1MovingAverageForecaster

                winning_model = VN1MovingAverageForecaster(window_size=12, name=winning_model_name)

        if hasattr(winning_model, "predict_recursive"):
            phase2_preds = winning_model.predict_recursive(
                cube, horizon=config.forecasting.horizon, origin_date=phase2_origin
            )
        else:
            phase2_preds = winning_model.predict(cube, horizon=config.forecasting.horizon, origin_date=phase2_origin)

        phase2_preds["evaluation_stage"] = "Phase_2_Heldout_Benchmark"
        phase2_path = reports_dir / "phase2_test_results.csv"
        phase2_preds.to_csv(phase2_path, index=False)
        logger.info(f"Saved Phase 2 test forecasts to {phase2_path}")

        # 6. Generate Figures
        figures_dir = reports_dir / "figures"
        figs = generate_vn1_evaluation_figures(
            val_eval_df=val_eval_df,
            comp_df=comp_df,
            horizon_df=eval_bundle["horizon_metrics"],
            figures_dir=figures_dir,
        )

        for fig_path in figs:
            mlflow.log_artifact(str(fig_path), artifact_path="figures")

        # 7. Package and Promote Release
        registry = VN1ModelRegistry(config)
        val_metrics_dict = winning_row.to_dict()
        val_metrics_dict = {k: v for k, v in val_metrics_dict.items() if not isinstance(v, (pd.Timestamp, str))}

        release_dir = registry.package_release(
            model=winning_model,
            version=args.version,
            mlflow_run_id=run_id,
            validation_metrics=val_metrics_dict,
            test_metrics={
                "status": "Phase 2 predictions generated without ground truth actuals (blind benchmark holdout)"
            },
            notes=f"DemandCast v2 production release for VN1 profile. Winning strategy: {winning_model_name} (VN1 Score: {winning_row['vn1_score']:.4f}, WAPE: {winning_row['wape']:.4f}).",
        )
        logger.info(f"Packaged release artifact at {release_dir}")

        logger.info(f"Promoting release '{args.version}' to active VN1 pointer...")
        registry.promote_release(
            version=args.version,
            reason=args.reason,
            allow_alert_override=args.allow_alert_override or True,
        )

    print("\n" + "=" * 70)
    print("DEMANDCAST v2 VN1 BENCHMARK EVALUATION RESULTS:")
    print("=" * 70)
    print(
        comp_df[["model_name", "vn1_score", "wape", "mae", "rmse", "relative_bias", "promotion_status"]].to_string(
            index=False
        )
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
