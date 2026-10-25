"""Final VN1 Phase 2 Ground-Truth Evaluation Script.

Evaluates frozen and candidate models on officially published Phase 2 actual sales:
1. 250-series development subset across all 6 candidates (Baselines + LightGBM models).
2. Full 15,053-series competition benchmark across official baseline methods.
Generates comprehensive reports, JSON summaries, and publication figures.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from demandcast.config import get_project_root
from demandcast.vn1.baselines import (
    VN1MovingAverageForecaster,
    VN1NaiveForecaster,
    VN1SeasonalNaiveForecaster,
)
from demandcast.vn1.config import load_vn1_config
from demandcast.vn1.data_loading import load_vn1_cube
from demandcast.vn1.metrics import (
    compute_all_vn1_metrics,
)
from demandcast.vn1.models import VN1Forecaster
from demandcast.vn1.registry import VN1ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_vn1_phase2")

INDEX_COLS = ["Client", "Warehouse", "Product"]
HORIZON_WEEKS = 13


def read_weekly_matrix(path: Path) -> pd.DataFrame:
    """Read a VN1 wide matrix file: Client, Warehouse, Product as index, dates as columns."""
    df = pd.read_csv(path)
    df = df.set_index(INDEX_COLS)
    df.columns = pd.to_datetime(df.columns)
    return df


def official_score(forecast_arr: np.ndarray, actuals_arr: np.ndarray) -> tuple[float, float, float]:
    """Official VN1 competition scoring formula: Score = MAE% + |Bias%|."""
    err = forecast_arr - actuals_arr
    total_sales = np.nansum(actuals_arr)
    if total_sales <= 0.0:
        return 0.0, 0.0, 0.0
    mae_pct = np.nansum(np.abs(err)) / total_sales
    bias_pct = np.nansum(err) / total_sales
    return float(mae_pct + abs(bias_pct)), float(mae_pct), float(bias_pct)


def run_full_benchmark_evaluation(raw_dir: Path) -> list[dict]:
    """Execute official benchmarks across the full 15,053-series catalog."""
    logger.info("--- Evaluating Full 15,053-Series Official Competition Benchmarks ---")
    p0_sales = read_weekly_matrix(raw_dir / "Phase 0 - Sales.csv")
    p1_sales = read_weekly_matrix(raw_dir / "Phase 1 - Sales.csv")
    p2_actuals = read_weekly_matrix(raw_dir / "Phase 2 - Sales.csv")

    # Combined history up to Phase 2 origin (170 + 13 = 183 weeks)
    history_sales = pd.concat([p0_sales, p1_sales], axis=1)
    weeks = p2_actuals.columns

    results = []

    # 1. Naive Benchmark (Last Known Value at 2024-01-01)
    naive_vals = np.repeat(history_sales.values[:, -1:], HORIZON_WEEKS, axis=1)
    n_score, n_mae_pct, n_bias_pct = official_score(naive_vals, p2_actuals.values)
    results.append(
        {
            "evaluation_scope": "Full_Competition_Benchmark (15,053 series)",
            "model_name": "Baseline_Naive",
            "model_family": "Last Known Value",
            "series_count": len(p2_actuals),
            "total_instances": len(p2_actuals) * HORIZON_WEEKS,
            "vn1_score": round(n_score, 6),
            "wape": round(n_mae_pct, 6),
            "mae": round(float(np.nanmean(np.abs(naive_vals - p2_actuals.values))), 4),
            "rmse": round(float(np.sqrt(np.nanmean((naive_vals - p2_actuals.values) ** 2))), 4),
            "relative_bias": round(n_bias_pct, 6),
            "official_reference": "0.506970 (SupChains published benchmark)",
        }
    )

    # 2. Official 12-week Moving Average Benchmark
    ma12_vals = np.repeat(np.nanmean(history_sales.values[:, -12:], axis=1, keepdims=True), HORIZON_WEEKS, axis=1)
    m12_score, m12_mae_pct, m12_bias_pct = official_score(ma12_vals, p2_actuals.values)
    results.append(
        {
            "evaluation_scope": "Full_Competition_Benchmark (15,053 series)",
            "model_name": "Baseline_MovingAvg12_Official",
            "model_family": "Trailing Mean (12w)",
            "series_count": len(p2_actuals),
            "total_instances": len(p2_actuals) * HORIZON_WEEKS,
            "vn1_score": round(m12_score, 6),
            "wape": round(m12_mae_pct, 6),
            "mae": round(float(np.nanmean(np.abs(ma12_vals - p2_actuals.values))), 4),
            "rmse": round(float(np.sqrt(np.nanmean((ma12_vals - p2_actuals.values) ** 2))), 4),
            "relative_bias": round(m12_bias_pct, 6),
            "official_reference": "0.804585 (SupChains published benchmark)",
        }
    )

    # 3. DemandCast 4-week Moving Average Baseline
    ma4_vals = np.repeat(np.nanmean(history_sales.values[:, -4:], axis=1, keepdims=True), HORIZON_WEEKS, axis=1)
    m4_score, m4_mae_pct, m4_bias_pct = official_score(ma4_vals, p2_actuals.values)
    results.append(
        {
            "evaluation_scope": "Full_Competition_Benchmark (15,053 series)",
            "model_name": "Baseline_MovingAvg4_DemandCast",
            "model_family": "Trailing Mean (4w)",
            "series_count": len(p2_actuals),
            "total_instances": len(p2_actuals) * HORIZON_WEEKS,
            "vn1_score": round(m4_score, 6),
            "wape": round(m4_mae_pct, 6),
            "mae": round(float(np.nanmean(np.abs(ma4_vals - p2_actuals.values))), 4),
            "rmse": round(float(np.sqrt(np.nanmean((ma4_vals - p2_actuals.values) ** 2))), 4),
            "relative_bias": round(m4_bias_pct, 6),
            "official_reference": "DemandCast fast-reacting baseline",
        }
    )

    # 4. Seasonal Naive 52-week Baseline
    s52_cols = [w - pd.Timedelta(weeks=52) for w in weeks]
    s52_vals = history_sales[s52_cols].values
    s52_score, s52_mae_pct, s52_bias_pct = official_score(s52_vals, p2_actuals.values)
    results.append(
        {
            "evaluation_scope": "Full_Competition_Benchmark (15,053 series)",
            "model_name": "Baseline_SeasonalNaive52",
            "model_family": "Seasonal Lag (52w)",
            "series_count": len(p2_actuals),
            "total_instances": len(p2_actuals) * HORIZON_WEEKS,
            "vn1_score": round(s52_score, 6),
            "wape": round(s52_mae_pct, 6),
            "mae": round(float(np.nanmean(np.abs(s52_vals - p2_actuals.values))), 4),
            "rmse": round(float(np.sqrt(np.nanmean((s52_vals - p2_actuals.values) ** 2))), 4),
            "relative_bias": round(s52_bias_pct, 6),
            "official_reference": "Annual persistence baseline",
        }
    )

    logger.info("Completed full 15,053-series benchmark evaluation.")
    return results


def run_subset_evaluation(raw_dir: Path, config, subset_n: int = 250) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Execute evaluation across all 6 models on the 250-series development subset."""
    logger.info(f"--- Evaluating {subset_n}-Series Development Subset across All 6 Candidates ---")
    cube, _ = load_vn1_cube(config, subset_n_series=subset_n)
    active_series = sorted(cube["series_id"].unique().tolist())

    # Read Phase 2 actuals
    p2_df = pd.read_csv(raw_dir / "Phase 2 - Sales.csv")
    p2_df["series_id"] = (
        p2_df["Client"].astype(str) + "_" + p2_df["Warehouse"].astype(str) + "_" + p2_df["Product"].astype(str)
    )
    p2_sub = p2_df[p2_df["series_id"].isin(active_series)].copy()

    date_cols = [c for c in p2_sub.columns if c not in ["Client", "Warehouse", "Product", "series_id"]]
    p2_long = p2_sub.melt(
        id_vars=["series_id", "Client", "Warehouse", "Product"],
        value_vars=date_cols,
        var_name="week_start",
        value_name="actual_sales",
    )
    p2_long["week_start"] = pd.to_datetime(p2_long["week_start"])
    p2_long["actual_sales"] = p2_long["actual_sales"].fillna(0.0)

    # Origin is strictly 2024-01-01 (Phase 1 end)
    phase2_origin = pd.to_datetime(config.validation.phase2_origin)
    history_df = cube[cube["week_start"] <= phase2_origin].copy()

    # Generate forecasts for all 6 models
    # 1. Official 12w MA
    ma12 = VN1MovingAverageForecaster(window_size=12, name="Baseline_MovingAvg12_Official")
    ma12_preds = ma12.predict(history_df, horizon=13, origin_date=phase2_origin)

    # 2. DemandCast 4w MA
    ma4 = VN1MovingAverageForecaster(window_size=4, name="Baseline_MovingAvg4_DemandCast")
    ma4_preds = ma4.predict(history_df, horizon=13, origin_date=phase2_origin)

    # 3. Naive
    naive = VN1NaiveForecaster(name="Baseline_Naive")
    naive_preds = naive.predict(history_df, horizon=13, origin_date=phase2_origin)

    # 4. Seasonal Naive 52w
    s52 = VN1SeasonalNaiveForecaster(season_length=52, name="Baseline_SeasonalNaive52")
    s52_preds = s52.predict(history_df, horizon=13, origin_date=phase2_origin)

    # 5. LightGBM No Price
    lgb_noprice = VN1Forecaster(name="LightGBM_NoPrice", include_price=False)
    lgb_noprice.fit(history_df)
    lgb_noprice_preds = lgb_noprice.predict_recursive(history_df, horizon=13, origin_date=phase2_origin)

    # 6. LightGBM Price Aware (origin prices held constant, zero future leakage)
    lgb_price = VN1Forecaster(name="LightGBM_PriceAware", include_price=True)
    lgb_price.fit(history_df)
    lgb_price_preds = lgb_price.predict_recursive(history_df, horizon=13, origin_date=phase2_origin)

    all_preds = pd.concat(
        [ma12_preds, ma4_preds, naive_preds, s52_preds, lgb_noprice_preds, lgb_price_preds], ignore_index=True
    )

    # Merge with actuals
    merged = pd.merge(
        all_preds, p2_long[["series_id", "week_start", "actual_sales"]], on=["series_id", "week_start"], how="inner"
    )
    merged["error"] = merged["predicted_sales"] - merged["actual_sales"]
    merged["abs_error"] = merged["error"].abs()
    merged["forecast_origin"] = phase2_origin.strftime("%Y-%m-%d")

    # Mark frozen production model
    merged["is_production_model"] = merged["model_name"] == "Baseline_MovingAvg12_Official"

    # Compute metrics per model
    subset_results = []
    for m_name, grp in merged.groupby("model_name"):
        metrics = compute_all_vn1_metrics(grp["actual_sales"], grp["predicted_sales"])
        subset_results.append(
            {
                "evaluation_scope": f"Development_Subset ({subset_n} series)",
                "model_name": m_name,
                "model_family": (
                    "Gradient Boosted Trees"
                    if "LightGBM" in m_name
                    else "Trailing Mean"
                    if "MovingAvg" in m_name
                    else "Persistence"
                ),
                "series_count": len(grp["series_id"].unique()),
                "total_instances": len(grp),
                "vn1_score": round(metrics["vn1_score"], 6),
                "wape": round(metrics["wape"], 6),
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "smape": round(metrics["smape"], 4),
                "relative_bias": round(metrics["relative_bias"], 6),
                "official_reference": "N/A (Subset)",
            }
        )

    logger.info(f"Completed {subset_n}-series subset evaluation.")
    return subset_results, merged, p2_sub


def generate_phase2_figures(comp_df: pd.DataFrame, detailed_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Generate publication figures for Phase 2 holdout evaluation."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Model Comparison Chart (Subset vs Full)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subset plot
    sub_data = comp_df[comp_df["evaluation_scope"].str.contains("Development_Subset")].sort_values("vn1_score")
    sns.barplot(data=sub_data, x="vn1_score", y="model_name", hue="model_name", palette="viridis", legend=False, ax=ax1)
    ax1.set_title("Phase 2 Performance — 250-Series Development Subset", fontweight="bold")
    ax1.set_xlabel("Official VN1 Score (MAE% + |Bias%|)")
    ax1.set_ylabel("Model Candidate")
    for p in ax1.patches:
        val = p.get_width()
        ax1.annotate(
            f"{val:.4f}", (val, p.get_y() + p.get_height() / 2), xytext=(4, 0), textcoords="offset points", fontsize=9
        )

    # Full benchmark plot
    full_data = comp_df[comp_df["evaluation_scope"].str.contains("Full_Competition")].sort_values("vn1_score")
    sns.barplot(data=full_data, x="vn1_score", y="model_name", hue="model_name", palette="rocket", legend=False, ax=ax2)
    ax2.set_title("Phase 2 Performance — Full 15,053-Series Benchmark", fontweight="bold")
    ax2.set_xlabel("Official VN1 Score (MAE% + |Bias%|)")
    ax2.set_ylabel("")
    for p in ax2.patches:
        val = p.get_width()
        ax2.annotate(
            f"{val:.4f}", (val, p.get_y() + p.get_height() / 2), xytext=(4, 0), textcoords="offset points", fontsize=9
        )

    plt.suptitle("DemandCast v2 — VN1 Phase 2 Ground-Truth Evaluation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = figures_dir / "phase2_model_comparison.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(p1)

    # 2. Forecast Trajectory Examples vs Phase 2 Actuals
    sample_series = detailed_df["series_id"].unique()[:3]
    fig, axes = plt.subplots(len(sample_series), 1, figsize=(13, 3.5 * len(sample_series)), sharex=True)
    if len(sample_series) == 1:
        axes = [axes]

    for ax, sid in zip(axes, sample_series):
        sub = detailed_df[detailed_df["series_id"] == sid].sort_values("week_start")
        # Plot ground truth actuals
        actual_sub = sub[["week_start", "actual_sales"]].drop_duplicates().sort_values("week_start")
        ax.plot(
            actual_sub["week_start"],
            actual_sub["actual_sales"],
            label="Phase 2 Observed Sales",
            color="black",
            linewidth=2.5,
            marker="o",
        )

        for m_name in ["Baseline_MovingAvg12_Official", "Baseline_Naive", "LightGBM_PriceAware"]:
            m_sub = sub[sub["model_name"] == m_name].sort_values("week_start")
            ax.plot(m_sub["week_start"], m_sub["predicted_sales"], label=f"{m_name}", linestyle="--", alpha=0.85)

        ax.set_title(f"Phase 2 Holdout Trajectory — Series: {sid}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Weekly Sales Units")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p2 = figures_dir / "phase2_forecast_examples.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(p2)

    return generated


def main():
    root = get_project_root()
    raw_dir = root / "data" / "vn1" / "raw"
    reports_dir = root / "reports" / "vn1"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    config = load_vn1_config()

    logger.info("=== Starting Final VN1 Phase 2 Ground-Truth Evaluation ===")

    # 1. Full Benchmark Evaluation (15,053 series)
    full_results = run_full_benchmark_evaluation(raw_dir)

    # 2. Development Subset Evaluation (250 series across 6 models)
    subset_results, detailed_df, _ = run_subset_evaluation(raw_dir, config, subset_n=250)

    # 3. Combine into unified comparison report
    all_results = full_results + subset_results
    comp_df = pd.DataFrame(all_results)
    comp_path = reports_dir / "phase2_model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved Phase 2 model comparison report to {comp_path}")

    # 4. Save detailed predictions
    preds_path = reports_dir / "phase2_test_results.csv"
    detailed_df.to_csv(preds_path, index=False)
    logger.info(f"Saved detailed Phase 2 predictions with actuals to {preds_path}")

    # 5. Generate Figures
    figs = generate_phase2_figures(comp_df, detailed_df, figures_dir)
    logger.info(f"Generated {len(figs)} Phase 2 figures in {figures_dir}")

    # 6. Save comprehensive evaluation summary JSON
    registry = VN1ModelRegistry(config)
    curr_info = registry.get_current_release_info() or {}

    summary = {
        "evaluation_name": "DemandCast v2 VN1 Phase 2 Ground-Truth Holdout Evaluation",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "phase2_horizon": {
            "origin_date": "2024-01-01",
            "start_date": "2024-01-08",
            "end_date": "2024-04-01",
            "weeks_count": 13,
        },
        "active_production_release": {
            "model_version": curr_info.get("current_version", "v1.0.0"),
            "model_name": curr_info.get("metadata", {}).get("model_type", "Baseline_MovingAvg12_Official"),
            "checksum_sha256": curr_info.get("checksum_sha256"),
            "selection_basis": "Promoted strictly on Phase 1 validation results (origin 2023-10-02) prior to Phase 2 disclosure.",
            "production_governance_policy": "Phase 2 outcomes serve as an immutable holdout evaluation. Production models are NEVER altered post-hoc based on test holdout results to avoid data snooping.",
        },
        "price_feature_handling": {
            "policy": "Origin-only price projection (no future price leakage)",
            "description": "During recursive 13-week inference, price levels and price ratios known at origin (2024-01-01) are held constant forward across the horizon. No Phase 2 price information was accessed.",
            "causality_caveat": "Predictive improvements from price features reflect historical correlational patterns and do not establish causal price elasticity.",
        },
        "official_benchmark_reference_comparison": {
            "description": "Full 15,053 series evaluated using exact official vn1_score.py formula",
            "naive_benchmark_reproduced": {
                "measured_score": next(r["vn1_score"] for r in full_results if r["model_name"] == "Baseline_Naive"),
                "official_published": 0.506970,
                "relative_bias": next(r["relative_bias"] for r in full_results if r["model_name"] == "Baseline_Naive"),
            },
            "ma12_benchmark_reproduced": {
                "measured_score": next(
                    r["vn1_score"] for r in full_results if r["model_name"] == "Baseline_MovingAvg12_Official"
                ),
                "official_published": 0.804585,
                "relative_bias": next(
                    r["relative_bias"] for r in full_results if r["model_name"] == "Baseline_MovingAvg12_Official"
                ),
            },
        },
        "results_summary": all_results,
    }

    summary_path = reports_dir / "phase2_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Phase 2 evaluation summary JSON to {summary_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print("DEMANDCAST v2 — FINAL VN1 PHASE 2 GROUND-TRUTH EVALUATION RESULTS")
    print("=" * 95)
    print(
        comp_df[["evaluation_scope", "model_name", "vn1_score", "wape", "mae", "relative_bias"]].to_string(index=False)
    )
    print("=" * 95)


if __name__ == "__main__":
    main()
