"""Tests for VN1 Phase 2 ground-truth evaluation, official scoring, and report verification."""

import json

import numpy as np
import pandas as pd
import pytest

from demandcast.config import get_project_root
from demandcast.vn1.metrics import calculate_vn1_score, compute_all_vn1_metrics


def test_official_scoring_mathematical_hand_calculation():
    """Verify official VN1 score formula matches hand-calculated MAE% + |Bias%|."""
    # Actual sales: [10, 20, 30] -> Total = 60
    # Over-forecasted: [12, 22, 32] -> Errors: [+2, +2, +2]
    # MAE = 6 / 60 = 0.10
    # Bias = +6 / 60 = +0.10
    # Score = 0.10 + |0.10| = 0.20 (20%)
    actuals = np.array([10.0, 20.0, 30.0])
    preds = np.array([12.0, 22.0, 32.0])

    score = calculate_vn1_score(actuals, preds)
    metrics = compute_all_vn1_metrics(actuals, preds)
    assert pytest.approx(score, rel=1e-5) == 0.20
    assert pytest.approx(metrics["wape"], rel=1e-5) == 0.10
    assert pytest.approx(metrics["relative_bias"], rel=1e-5) == 0.10

    # Under-forecasted: [8, 18, 28] -> Errors: [-2, -2, -2]
    # MAE = 6 / 60 = 0.10
    # Bias = -6 / 60 = -0.10
    # Score = 0.10 + |-0.10| = 0.20
    preds_under = np.array([8.0, 18.0, 28.0])
    score_u = calculate_vn1_score(actuals, preds_under)
    metrics_u = compute_all_vn1_metrics(actuals, preds_under)
    assert pytest.approx(score_u, rel=1e-5) == 0.20
    assert pytest.approx(metrics_u["wape"], rel=1e-5) == 0.10
    assert pytest.approx(metrics_u["relative_bias"], rel=1e-5) == -0.10


def test_phase2_raw_dataset_schema_and_integrity():
    """Verify Phase 2 sales file matches competition specifications."""
    root = get_project_root()
    p2_path = root / "data" / "vn1" / "raw" / "Phase 2 - Sales.csv"
    assert p2_path.exists(), f"Phase 2 actuals missing at {p2_path}"

    df = pd.read_csv(p2_path)
    assert df.shape[0] == 15053, f"Expected 15,053 series, got {df.shape[0]}"
    assert df.shape[1] == 16, f"Expected 16 columns (3 IDs + 13 weeks), got {df.shape[1]}"

    # Verify ID columns
    for col in ["Client", "Warehouse", "Product"]:
        assert col in df.columns

    # Verify date columns conform to Monday week-start from 2024-01-08 to 2024-04-01
    date_cols = [c for c in df.columns if c not in ["Client", "Warehouse", "Product"]]
    dates = pd.to_datetime(date_cols)
    assert len(dates) == 13
    assert (dates.day_of_week == 0).all(), "All dates must be Mondays"
    assert dates.min() == pd.to_datetime("2024-01-08")
    assert dates.max() == pd.to_datetime("2024-04-01")

    # Verify no missing values in actuals
    assert not df[date_cols].isna().any().any(), "Phase 2 actuals contain unexpected NaN values"


def test_phase2_evaluation_artifacts_exist_and_consistent():
    """Verify that Phase 2 model comparison, predictions, summary JSON, and figures exist."""
    root = get_project_root()
    reports_dir = root / "reports" / "vn1"

    comp_path = reports_dir / "phase2_model_comparison.csv"
    assert comp_path.exists()
    comp_df = pd.read_csv(comp_path)
    assert "Full_Competition_Benchmark (15,053 series)" in comp_df["evaluation_scope"].values
    assert "Development_Subset (250 series)" in comp_df["evaluation_scope"].values

    # Verify full benchmark Naive score matches official 0.5070
    naive_row = comp_df[
        (comp_df["evaluation_scope"].str.contains("Full_Competition")) & (comp_df["model_name"] == "Baseline_Naive")
    ].iloc[0]
    assert pytest.approx(naive_row["vn1_score"], abs=0.001) == 0.506970

    # Verify full benchmark 12w MA score matches official 0.8046
    ma12_row = comp_df[
        (comp_df["evaluation_scope"].str.contains("Full_Competition"))
        & (comp_df["model_name"] == "Baseline_MovingAvg12_Official")
    ].iloc[0]
    assert pytest.approx(ma12_row["vn1_score"], abs=0.001) == 0.804585

    # Verify summary JSON
    summary_path = reports_dir / "phase2_evaluation_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "active_production_release" in summary
    assert "price_feature_handling" in summary
    assert "official_benchmark_reference_comparison" in summary

    # Verify figures
    p1 = reports_dir / "figures" / "phase2_model_comparison.png"
    p2 = reports_dir / "figures" / "phase2_forecast_examples.png"
    assert p1.exists() and p1.stat().st_size > 10000
    assert p2.exists() and p2.stat().st_size > 10000
