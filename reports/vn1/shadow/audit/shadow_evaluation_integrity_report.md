# DemandCast v2.2.1 — Shadow Evaluation Integrity Audit Report

**Date**: 2026-10-25
**Audit Target**: DemandCast VN1 Shadow Pipeline (`scripts/run_vn1_shadow.py`, `src/demandcast/vn1/shadow.py`)
**Status**: **AUDIT_COMPLETE (PASSED WITH CALENDAR CLARIFICATION)**
**Production Integrity**: **CONFIRMED UNCHANGED** (`v1.0.0`, `Baseline_MovingAvg12_Official`)
**Candidate D Governance**: **EXPERIMENTAL RESEARCH CANDIDATE (PROMOTION DENIED/NOT APPLIED)**

---

## 1. Executive Summary & Production Protection

- **VN1 Active Release**: `models/profiles/vn1/current.json` strictly verified at `v1.0.0` (`Baseline_MovingAvg12_Official`, SHA-256: `2e98277531f730db4d57b6e58bf7da2c62998e5341946da58ec202a9a65ab809`).
- **UCI Active Release**: `models/current.json` strictly verified at `v1.0.0` (`Baseline_MovingAvg4`, SHA-256: `428ea2d98439c98efcccc8ebafbecf057a9e1fd54a56ba26edd8e386bcd9b728`).
- **Phase 2 Holdout Actuals**: `Phase 2 - Sales.csv` SHA-256: `ef5b998226495fdfc44ea4dcda8df9b45b1c76f6adb50441e2217ad3a623c6ac` intact.
- **Candidate D Status**: Explicitly maintained as an experimental research candidate (`is_candidate = 1`). No automated promotion was executed.

---

## 2. Explanation of the 600 / 50 Calendar Split

The user requested an audit of why the 13-week shadow replay generated **600 post-holiday observations** and **50 ordinary-calendar observations** across 50 series.

### Mathematical & Calendar Derivation

In standard retail time-series forecasting, an origin date $T$ is the **last historical observation**. Forecast horizon steps $h \in [1, \dots, 13]$ project forward to target dates $T + h$ weeks:
- **Forecast Origin**: `2023-01-02` (Monday, ISO Week 1 of 2023). This is historical data, not a forecast target.
- **Horizon Steps and Target Weeks**:
  - $h=1$: `2023-01-09` $\rightarrow$ **ISO Week 2** (Post-Holiday Window)
  - $h=2$: `2023-01-16` $\rightarrow$ **ISO Week 3** (Post-Holiday Window)
  - $h=3$: `2023-01-23` $\rightarrow$ **ISO Week 4** (Post-Holiday Window)
  - $h=4$: `2023-01-30` $\rightarrow$ **ISO Week 5** (Post-Holiday Window)
  - $h=5$: `2023-02-06` $\rightarrow$ **ISO Week 6** (Post-Holiday Window)
  - $h=6$: `2023-02-13` $\rightarrow$ **ISO Week 7** (Post-Holiday Window)
  - $h=7$: `2023-02-20` $\rightarrow$ **ISO Week 8** (Post-Holiday Window)
  - $h=8$: `2023-02-27` $\rightarrow$ **ISO Week 9** (Post-Holiday Window)
  - $h=9$: `2023-03-06` $\rightarrow$ **ISO Week 10** (Post-Holiday Window)
  - $h=10$: `2023-03-13` $\rightarrow$ **ISO Week 11** (Post-Holiday Window)
  - $h=11$: `2023-03-20` $\rightarrow$ **ISO Week 12** (Post-Holiday Window)
  - $h=12$: `2023-03-27` $\rightarrow$ **ISO Week 13** (Post-Holiday Window)
  - $h=13$: `2023-04-03` $\rightarrow$ **ISO Week 14** (Ordinary Calendar Window!)

### The Exact Cause
Under the documented ISO definition (`post_holiday_start_week = 1`, `post_holiday_end_week = 13`):
1. **Target Weeks $h=1$ to $12$** fall into ISO weeks 2 through 13.
   - For 50 series, this produces $12 \times 50 = \mathbf{600}$ observations.
   - For these 12 weeks, Candidate D applied its shrunk post-holiday multiplier (mean factor = 0.9673).
2. **Target Week $h=13$** (`2023-04-03`) falls into ISO week 14.
   - For 50 series, this produces $1 \times 50 = \mathbf{50}$ observations.
   - Because ISO week 14 is outside weeks 1..13, Candidate D's targeting condition returned `1.0` (neutral, unadjusted).
   - Consequently, for $h=13$, Candidate D and Production had **100% identical predictions**, producing an exact delta of **+0.0000** (both models scored 0.4339).

The 600/50 observation split is **mathematically exact and completely consistent** with the documented ISO calendar window.

---

## 3. Official VN1 Metric Verification & Parity

- **Formula**:
  $$\text{VN1 Score} = \frac{\sum_i |\hat{y}_i - y_i| + \left|\sum_i (\hat{y}_i - y_i)\right|}{\sum_i y_i}$$
- **Verification Findings**:
  - The shadow evaluator directly calls `compute_all_vn1_metrics` from `demandcast.vn1.metrics`.
  - Portfolio bias $\left|\sum_i (\hat{y}_i - y_i)\right|$ is computed globally across the entire evaluated slice.
  - Zero actual demand handling: Returns 0.0 when both predicted and actual demand are 0.0; returns NaN if actual is 0.0 but predictions are non-zero.
  - Slicing integrity: Slices (`by_horizon`, `post_holiday_segment`, `ordinary_segment`) are aggregated within their respective sub-populations, preserving true non-linear portfolio bias.

---

## 4. Exact Reproduction of Reported Shadow Numbers

Re-evaluating the 650 matched forecast pairs from SQLite storage matches the reported numbers with **zero error**:

| Metric | Reported Production | Recomputed Production | Diff | Reported Candidate D | Recomputed Candidate D | Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VN1 Score** | 0.4264 | 0.4264 | 0.0000 | 0.3570 | 0.3570 | 0.0000 |
| **WAPE** | 0.3706 | 0.3706 | 0.0000 | 0.3121 | 0.3121 | 0.0000 |
| **MAE** | 405.5518 | 405.5518 | 0.0000 | 341.6112 | 341.6112 | 0.0000 |
| **RMSE** | 783.8310 | 783.8310 | 0.0000 | 700.1719 | 700.1719 | 0.0000 |
| **Rel Bias** | 0.0559 | 0.0559 | 0.0000 | 0.0448 | 0.0448 | 0.0000 |

---

## 5. Reconciliation & Point-in-Time Integrity

- **Leakage Check**: Passed. The pipeline strictly filters input snapshots to `week_start <= origin_date`. Rows with future timestamps never enter feature creation or baseline estimation.
- **Immutability Check**: Passed. Forecasts written at inference time are never overwritten. Reconciling actuals updates `actual_sales` and `reconciliation_status = 'MATCHED'` without modifying `predicted_sales` or timestamps.
- **Idempotency**: Repeatedly running the same snapshot skips existing forecasts (`0 inserted, N skipped`). Repeatedly ingesting the same actuals skips duplicates.
- **Conflict Rejection**: Conflicting actuals arriving for a matched forecast trigger `CONFLICT_REJECTED` and are blocked from corrupting audit records.

---

## 6. Cross-Experiment Variance Analysis (50-Series vs. 250-Series Study)

Why did the 50-series shadow run yield VN1 score = 0.4264 (Production) and 0.3570 (Candidate D), while the earlier 250-series study reported 0.3813 (Production) and 0.3619 (Candidate D)?

1. **Evaluation Scope**:
   - The earlier study pooled results across **4 historical origins** (`2022-01-03`, `2022-07-04`, `2023-01-02`, `2023-04-03`), combining both holiday and mid-year cycles.
   - The shadow run evaluated a **single post-holiday origin** (`2023-01-02`).
   - Looking strictly at origin `2023-01-02` in both studies:
     - 250 series: Production = 0.4988, Candidate D = 0.4488 (Delta: -0.0500).
     - 50 series: Production = 0.4264, Candidate D = 0.3570 (Delta: -0.0694).
2. **Catalog Volume & Sparsity Differences**:
   - The 50 series are the highest-volume products in Phase 0 up to `2023-01-02`. Higher sales volumes reduce the intermittency penalty, lowering WAPE (37% vs 44%) and improving baseline predictability.
3. **Model Configuration Parity**:
   - Candidate D's architecture, window length ($12w$), shrinkage parameter ($k=100.0$), bounds ($[0.50, 1.25]$), and targeting logic are **identical across both experiments**.

---

## 7. Audit Conclusion & Recommendation

1. **Integrity Verdict**: **AUDIT_PASSED**. The shadow pipeline exhibits strict point-in-time correctness, exact metric reproduction, robust idempotency, and proper failure isolation.
2. **Calendar Clarification**: The 600/50 split is mathematically exact because origin `2023-01-02` projects 13 weeks forward into ISO week 14 (`2023-04-03`), which correctly lies outside the ISO 1..13 post-holiday window.
3. **Operational Recommendation**: The shadow pipeline architecture is verified and ready for prospective feed testing. Candidate D remains an **experimental candidate** and must not be promoted without live-replenishment verification.
