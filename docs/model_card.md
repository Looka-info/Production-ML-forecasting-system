# DemandCast Model Card

## 1. Model Details
- **System**: DemandCast Weekly Retail Demand Forecasting System
- **Active Production Release (`v1.0.0`)**: 4-Week Moving Average Forecaster (`Baseline_MovingAvg4`).
- **Tracked ML Candidate Release (`v1.0.0-candidate-hgb`)**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor`) with Ordinal Categorical Encoding and Recursive Multi-Step Auto-regressive Forecasting.
- **Developer**: Senior ML Engineering Team
- **Release Date**: October 2026
- **Licence**: MIT (Software), CC BY 4.0 (Dataset Attribution)
- **Active Release Strategy Rationale**: The simple 4-week moving average baseline decisively outperformed the supervised ML model across all validation metrics (Validation WAPE: 55.16% vs. 88.30%; MASE: 0.5837 vs. 0.9680; MAE: 255.94 vs. 409.73; RMSE: 491.47 vs. 683.91). Adhering to production ML integrity and the documented business objective, the superior baseline was promoted to active production serving, while HistGradientBoosting is tracked in the registry as an experimental candidate.

---

## 2. Intended Use & Target Audience
- **Primary Use Case**: Forecasting weekly product demand up to 4 weeks ahead for replenishment planning, safety stock calibration, and supply chain inventory management.
- **Operating Context**: Automated weekly batch or on-demand API inference serving retail merchandising systems.
- **Out-of-Scope Uses**:
  - High-frequency intra-day or daily order generation.
  - Long-term strategic capacity planning (> 12 weeks).
  - Unconstrained market appetite estimation (does not account for lost sales during unrecorded stockouts).
  - Autonomous automated purchasing without human buyer sign-off.
  - Modern retail forecasting without retraining on contemporary transaction streams.

---

## 3. Training & Validation Data
- **Source**: UCI Machine Learning Repository — Online Retail II ([Chen, 2012](https://archive.ics.uci.edu/dataset/502/online+retail+ii)).
- **Time Span**: December 2009 to December 2011.
- **Raw Transaction Volume**: 1,067,371 rows across two annual sheets.
- **Data Filtering & Cleaning**:
  - Cancelled invoices (invoices prefixed with `C`) and returns excluded from primary demand target.
  - Non-positive quantities ($q \le 0$) and prices ($p \le 0$) removed.
  - Non-merchandise items (e.g., postage, bank charges, manual adjustments) removed.
  - Customer identifiers purged at ingestion to protect privacy.
- **Product Scope**: 15 high-volume, continuous product series selected strictly using the pre-cutoff training period (prior to `2010-10-01`) to eliminate lookahead bias.
- **Training Cutoff Date**: Observations up to `2011-11-07` are utilized for model fitting. Forecast requests with origins predating `2011-11-07` are rejected by the API as temporal contract violations.

---

## 4. Evaluation Methodology
- **Validation Framework**: Expanding-window chronological rolling-origin backtesting (3 distinct origins spaced 4 weeks apart: `2011-08-15`, `2011-09-12`, `2011-10-10`).
- **Held-Out Test Set**: 3 complete weeks of the dataset (`2011-11-14`, `2011-11-21`, `2011-11-28`, 45 evaluation observations) isolated completely as an untouched evaluation horizon from origin `2011-11-07`. The partial final week starting `2011-12-05` (collection ended Friday Dec 9 at 12:50 PM) is excluded from target evaluation to avoid evaluating incomplete demand.
- **Model Selection Protocol**: Selection is strictly determined by validation backtesting performance. Final held-out test results are never used to justify model promotion.
- **Primary Metric**: Weighted Absolute Percentage Error (WAPE) and Mean Absolute Scaled Error (MASE).
- **Secondary Metrics**: MAE, RMSE, sMAPE, and Relative Forecast Bias.
- **MASE Formulation**: Evaluated per series and origin using training-side in-sample naive persistence scaling:
  $$\text{MASE} = \frac{\text{MAE}}{\frac{1}{T-1} \sum_{t=2}^T |y_t - y_{t-1}|}$$
  Where the denominator is computed strictly from training history prior to the forecast origin. Denominators equal to zero (constant series) are explicitly handled by returning `None` to prevent division by zero.

---

## 5. Performance Summary & Empirical Validation

### 5.1 Validation Backtests (3 Rolling Origins, 180 Evaluation Observations)

| Model | Evaluation Stage | Origins | MAE (Units) | RMSE | WAPE | sMAPE (%) | MASE | Relative Bias | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline_MovingAvg4** | Validation Backtests | 3 | **255.94** | **491.47** | **55.16%** | **52.12%** | **0.5837** | **-5.31%** | **Active Production (`v1.0.0`)** |
| **Baseline_Naive** | Validation Backtests | 3 | 309.23 | 549.10 | 66.64% | 60.42% | 0.7380 | -0.46% | Benchmark |
| **HistGradientBoosting** | Validation Backtests | 3 | 409.73 | 683.91 | 88.30% | 74.70% | 0.9680 | +51.96% | Tracked Candidate (`v1.0.0-candidate-hgb`) |
| **Baseline_SeasonalNaive52**| Validation Backtests | 3 | 578.94 | 1311.08 | 124.77% | 82.33% | 1.2348 | +81.11% | Benchmark |

### 5.2 Untouched Held-Out Test Horizon (Origin 2011-11-07, 3 Complete Weeks, 45 Observations)

| Model | Evaluation Stage | Origins | MAE (Units) | RMSE | WAPE | sMAPE (%) | MASE | Relative Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline_MovingAvg4** | Final Held-Out Test (3w, 45 obs) | 1 | **221.54** | **373.52** | **49.41%** | **50.27%** | **0.5183** | **+16.25%** |
| **Baseline_Naive** | Final Held-Out Test (3w, 45 obs) | 1 | 247.33 | 398.41 | 55.16% | 52.41% | 0.6604 | +12.80% |
| **HistGradientBoosting** | Final Held-Out Test (3w, 45 obs) | 1 | 247.74 | 454.88 | 55.25% | 59.68% | 0.6099 | +23.13% |
| **Baseline_SeasonalNaive52**| Final Held-Out Test (3w, 45 obs) | 1 | 473.24 | 706.65 | 105.55% | 83.30% | 1.0858 | +79.60% |

### 5.3 Candidate Provenance & Evaluation Horizon Reconciliation
- **45-Observation Evaluation (Run `845005ef`, Authoritative)**: Evaluates the 3 complete calendar weeks (`2011-11-14` through `2011-11-28`). HGB achieved test MAE 247.74 and WAPE 55.25%; Baseline_MovingAvg4 achieved MAE 221.54 and WAPE 49.41%.
- **60-Observation Evaluation (Run `85f46113`, Pre-Audit)**: Evaluated 4 calendar weeks including partial week `2011-12-05` (data collection truncated Friday Dec 9 at 12:50 PM, ~50% actual volume), yielding HGB test MAE 264.80 and WAPE 64.60% (and Moving Average MAE 226.37, WAPE 55.23%).
- **No Test-Driven Model Tuning**: Hyperparameters were not tuned based on test feedback; validation results are identical across both runs (WAPE 88.30%). Production model selection remains based solely on chronological validation data.

---

## 6. Feature Schema & History Depth Requirements

The supervised ML model operates on a verified 18-feature schema when `include_seasonal_lag: false`:
1. `series_id` (Categorical, Ordinal Encoded).
2. Lags (6 features): `lag_1`, `lag_2`, `lag_3`, `lag_4`, `lag_8`, `lag_12`.
3. Shifted Rolling Statistics (6 features): `rolling_mean_4`, `rolling_std_4`, `rolling_mean_8`, `rolling_std_8`, `rolling_mean_12`, `rolling_std_12`.
4. Zero-Demand Frequency (1 feature): `zero_freq_4`.
5. Calendar & Cyclical Signals (4 features): `week_of_year`, `month`, `sin_week`, `cos_week`.
*Note on 19th feature*: If `include_seasonal_lag: true`, `lag_52` is included as the 19th feature, requiring at least 53 consecutive weeks of history.

**Minimum History Depth Contract**:
- Active Moving Average Baseline (`v1.0.0`): Requires $\ge 4$ weeks of history.
- Supervised ML Model (`v1.0.0-candidate-hgb`): Requires $\ge 13$ weeks of history (or $\ge 53$ weeks if `seasonal_lag_52` is enabled).
- The API dynamically enforces each model's history requirements, returning HTTP 422 if insufficient history is provided.

---

## 7. Limitations & Operational Boundaries
1. **Quality Alert Status**: The promoted baseline (`v1.0.0`) exhibits validation WAPE of 55.16%, which exceeds the target operational alert threshold of 50.00%. It was promoted under an audited operational override because it decisively out-performed the candidate ML model (88.30% WAPE) on identical rolling-origin evaluations. The alert status is transparently reported via `/ready`, `/metadata`, and `reports/monitoring_report.json`.
2. **Historical Data Limitation & Model Freshness**: The active model was trained on historical data ending November 7, 2011 (`stale_historical_dataset`). Predictions for inputs dated after the training cutoff represent historical backtesting/replay and do not establish validity in modern retail environments. The API automatically returns a `StaleModelWarning` if input history origin exceeds the training cutoff date by $> 90$ days.
3. **Data Refresh Requirements**: To refresh the model for live operations, practitioners must:
   - Ingest contemporary retail transaction streams (ERP/POS).
   - Aggregate to Monday week-start intervals ($[Monday\ 00:00:00, Sunday\ 23:59:59.999999]$).
   - Ensure $\ge 52$ continuous weeks of history per SKU to support seasonality modeling.
   - Re-run expanding-window chronological validation backtesting.
   - Achieve validation WAPE $\le 50.00\%$ before non-override promotion.
4. **Sales Proxy Limitation**: Positive sales reflect observed transactions, not unconstrained demand. If an item was out of stock, observed sales are zero even if customer demand was high.
5. **Temporal Uniformity**: The model requires strictly consecutive weekly observations starting on Monday. Missing or irregular intervals are rejected.
6. **Product Granularity**: Predictions are calibrated for the supported product series catalog. Unsupported SKUs are rejected by the API with HTTP 400.
7. **Deployment Status**: Multi-stage Docker containerization and compose orchestration are implemented; local execution is verified when the host Docker Desktop engine is started.

---

## 8. Governance, Drift & Rollback
- **Artifact Verification**: Immutable versioned release directory containing `model.joblib` and `metadata.json` with SHA-256 checksums.
- **Promotion Gate**: Candidate models must pass validation error checks and automated inference smoke tests before updating the active pointer `models/current.json`.
- **Operational Alerts**:
  - WAPE $> 50\%$ or normalized relative Bias $> \pm 25\%$ triggers an alert.
  - Two-sample Kolmogorov-Smirnov distribution test ($p < 0.05$) serves as a descriptive marginal distribution diagnostic.
  - Residual tracking monitors error distribution per horizon step and per SKU.
- **Rollback Protocol**: Instantaneous atomic rollback via `python scripts/rollback_model.py --to <version>`.
