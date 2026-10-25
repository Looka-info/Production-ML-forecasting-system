# DemandCast — Complete Production ML Forecasting System

[![DemandCast CI](https://github.com/mumer/demandcast/actions/workflows/ci.yml/badge.svg)](https://github.com/mumer/demandcast/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**DemandCast** is a complete, production-grade retail demand forecasting and model serving system developed as an advanced Machine Learning Engineering capstone. It takes a weekly demand forecasting problem through the entire ML lifecycle: dataset ingestion and provenance tracking, auditable data cleaning, training-only product selection, leakage-free feature engineering, chronological backtesting, MLflow experiment tracking, immutable release versioning, FastAPI serving, Docker containerization, post-forecast quality monitoring, and automated rollback workflows.

---

## 1. Business Problem & Forecasting Objective

Retail replenishment planning requires anticipating future weekly sales volumes per product SKU while balancing stockout penalties against inventory holding costs.

- **Entity**: Product SKU / Series Identifier (`series_id`).
- **Time Granularity**: Weekly, anchored strictly to **Monday week-start** (`W-MON`).
- **Prediction Target**: Total eligible positive sales units per selected SKU during each calendar week.
- **Forecast Horizon**: 1 to 4 weeks forward ($h \in \{1, 2, 3, 4\}$).
- **Observed Sales vs. Unconstrained Demand**: The underlying transaction dataset does not record inventory stock levels. Therefore, observed positive sales serve as an **observed demand proxy**. Weeks with zero observed sales indicate either zero customer demand or unrecorded stockouts. DemandCast documents this distinction explicitly and avoids claiming that predictions represent unconstrained market appetite.

---

## 2. Dataset Provenance & Attribution

- **Dataset**: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Original Source Attribution**: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Time Span**: December 1, 2009 to December 9, 2011 (2 annual transaction sheets).
- **Raw Transaction Volume**: 1,067,371 rows.
- **Archive File Checksum (SHA-256)**:
  `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` (45,622,418 bytes)
- **Extracted File**: `online_retail_II.xlsx` (45,622,278 bytes)

---

## 3. Auditable Preprocessing & Data Cleaning

Data cleaning rules are defined in `configs/config.yaml` and executed via `scripts/prepare_data.py`. Cleaning results are recorded in `reports/data_quality_report.json`:

| Cleaning Step | Exclusion Criterion | Count Excluded | Retained / Action |
| :--- | :--- | :--- | :--- |
| **Cancelled Invoices** | Invoice ID prefixed with `C` | 19,494 rows | Excluded from positive demand |
| **Non-Positive Quantities** | Quantity $\le 0$ (returns/adjustments) | 3,457 rows | Excluded from positive sales proxy |
| **Invalid Unit Prices** | Unit Price $< £0.001$ | 2,750 rows | Excluded from commercial transactions |
| **Non-Merchandise Codes** | Codes: `POST`, `D`, `M`, `BANK CHARGES`, `PADS`, `DOT`, `CRUK`, `TEST` | 4,299 rows | Excluded from merchandise demand |
| **Duplicate Transactions** | Exact duplicates on invoice, stock, qty, date, price | 33,672 rows | Deduplicated |
| **Customer ID Sanitization** | `Customer ID` stripped at ingestion | — | Purged to ensure customer privacy |
| **Final Retained Transactions** | Valid merchandise sales | **1,003,699 rows** | **94.03% row retention** |

---

## 4. Product Selection Without Lookahead Leakage

Selecting high-volume products using full-dataset totals introduces **lookahead bias** because future performance dictates model inclusion. DemandCast resolves this by filtering products **strictly using transactions prior to `2010-10-01`**:

1. **Candidate Pool**: 3,923 products with transaction history before the cutoff date.
2. **Eligibility Gates**: $\ge 35$ active transaction weeks and $\ge 500$ total units sold in the training window (718 eligible candidates).
3. **Selected Catalog (Top 15)**:
   `['21212', '85123A', '85099B', '84077', '84991', '17003', '84879', '21977', '21980', '84755', '21984', '21232', '84270', '21213', '71459']`
4. Stored in immutable catalog `data/processed/selected_series.json`.

---

## 5. Weekly Aggregation & Feature Engineering

### 5.1 Weekly Calendar Grid & Boundary Semantics
Transactions are grouped by product SKU and mapped to Monday week-start (`to_monday_week_start`) with explicit interval semantics: $[Monday\ 00:00:00,\ Sunday\ 23:59:59.999999] \to Monday\ 00:00:00$. This prevents the discrepancy inherent in pandas default `resample("W-MON")`, which defaults to right-closed $(Tuesday..Monday]$ and misassigns weekend transactions into the following week.

Furthermore, raw collection in the UCI Online Retail II dataset abruptly ends on **Friday, December 9, 2011 at 12:50:00 PM**. Under Monday-start convention, the week of `2011-12-05` contains only 4.5 days of transactions (missing Friday afternoon, Saturday, and Sunday). DemandCast automatically detects and drops this incomplete final week (`drop_incomplete_final_week: true`), yielding a clean dataset of **105 complete calendar weeks** (2009-11-30 to 2011-11-28) across 15 SKUs (1,575 total observations). Intermediate missing weeks are filled with zero demand.

### 5.2 Leakage-Safe Features
At forecast target week $t$, only information prior to $t$ is utilized:
- **Historical Lags** (6 features): $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}, y_{t-8}, y_{t-12}$. (Optional 19th feature $y_{t-52}$ is included when `include_seasonal_lag: true`).
- **Shifted Rolling Statistics** (6 features): Rolling windows apply `.shift(1)` to ensure demand at week $t$ never enters rolling calculations:
  - 4-week, 8-week, 12-week rolling means.
  - 4-week, 8-week, 12-week rolling standard deviations.
- **Zero-Demand Frequency** (1 feature): 4-week trailing zero-demand frequency (`zero_freq_4`).
- **Calendar Signals** (4 features): Week of year ($1..52$), Month ($1..12$), and cyclical sinusoidal encodings $\sin(2\pi \cdot \text{week} / 52)$, $\cos(2\pi \cdot \text{week} / 52)$.
- **Categorical Series Feature** (1 feature): Product SKU encoded with `OrdinalEncoder` fitted only on training data.
- **Total Feature Count**: **18 features** in standard configuration (`include_seasonal_lag: false`); **19 features** if seasonal lag 52 is enabled.

---

## 6. Benchmarking & Empirical Evaluation

### 6.1 Evaluation Protocol
- **Expanding-Window Chronological Backtests**: 3 validation origins (`2011-08-15`, `2011-09-12`, `2011-10-10`) with 4-week horizons evaluated across all 15 SKUs (180 validation forecast instances). All validation origins precede the test horizon.
- **Held-Out Final Test Set**: Cutoff `2011-11-07`, evaluating the subsequent untouched 3 complete weeks (`2011-11-14`, `2011-11-21`, `2011-11-28`, exactly 45 test forecast instances). Incomplete week `2011-12-05` (data collection truncated Friday Dec 9) is excluded to protect against partial-week target distortion.
- **Model Selection Rule**: Selection is determined exclusively by validation backtest performance against the business objective of minimizing volume-weighted absolute percentage error (WAPE) and mean absolute scaled error (MASE). Final held-out test results are never used to justify model promotion.

### 6.2 Empirical Results (`reports/model_comparison.csv`)

| Model | Evaluation Stage | Origins | MAE (Units) | RMSE | WAPE | sMAPE (%) | MASE | Relative Bias | Promotion Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline_MovingAvg4** | Validation Backtests | 3 | **255.94** | **491.47** | **55.16%** | **52.12%** | **0.5837** | **-5.31%** | **Active Production (`v1.0.0`)** |
| **Baseline_Naive** | Validation Backtests | 3 | 309.23 | 549.10 | 66.64% | 60.42% | 0.7380 | -0.46% | Benchmark |
| **HistGradientBoosting** | Validation Backtests | 3 | 409.73 | 683.91 | 88.30% | 74.70% | 0.9680 | +51.96% | Tracked Candidate (`v1.0.0-candidate-hgb`) |
| **Baseline_SeasonalNaive52** | Validation Backtests | 3 | 578.94 | 1311.08 | 124.77% | 82.33% | 1.2348 | +81.11% | Benchmark |
| | | | | | | | | | |
| **Baseline_MovingAvg4** | Held-Out Final Test (3w, 45 obs) | 1 | **221.54** | **373.52** | **49.41%** | **50.27%** | **0.5183** | **+16.25%** | **Confirmatory Evaluation** |
| **Baseline_Naive** | Held-Out Final Test (3w, 45 obs) | 1 | 247.33 | 398.41 | 55.16% | 52.41% | 0.6604 | +12.80% | Benchmark |
| **HistGradientBoosting** | Held-Out Final Test (3w, 45 obs) | 1 | 247.74 | 454.88 | 55.25% | 59.68% | 0.6099 | +23.13% | Tracked Candidate |
| **Baseline_SeasonalNaive52** | Held-Out Final Test (3w, 45 obs) | 1 | 473.24 | 706.65 | 105.55% | 83.30% | 1.0858 | +79.60% | Benchmark |

### 6.3 Scientific Analysis & Model Selection Integrity
1. **The Reality of Retail Intermittency & Volatility**: The 4-week Moving Average benchmark decisively outperforms the supervised ML model across all validation metrics (Validation WAPE: 55.16% vs. 88.30%; MASE: 0.5837 vs. 0.9680; MAE: 255.94 vs. 409.73; RMSE: 491.47 vs. 683.91).
2. **Promoting the Best Performer**: In strict adherence to ML engineering integrity, DemandCast promotes the winning **4-week Moving Average baseline** as the active production release (`v1.0.0`). The project avoids complex model bias: a complex model is not deployed when a simple baseline delivers superior accuracy.
3. **Tracking the ML Candidate**: `HistGradientBoosting` is retained and tracked in the model registry as `v1.0.0-candidate-hgb` for continuous research and hyperparameter optimization.
4. **Honest Evaluation Protocol**: Test set performance across the 3 complete held-out weeks (where Baseline_MovingAvg4 also led with 49.41% WAPE vs. 55.25% for HGB) is preserved purely for confirmatory evaluation and was never leaked to influence model selection decisions.
5. **Evaluation Horizon Provenance Reconciliation**:
   - **45-Observation Corrected Holdout (Authoritative, Run `845005ef`)**: Evaluates the 3 complete calendar weeks (`2011-11-14` through `2011-11-28`). HGB test MAE is **247.74** and WAPE is **55.25%**; Baseline_MovingAvg4 test MAE is **221.54** and WAPE is **49.41%**.
   - **60-Observation Pre-Audit Holdout (Historical, Run `85f46113`)**: Included the 4th calendar week (`2011-12-05`). Because the raw UCI data ended Friday Dec 9 at 12:50 PM, this 4th week had only 4.5 days observed (~50% actual volume), causing all models to appear over-predictive and elevating HGB test MAE to **264.80** and WAPE to **64.60%** (and Moving Average MAE to **226.37**, WAPE to **55.23%**).
   - **Hyperparameter Stability**: Neither model hyperparameters nor feature schemas were tuned in response to test feedback. Validation metrics are identical across both runs (HGB Val WAPE 88.30%, Baseline Val WAPE 55.16%).

---

## 7. Model Governance, Quality Alerts & Freshness Safeguards

### 7.1 Promotion Gates vs. Quality Alerts
DemandCast establishes a strict, audited boundary between candidate model selection and operational quality gates:
1. **Selection Rule**: Candidates are ranked by validation WAPE. The 4-week Moving Average baseline wins with 55.16% WAPE (vs. 88.30% for HistGradientBoosting).
2. **Quality Alert Gate**: The operational target threshold is configured at `wape_promotion_gate_threshold: 0.50` (50.00%). The promoted baseline's validation WAPE (55.16%) exceeds this target.
3. **Denied Overrides by Default**: The system policy (`allow_alert_promotion_override: false`) strictly denies promotion of any candidate exceeding the threshold unless an explicit `--allow-alert-override` flag AND a documented, non-empty `--reason <text>` string are supplied.
4. **Audited Operational Override**: `Baseline_MovingAvg4` was promoted under an explicit audited override because it decisively minimizes error compared to the ML model.
5. **Transparent Alert Status**: The alert is **never hidden or suppressed**. Active pointer `models/current.json`, `/ready`, `/metadata`, and monitoring reports all explicitly broadcast `quality_status: "ALERT"`.

### 7.2 Model Freshness & Stale-Model Warning
The active release `v1.0.0` was trained on historical transactions ending **November 7, 2011**:
- **Dataset Vintage**: UCI Online Retail II (2009–2011).
- **Modern Retail Disclaimer**: The model is **not calibrated for modern retail markets** without contemporary retraining. Predictions on historical inputs represent retrospective simulation / backtesting only.
- **Configurable Stale-Model Warning**: If an inference request provides an origin date exceeding the model training cutoff by more than `stale_model_threshold_days: 90`, the API automatically attaches a `StaleModelWarning` to the prediction payload:
  `"StaleModelWarning: Forecast origin is X days past model training cutoff (2011-11-07). Model was trained on historical data (2009-2011); predictions beyond the training horizon represent historical backtesting/replay and do not establish validity for modern retail markets."`
- **Historical Reproducibility**: Requests ending on the cutoff date (`2011-11-07`) run cleanly without stale warnings, ensuring historical verification remains fully reproducible.

### 7.3 Model Refresh & Retraining Protocol
To refresh the model for live operations:
1. Ingest contemporary retail transaction feeds (ERP/POS).
2. Aggregate to Monday week-start intervals ($[Monday\ 00:00:00, Sunday\ 23:59:59.999999]$).
3. Ensure $\ge 52$ consecutive weeks of continuous history for each SKU to support seasonal modeling.
4. Re-run expanding-window chronological validation backtesting.
5. Verify validation WAPE $\le 50.00\%$ before promoting without operational override.

### 7.4 Versioned Registry Commands
Releases are managed via `src/demandcast/registry.py` under `models/releases/<version>/`:
- `model.joblib`: Serialized forecaster bundle and metadata.
- `metadata.json`: SHA-256 artifact checksum, validation metrics, test metrics, quality status, model freshness, and MLflow run ID.
- `models/current.json`: Atomic pointer pointing to the active release.

```powershell
# Promote a release (runs artifact verification, inference smoke test, and quality gate check)
python scripts/promote_model.py --version v1.0.0 --reason "Promoted winning 4-week moving average baseline under audited override"

# Roll back to a previous release atomically
python scripts/rollback_model.py --to v1.0.0 --reason "Operational rollback requested"
```

---

## 8. REST API & Serving Layer

The production API is built with **FastAPI** (`src/demandcast/api/`):

### Endpoints
- `GET /health`: Liveness probe indicating server process health (`{"status": "healthy"}`).
- `GET /ready`: Readiness probe verifying that model release, metadata, quality status, and freshness are loaded. Returns `quality_status: "ALERT"`, `quality_alert`, and `model_freshness_status: "stale_historical_dataset"`. Returns HTTP 503 if not ready.
- `GET /metadata`: Governance endpoint returning model version, quality status, detailed model freshness (dataset vintage, training cutoff, refresh requirements), validation metrics, and limitations.
- `POST /forecast`: Generates recursive 4-week point forecasts. Rejects origins predating training cutoff (`2011-11-07`) with HTTP 400 `TemporalContractViolation`; attaches `StaleModelWarning` if origin exceeds cutoff by $> 90$ days.

### Example Request (`POST /forecast`)
Using historical weekly sales observations through the model's training cutoff date (`2011-11-07`):
```json
{
  "series_id": "17003",
  "horizon": 4,
  "history": [
    { "week_start": "2011-08-08", "demand": 140.0 },
    { "week_start": "2011-08-15", "demand": 78.0 },
    { "week_start": "2011-08-22", "demand": 940.0 },
    { "week_start": "2011-08-29", "demand": 291.0 },
    { "week_start": "2011-09-05", "demand": 180.0 },
    { "week_start": "2011-09-12", "demand": 109.0 },
    { "week_start": "2011-09-19", "demand": 189.0 },
    { "week_start": "2011-09-26", "demand": 412.0 },
    { "week_start": "2011-10-03", "demand": 218.0 },
    { "week_start": "2011-10-10", "demand": 420.0 },
    { "week_start": "2011-10-17", "demand": 380.0 },
    { "week_start": "2011-10-24", "demand": 490.0 },
    { "week_start": "2011-10-31", "demand": 620.0 },
    { "week_start": "2011-11-07", "demand": 607.0 }
  ]
}
```

### Example Response
```json
{
  "forecast_run_id": "8f3b251a-4bc1-49b8-a612-92e105db089e",
  "created_at": "2026-10-25T10:00:00.000000+00:00",
  "series_id": "17003",
  "model_version": "v1.0.0",
  "model_name": "Baseline_MovingAvg4",
  "last_historical_date": "2011-11-07",
  "forecast_start_date": "2011-11-14",
  "forecast_end_date": "2011-12-05",
  "horizon": 4,
  "forecasts": [
    { "week_start": "2011-11-14", "horizon_step": 1, "predicted_demand": 524.25 },
    { "week_start": "2011-11-21", "horizon_step": 2, "predicted_demand": 524.25 },
    { "week_start": "2011-11-28", "horizon_step": 3, "predicted_demand": 524.25 },
    { "week_start": "2011-12-05", "horizon_step": 4, "predicted_demand": 524.25 }
  ],
  "warnings": []
}
```

---

## 9. Operational Monitoring & Drift Detection

## 9. Operational Monitoring & Drift Detection

DemandCast provides an offline evaluation workflow in `src/demandcast/monitoring.py` and `scripts/evaluate_forecasts.py`:
- **WAPE & Bias Alerts**: Alerts triggered if aggregate WAPE $> 50\%$ or normalized relative bias exceeds $\pm 25\%$ ($\text{Relative Bias} = \frac{\sum (\hat{y} - y)}{\sum y}$).
- **Two-Sample Kolmogorov-Smirnov Test**: Serves as a descriptive marginal distribution diagnostic comparing forecast outputs against observed actual demand.
- **Residual Tracking**: Measures error distribution across percentiles, standard deviations, individual forecast horizons ($h \in \{1, 2, 3, 4\}$), and product series.
- **Input Data Drift**: Evaluates incoming series mean shifts against training baseline statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) to flag upstream data shifts before prediction.
- **Truthful Status**: Monitoring quality cannot be evaluated without ground-truth actuals; synthetic actuals are never fabricated.
- **Output Audit**: Stored in `reports/monitoring_report.json`.

---

## 10. Reproducibility & Commands Guide

### 10.1 Environment Setup (Windows PowerShell)
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 10.2 Run Full Data & Training Pipeline
```powershell
# 1. Download official UCI dataset and verify checksum
python scripts/download_data.py

# 2. Clean transactions, select products, and aggregate weekly demand
python scripts/prepare_data.py

# 3. Execute expanding-window backtesting across baselines and ML
python scripts/run_backtests.py

# 4. Train candidate model and package release v1.0.0
python scripts/train_model.py --version v1.0.0

# 5. Promote release to active model pointer
python scripts/promote_model.py --version v1.0.0

# 6. Run post-forecast monitoring and drift evaluation
python scripts/evaluate_forecasts.py

# 7. Verify live API endpoints
python scripts/verify_api_live.py
```

### 10.3 Run Automated Tests & Code Quality
```powershell
# Run Ruff linting
ruff check .

# Run Pytest suite (52 unit and integration tests, warnings treated as errors)
pytest -v
```

### 10.4 Inspect Experiments in MLflow
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Visit `http://localhost:5000` to inspect experiment parameters, validation backtests, model artifacts, and evaluation figures.

### 10.5 Local Container Deployment (Docker Compose)
```powershell
# Build and run API with mounted model releases
docker compose up --build
```
Interactive API documentation: `http://localhost:8000/docs`

---

## 11. Project Directory Structure

```
demandcast/
├── README.md                           # Master project documentation
├── pyproject.toml                      # Package build configuration & tool settings
├── requirements.txt                    # Pinned runtime and development dependencies
├── Dockerfile                          # Multi-stage production container definition
├── compose.yaml                        # Local container orchestration
├── .gitignore                          # Git exclusions (raw data, virtualenvs, mlruns)
├── .dockerignore                       # Docker build exclusions
├── configs/
│   └── config.yaml                     # Central system configuration
├── data/
│   ├── raw/                            # Raw UCI archive & extracted excel files
│   ├── interim/                        # Cleaned transactions (parquet)
│   └── processed/                      # Weekly demand parquet & selected_series.json
├── src/demandcast/
│   ├── __init__.py                     # Package init
│   ├── config.py                       # Pydantic configuration schemas & path resolution
│   ├── data_loading.py                 # UCI download, SHA-256 verification, Excel loader
│   ├── data_validation.py              # Auditable transaction cleaning & quality reporting
│   ├── aggregation.py                  # Monday weekly aggregation & training-only SKU selection
│   ├── features.py                     # Leakage-safe lags, shifted rolling stats, calendar
│   ├── baselines.py                    # Naive persistence, 4w moving average, 52w seasonal
│   ├── models.py                       # HistGradientBoostingRegressor recursive forecaster
│   ├── backtesting.py                  # Chronological expanding-window backtesting engine
│   ├── metrics.py                      # MAE, RMSE, WAPE, sMAPE, MASE, Forecast Bias
│   ├── train.py                        # Model training pipeline & MLflow logging
│   ├── evaluate.py                     # Benchmark comparison & visualization plotting
│   ├── tracking.py                     # MLflow experiment tracking helper
│   ├── registry.py                     # Versioned model packaging, promotion & rollback
│   ├── forecasting.py                  # Pre-inference validation & response formatting
│   ├── monitoring.py                   # Input quality checks, actuals reconciliation, drift
│   └── api/
│       ├── __init__.py
│       ├── main.py                     # FastAPI application factory & lifespan context
│       ├── routes.py                   # Routes (/health, /ready, /metadata, /forecast)
│       ├── schemas.py                  # Pydantic request & response models
│       └── dependencies.py             # Active model dependency injection container
├── scripts/
│   ├── download_data.py                # Download UCI archive
│   ├── prepare_data.py                 # Clean, select SKUs, aggregate weekly demand
│   ├── run_backtests.py                # Run expanding-window backtests
│   ├── train_model.py                  # Train and package model release
│   ├── evaluate_model.py               # Generate model comparison report
│   ├── promote_model.py                # Validate and promote release
│   ├── rollback_model.py               # Safe rollback to prior release
│   ├── evaluate_forecasts.py           # Offline reconciliation & drift evaluation
│   └── verify_api_live.py              # End-to-end live API validation test
├── reports/
│   ├── model_comparison.csv            # Empirical benchmark comparison table
│   ├── backtest_results.csv            # Detailed validation forecasts vs actuals
│   ├── test_horizon_results.csv        # Final held-out test predictions
│   ├── data_quality_report.json        # Auditable cleaning and exclusion counts
│   ├── series_summary.csv              # Time-series summary statistics per SKU
│   ├── monitoring_report.json          # Post-forecast reconciliation & alert metrics
│   └── figures/
│       ├── forecast_comparison.png     # Forecasts vs actuals over time
│       ├── metrics_by_horizon.png      # Error growth across weeks 1 to 4
│       └── residual_distribution.png   # ML model error distribution
├── models/
│   ├── releases/
│   │   └── v1.0.0/                     # Immutable release v1.0.0
│   │       ├── model.joblib            # Serialized pipeline & feature schema
│   │       └── metadata.json           # Model metadata, checksum, training range, metrics
│   └── current.json                    # Active promoted release pointer
├── tests/
│   ├── conftest.py                     # Fast, deterministic test fixtures
│   ├── test_data_validation.py         # Tests for cleaning rules & exclusions
│   ├── test_aggregation.py             # Tests for Monday aggregation & SKU selection
│   ├── test_features.py                # Tests for shifted rolling stats & lags
│   ├── test_backtesting.py             # Tests for expanding-window temporal integrity
│   ├── test_metrics.py                 # Tests for MAE, RMSE, WAPE, sMAPE, MASE
│   ├── test_training.py                # Tests for ML model fitting & recursive forecasting
│   ├── test_registry.py                # Tests for release packaging, checksum & tampering
│   ├── test_api.py                     # Tests for FastAPI endpoints & input validation
│   ├── test_monitoring.py              # Tests for drift detection & quality checks
│   └── test_rollback.py                # Tests for model promotion & rollback lifecycle
├── docs/
│   ├── architecture.md                 # System architecture & Mermaid sequence diagrams
│   ├── methodology.md                  # Forecasting problem formulation & backtest design
│   ├── monitoring.md                   # Operational monitoring & rollback protocol
│   └── model_card.md                   # Comprehensive model governance card
└── .github/
    └── workflows/
        └── ci.yml                      # GitHub Actions CI workflow (lint, test, build)
```

---

## 12. Known Limitations & Production Caveats

1. **Observed Sales Proxy**: The target is observed positive sales volume. In retail, unobserved stockouts suppress sales, meaning that historical zero-sales weeks do not prove zero customer demand.
2. **Dataset Age**: The dataset represents UK e-commerce giftware transactions from 2009–2011. While valuable for methodology demonstration, deploying this model in a modern retail setting requires fresh transactional and promotional data.
3. **Temporal Contract**: The standard `/forecast` endpoint rejects forecast origins predating the model training cutoff (`2011-11-07`) to prevent in-sample lookahead bias.
4. **History Depth Requirements**: Minimum history is 4 weeks for the active Moving Average forecaster, 13 weeks for the supervised ML candidate (or 53 weeks if seasonal lag 52 is enabled).
5. **Container Deployment Status**: The multi-stage `Dockerfile` and `compose.yaml` configurations are fully implemented with read-only release volume mounts; local container runtime execution could not be verified because the host Docker Desktop engine is currently stopped.
6. **Continuous Integration Status**: GitHub Actions CI workflow is configured in `.github/workflows/ci.yml` (linting, pytest suite, and docker build); execution has not yet run on a remote hosted runner because this local repository has no remote tracking branch configured.
7. **Git Repository Status**: The local repository has been initialized on branch `main` with the root commit tracking all verified production code, configs, tests, documentation, and release v1.0.0 artifacts. To connect to a remote repository when ready:
   ```powershell
   git remote add origin <your-repository-url>
   git push -u origin main
   ```
