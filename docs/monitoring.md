# DemandCast Production Monitoring & Quality Assurance

## 1. Monitoring Strategy Overview

Production demand forecasting requires continuous monitoring across two distinct operational boundaries:
1. **Input Data Quality Monitoring**: Pre-inference validation of incoming series structure, continuity, and value bounds.
2. **Post-Forecast Reconciliation Monitoring**: Offline evaluation of forecast accuracy, bias, and distribution drift as future weekly sales are finalized.

---

## 2. Pre-Inference Input Data Quality Checks

Implemented in `src/demandcast/monitoring.py` and enforced via FastAPI route middleware:

| Check | Failure Condition | Action |
| :--- | :--- | :--- |
| **Series Support** | `series_id` not in model's trained catalog | HTTP 400 Bad Request |
| **History Length** | Fewer than 13 weekly observations provided | HTTP 422 Unprocessable Entity |
| **Date Continuity** | Any step difference $\ne 7$ days (gaps or irregular intervals) | HTTP 400 Bad Request |
| **Date Order** | Non-chronological or duplicate timestamps | HTTP 400 Bad Request |
| **Value Boundaries** | Non-finite, NaN, or negative demand ($y < 0$) | HTTP 422 Unprocessable Entity |
| **Zero-Demand Streak** | 3 or more consecutive zero-demand weeks | Returns warning flag in payload |
| **Outlier Detection** | Observation $> \mu + 4\sigma$ of training distribution | Returns warning flag in payload |

---

## 3. Post-Forecast Actuals Reconciliation

Once a forecast horizon has passed and ground-truth sales are recorded, the offline monitoring job runs via:
```powershell
python scripts/evaluate_forecasts.py --forecasts reports/backtest_results.csv --actuals data/processed/weekly_demand.parquet
```

### 3.1 Error & Performance Metrics
- **Weighted Absolute Percentage Error (WAPE)**: Measures aggregate volume error:
  $$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$$
- **Forecast Bias**:
  - **Relative Bias (Normalized Ratio)**:
    $$\text{Relative Bias} = \frac{\sum (\hat{y} - y)}{\sum y}$$
    Dimensionless ratio directly comparable against configured alert threshold (default: $\pm 0.25$ or $\pm 25\%$).
    - Positive relative bias denotes systematic over-forecasting (leads to inventory glut).
    - Negative relative bias denotes systematic under-forecasting (leads to stockouts).
  - **Raw Mean Error (Bias)**:
    $$\text{Raw Bias} = \frac{1}{N} \sum (\hat{y} - y)$$
    Measured in forecast demand units (sales items).
- **Residual Tracking**: Full error distribution tracking (mean, standard deviation, median, min, max, 25th, and 75th percentiles).
- **MAE / RMSE / WAPE Growth by Horizon & Series**: Tracks error progression across weeks 1, 2, 3, and 4, as well as per-SKU performance.

### 3.2 Drift & Distribution Shifts
Monitoring strictly separates three distinct operational phenomena:
1. **Input Data Drift**: Incoming historical demand distributions compared against training reference statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) to flag shifts prior to inference.
2. **Forecast vs. Actual Marginal Distribution Divergence**: Evaluated via a two-sample Kolmogorov-Smirnov (KS) comparison. Because forecasts and actuals are paired by series and date, this is labeled and interpreted as a **descriptive marginal diagnostic**, not an independent-sample hypothesis test.
3. **Forecast Quality & Reconciliation Errors**: Evaluated post-hoc after actual demand is observed. Future forecast quality cannot be measured at inference time without ground truth; synthetic actuals are never fabricated.

---

## 4. Alert Thresholds & Investigation Triggers

DemandCast defines operational alert triggers in `configs/config.yaml`:

```yaml
monitoring:
  wape_alert_threshold: 0.50       # Alert if aggregate WAPE exceeds 50%
  bias_alert_threshold: 0.25       # Alert if normalized bias exceeds ±25%
  zero_demand_streak_warning: 3    # Warn if 3+ consecutive zeroes observed
  extreme_value_std_multiplier: 4.0# Warn if demand spikes > 4 standard deviations
```

### Action Protocols:
- **Low Severity (Streak / Outlier Warning)**: Forecasts are served with warning metadata attached to response payload; logs are marked for audit.
- **Medium Severity (Bias Alert)**: Triggers inspection of promotional activity, seasonal transitions, or out-of-stock events.
- **High Severity (WAPE > 50% or KS Drift)**: Initiates model review and triggers the automated retraining workflow or rollback.

---

## 5. Model Rollback Procedure

If a deployed release displays degraded performance in production, DemandCast supports non-destructive, atomic rollback:

```powershell
# Check current active version
curl http://localhost:8000/metadata

# Execute rollback to previously verified release
python scripts/rollback_model.py --to v1.0.0 --reason "Degraded performance observed in monitoring"

# Verify active status
curl http://localhost:8000/ready
```

The rollback command:
1. Validates the target release directory, model artifact, and SHA-256 checksum.
2. Atomically updates `models/current.json`.
3. The running service automatically serves the restored release.
