# DemandCast Forecasting Methodology

## 1. Problem Formulation & Business Objective

In retail demand forecasting, accurate short-to-medium-term replenishment planning is critical to minimizing both stockouts (lost sales and customer dissatisfaction) and overstocking (excess holding costs, depreciation, and working capital lockup).

The objective of **DemandCast** is to forecast weekly product demand for up to **four weeks ahead** ($h \in \{1, 2, 3, 4\}$) for selected retail product series.

### Key Definitions:
- **Entity**: Product SKU / Series Identifier (`series_id`).
- **Time Granularity**: Weekly, using a strict **Monday week-start** convention (`W-MON`).
- **Prediction Target**: Total eligible positive sales quantities for a product during each week.
- **Forecast Horizon**: 1 to 4 weeks forward from the forecast origin.
- **Inputs**: Historical weekly demand values, product identifier, and known calendar features.

---

## 2. Observed Sales vs. True Unconstrained Demand

A fundamental challenge in retail forecasting is the distinction between **observed sales** and **true unconstrained customer demand**:
- In the UCI Online Retail II dataset, inventory levels and out-of-stock events are unrecorded.
- When an item sells zero units in a week, this may indicate that customer demand was zero, OR that the product was completely out of stock and unavailable for purchase.
- Consequently, DemandCast explicitly treats positive sales quantities as an **observed sales proxy** for demand.
- We document this constraint to prevent false assumptions that the model captures unconstrained market appetite.

---

## 3. Product Selection Without Data Leakage

A common defect in machine learning projects is ranking or selecting products using full-dataset sales totals. Doing so leaks future commercial success into the model's training configuration.

### Selection Procedure:
1. **Cutoff Date**: Products are evaluated strictly on observations prior to `2010-10-01` (the initial training window).
2. **Criteria**:
   - Minimum active transaction weeks: $\ge 35$ weeks in the training period.
   - Minimum total units sold: $\ge 500$ units.
   - Top 15 product series ranked strictly by training-period volume.
3. **Outcome**: The selected product catalog is versioned in `data/processed/selected_series.json`. All evaluations, API schemas, and deployment validation gates enforce this catalog.

---

## 4. Leakage-Safe Feature Engineering

Features are generated such that at any target week $t$, only observations from weeks strictly prior to $t$ are accessible:

### 4.1 Historical Lags
- $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}, y_{t-8}, y_{t-12}$
- Seasonal lag $y_{t-52}$ (included only when historical depth $> 52$ weeks).

### 4.2 Shifted Rolling Statistics
To prevent target leakage, every rolling window is preceded by a 1-step lag:
$$\text{RollingMean}_k(t) = \frac{1}{k} \sum_{i=1}^k y_{t-i}$$
- Rolling Means: 4-week, 8-week, 12-week.
- Rolling Standard Deviation: 4-week trailing volatility.
- Zero-Demand Frequency: Proportion of weeks with $y = 0$ in the last 4 weeks.

### 4.3 Calendar & Cyclical Features
Known precisely in advance:
- $\text{Week of Year} \in [1, 52]$
- $\text{Month} \in [1, 12]$
- $\sin(2\pi \cdot \text{week} / 52.0)$ and $\cos(2\pi \cdot \text{week} / 52.0)$

---

## 5. Benchmarking & Baseline Models

To evaluate whether machine learning complexity is justified, three standard forecasting baselines are evaluated on identical backtest origins:

1. **Baseline A: Naive Persistence Forecaster**
   $$\hat{y}_{T+h} = y_T$$
   Repeats the most recent observed demand value across the entire horizon.

2. **Baseline B: 4-Week Moving Average**
   $$\hat{y}_{T+h} = \frac{1}{4} \sum_{i=0}^3 y_{T-i}$$
   Smooths short-term fluctuations using the trailing month average.

3. **Baseline C: Seasonal Naive Forecaster (52-week lag)**
   $$\hat{y}_{T+h} = y_{T+h-52}$$
   Projects demand from exactly one year prior. If 52 weeks of prior history is unavailable, records prediction as unavailable rather than fabricating zeroes.

---

## 6. Supervised ML Model Architecture

The primary machine learning candidate is scikit-learn's **`HistGradientBoostingRegressor`**:
- Efficient histogram binning ($O(N)$ training time).
- Categorical series feature encoding via `OrdinalEncoder` fitted only on training data.
- Hyperparameters:
  - `learning_rate`: 0.05
  - `max_iter`: 150
  - `max_leaf_nodes`: 31
  - `min_samples_leaf`: 5
  - `loss`: squared error

### Multi-Step Recursive Strategy
At each forecast step $h \in \{1, 2, 3, 4\}$:
1. Feature vector for target week $T+h$ is generated from historical observations up to $T$ plus already-predicted values for steps $1 \dots h-1$.
2. Model predicts point estimate $\hat{y}_{T+h}$.
3. Nonnegative postprocessing is applied: $\hat{y}_{T+h} = \max(0, \hat{y}_{T+h})$.
4. Prediction is appended to working history for step $h+1$.

---

## 7. Chronological Backtesting & Evaluation Design

**Random train/test splits are strictly prohibited** in time series forecasting because they destroy autocorrelation and leak future observations into past training rows.

### Expanding-Window Backtesting
- Three chronological validation origins spaced by 4-week intervals (`2011-08-15`, `2011-09-12`, `2011-10-10`).
- At each origin, the model is trained strictly on observations $\le \text{origin}$.
- Predictions are generated for the subsequent 4 weeks ($h=1, 2, 3, 4$), yielding 180 validation forecast observations.
- Forecasts are evaluated against ground truth only after all predictions have been issued.
- Validation origins remain strictly segregated prior to the final test period.

### Isolated Held-Out Test Horizon
- Cutoff origin: `2011-11-07`.
- **Test Horizon Integrity**: The raw UCI Online Retail II dataset abruptly terminates on Friday, December 9, 2011 at 12:50 PM. Under Monday-start Sunday-end weekly aggregation ($[Monday\ 00:00:00, Sunday\ 23:59:59.999999]$), the calendar week of `2011-12-05` has only 4.5 days of transactions. To prevent partial-week target distortion, this incomplete week is excluded from target evaluation.
- The held-out test evaluation is conducted over the **3 complete test weeks** (`2011-11-14`, `2011-11-21`, `2011-11-28`), yielding exactly **45 product-week test observations** across the 15 selected SKUs.
- Validation backtests guide all model selection decisions.
- The finalized pipeline is evaluated on the test horizon once, with zero post-hoc parameter adjustments.

---

## 8. Evaluation Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Weighted Absolute Percentage Error (WAPE)**:
   $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$$
4. **Symmetric Mean Absolute Percentage Error (sMAPE)**:
   $$\text{sMAPE} = \frac{100\%}{N} \sum \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$
5. **Mean Absolute Scaled Error (MASE)**:
   $$\text{MASE} = \frac{\text{MAE}}{\frac{1}{T-1} \sum_{t=2}^T |y_t - y_{t-1}|}$$
   Where the scaling denominator is computed on in-sample training history prior to the forecast origin. Denominators equal to zero (constant series) return `None` to prevent division by zero.
6. **Raw Forecast Bias (Mean Error in Demand Units)**:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
7. **Relative Forecast Bias (Dimensionless Ratio)**:
   $$\text{Relative Bias} = \frac{\sum (\hat{y}_i - y_i)}{\sum y_i}$$

---

## 9. Operational Quality Alert Policy & Retraining Criteria

### Quality Alert vs. Candidate Selection
- **Target Operational Threshold**: $\text{WAPE} \le 50.00\%$.
- **Validation Reality**: The best performing model, `Baseline_MovingAvg4`, achieves validation WAPE of $55.16\%$. Because it decisively outperforms all other candidates (HGB: $88.30\%$, Naive: $66.64\%$), it is promoted under an **audited operational override**.
- **Alert Persistence**: The alert is not hidden; the system reports `quality_status: "ALERT"` across all serving and monitoring interfaces.

### Model Freshness & Stale-Model Safeguards
- Training data vintage ends on `2011-11-07`.
- An operational freshness policy flags inference requests with origins $> 90$ days past cutoff with `StaleModelWarning`.
- Inputs beyond the cutoff represent retrospective simulation or backtesting, not verified modern market performance.

### Data Refresh Requirements
1. Continuous weekly transaction streams from modern retail POS/ERP systems.
2. Explicit Monday week-start interval mapping $[Monday\ 00:00:00, Sunday\ 23:59:59.999999]$.
3. Minimum 52 weeks history per series to capture annual seasonality.
4. Expanding-window chronological validation backtesting across at least 3 origins.
5. Achieving validation WAPE $\le 50.00\%$ to lift the operational quality alert without override.
