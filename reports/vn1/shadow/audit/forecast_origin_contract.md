# DemandCast VN1 Forecast-Origin Time Contract Specification

**Document Version**: 2.2.2
**Date**: 2026-10-25
**Governing Standard**: Strict Point-in-Time Point Availability and Weekly Interval Boundaries

---

## 1. Core Principles & Concept Separation

To prevent subtle temporal leakage, the DemandCast forecasting system explicitly separates calendar labels from data availability timestamps:

```
[Monday 00:00:00 .......................... Sunday 23:59:59.999999]  ---> Ingestion / Cutoff
|----------------- 7-Day Weekly Bucket ---------------------------|  ---> as_of_timestamp
^                                                                 ^
week_start                                                        Bucket End
```

| Concept | Formal Definition | Type / Representation |
| :--- | :--- | :--- |
| **`last_observed_week_start`** | The Monday `00:00:00` date of the latest **fully completed** historical 7-day accounting period. Observations with `week_start <= last_observed_week_start` are eligible for feature calculation and parameter estimation. | `pd.Timestamp` (Monday `YYYY-MM-DD`) |
| **`as_of_timestamp`** | The wall-clock evaluation timestamp (in UTC) when the forecast pipeline executes and verifies data eligibility. | ISO 8601 UTC String (`YYYY-MM-DDTHH:MM:SSZ`) |
| **`target_week_start`** | The start date of a forward planning period: `last_observed_week_start + pd.Timedelta(weeks=h)` for $h \in [1, \dots, H]$. Target weeks are strictly in the future relative to `last_observed_week_start`. | `pd.Timestamp` (Monday `YYYY-MM-DD`) |
| **`data_available_at`** | The system ingestion timestamp recording when a given weekly record was finalized by the upstream provider. | ISO 8601 UTC String |
| **`is_bucket_complete`** | Boolean status indicating whether the 7-day accounting interval is closed and complete. | Boolean (`True` / `False`) |

---

## 2. VN1 Weekly Sales Data Meaning

1. **Aggregation Window**: In the VN1 Retail Challenge dataset, each column header (`2020-07-06`, `2020-07-13`, etc.) denotes the **Monday start** of a completed 7-day accounting bucket ending on Sunday `23:59:59.999999`.
2. **Sales Values**: Values represent total weekly unit sales aggregated across physical register and replenishment transactions within that interval.
3. **No Intermediate Days**: Daily intra-week transactions are already aggregated into weekly totals.
4. **Zero Sales vs. Missing Data**: A zero sales value represents an observed period of zero customer demand. Series with no sales in an active catalog period remain zero.

---

## 3. Handling Incomplete Final Weeks

- **Historical Replay Mode**:
  - Replays use a deterministic period-based cutoff.
  - Rows with `week_start <= last_observed_week_start` are guaranteed to represent closed, completed historical weeks.
  - Rows with `week_start > last_observed_week_start` are excluded automatically before inference.
  - If a snapshot explicitly contains a row marked `is_bucket_complete == False`, it is purged to prevent partial-week volume distortion.
- **Prospective Live Mode (Fail-Closed)**:
  - If a live feed supplies an unclosed trailing week (where `as_of_timestamp` is prior to Sunday `23:59:59` of that week), the pipeline **fails closed** and returns `INCOMPLETE_INPUT_REJECTED`.
  - Live data without explicit availability timestamps or completion flags is rejected rather than silently assumed to be complete.

---

## 4. Leakage Prevention Guarantees

1. **Target Feature Leakage**: Target-week sales observations can never enter moving average windows or calibration estimators.
2. **Future Price Leakage**: Prices beyond `last_observed_week_start` are purged prior to calculating price ratios.
3. **Calibration Provenance**: Seasonal calibration multipliers are fitted strictly on completed historical transitions occurring strictly prior to `last_observed_week_start`.
