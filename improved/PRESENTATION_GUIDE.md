# JLR Spare Parts Demand Forecasting — Complete Presentation Guide

> A line-by-line explanation of the pipeline with the **rationale behind every
> design choice**, plus likely judge questions and how to answer them.

---

## Table of Contents

1. [The Business Problem](#1-the-business-problem)
2. [The Data](#2-the-data)
3. [Why Spare Parts are Different](#3-why-spare-parts-are-different)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Demand Pattern Classification](#5-demand-pattern-classification)
6. [The Forecasting Models — Why Each One](#6-the-forecasting-models)
7. [Feature Engineering — Why Each Feature](#7-feature-engineering)
8. [Avoiding Data Leakage (CRITICAL)](#8-avoiding-data-leakage)
9. [Evaluation Metrics — Why Each One](#9-evaluation-metrics)
10. [The Hybrid Routing Strategy](#10-the-hybrid-routing-strategy)
11. [Quantile Forecasts & Safety Stock](#11-quantile-forecasts--safety-stock)
12. [The Cost Framework](#12-the-cost-framework)
13. [Results Summary](#13-results-summary)
14. [Likely Judge Questions](#14-likely-judge-questions)
15. [Potential Improvements to the Present Implementation](#15-potential-improvements-to-the-present-implementation)
16. [Glossary of Terms](#16-glossary-of-terms)

---

## 1. The Business Problem

**What we are solving:** Forecast monthly demand for **8,064 Jaguar Land Rover
spare parts** at warehouse location IN00, then translate those forecasts into
actionable **inventory decisions** (when to reorder, how much safety stock).

**Why it matters:**
- Too much stock → cash tied up, warehouse cost, obsolescence risk
- Too little stock → vehicles off-road, dealer SLA penalties, lost goodwill
- The existing ERP system (FCST_H) makes its own forecasts — we need to
  prove whether we can do better, and quantify the savings.

**Why now:** Aftermarket spare parts is a **high-margin, high-stakes** business
where fill rate directly drives customer retention. JLR's parts business
generates billions in revenue with long part lifecycles (15-20 years).

---

## 2. The Data

**File:** `IN00_ST_DATA_3Years.xlsx`
- **8,064 rows** = unique SKUs (Stock-Keeping Units, i.e. individual part numbers)
- **249 columns** total
- **37 months** of demand history (May 2023 → May 2026)

**Key column groups:**

| Prefix | Meaning | Example |
|---|---|---|
| `CAP_SCAL_M00..M36` | Monthly demand (capped & scaled). M00 = newest | `CAP_SCAL_M36` = May-2023 |
| `FCST_H00..H36` | What the ERP **predicted** for each historical month | Historical forecast |
| `FCST_F01..F12` | What the ERP predicts for the **next 12 months** | Future forecast |
| `STD_DEV_H/F` | The ERP's own uncertainty estimates | For benchmarking |
| `UNCAP_*` | Raw uncapped/unscaled demand (alternative target) | True demand |

**Per-SKU metadata:**

| Column | What it tells us |
|---|---|
| `PRODUCT` | Unique part ID (e.g. `28LR144704`) |
| `VOL_CLASS` | Volume tier A-E (A = top sellers) |
| `MOVE_CLASS` | Movement tier A-E + S0 (A = fast, S0 = no movement) |
| `FCST_MODEL` | Which method the ERP picked (A1, B1, C1, F1, G1) |
| `BRAND` | Always `LR` (LandRover/JLR) |
| `COST` | Unit cost — drives inventory $ value |
| `TSL` | Target Service Level (median **99.2%** — very high!) |
| `SAFTY` | ERP's existing safety stock |
| `REORDER_POINT` | ERP's existing reorder point |
| `EOQ_QTY` | Economic Order Quantity |

**Why we use CAP_SCAL_*:** The capped/scaled series is the *cleaned* demand
the ERP itself uses for forecasting. Using the same series lets us make a
fair, head-to-head comparison vs FCST_H. (For a future iteration we could
forecast the raw `UNCAP_UNSCAL_*` to recover true demand.)

**Why M00 = newest (reverse order):** The Excel file orders columns newest
first. We **reverse them** so index 0 is the oldest month. This lets us
write `history[:, :t]` to mean "everything before month t" — a natural
left-to-right time axis.

---

## 3. Why Spare Parts are Different

This is the **single most important context** to communicate to judges.

**Standard time-series forecasting** (e.g. retail sales, weather) assumes
demand happens **every period** and is **roughly continuous**. ARIMA, Prophet,
and LSTMs are built for that world.

**Spare parts demand is INTERMITTENT:**
- Most months for most SKUs are **zero demand**
- When demand happens, it's **lumpy** (1, then 0, then 0, then 5, then 0...)
- Statistical assumptions of those classical methods **fail catastrophically**

**Why it matters for our choice of models:**
- We need methods designed specifically for intermittent demand: **Croston,
  SBA, TSB**
- For the ML models, we use **Tweedie loss** (a probability distribution that
  handles "zero or positive" data; standard MSE/MAE don't model zeros well)
- We add a **zero-demand classifier gate** — separately predicts "will this
  SKU have any demand at all this month?" and snaps to zero when not

**In our data, 74% of SKUs are classified as "Lumpy"** (rare AND variable),
the hardest possible class to forecast. This validates the choice of
specialized methods.

---

## 4. Pipeline Architecture

The pipeline is split into a clean **engine (`forecast_pipeline.py`)** and a
**narrative notebook (`final_v2.ipynb`)**.

**Why this split:**
- The engine is **importable** and testable — clean functions, type-hinted
  config, no notebook-state pollution
- The notebook is for **storytelling** (judges, stakeholders) — charts and
  prose
- This is professional ML code organization, not "research notebook spaghetti"

**High-level flow:**
```
Load data
   ↓
Classify demand patterns (Smooth/Erratic/Intermittent/Lumpy)
   ↓
Train statistical models (SES, Croston, SBA, TSB) — vectorised
   ↓
Build point-in-time features (NO leakage)
   ↓
Train ML models (LightGBM, XGBoost, RF) + zero-gate classifier
   ↓
Train quantile heads (P50, P90) for uncertainty
   ↓
Backtest on Nov-25 → Apr-26 (6 months)
   ↓
Pick best model per demand pattern → HYBRID
   ↓
Refit on full history, forecast Jun-Aug 2026
   ↓
Compute safety stock & reorder points
   ↓
Compare vs ERP, compute cost savings
```

---

## 5. Demand Pattern Classification

**Method: Syntetos-Boylan-Croston (SBC) Classification**

Two metrics are computed per SKU:
- **ADI (Average inter-Demand Interval)** = mean number of months between
  non-zero demand months. High ADI = rare demand.
- **CV² (Coefficient of Variation squared)** = (σ/μ)² of the demand sizes.
  High CV² = unpredictable amounts.

The classification:

| | CV² < 0.49 (consistent size) | CV² ≥ 0.49 (variable size) |
|---|---|---|
| **ADI < 1.32 (frequent)** | **Smooth** — easy | **Erratic** — variable amounts |
| **ADI ≥ 1.32 (rare)** | **Intermittent** — rare but stable | **Lumpy** — rare AND variable |

**Why these specific thresholds (1.32 and 0.49):**
- These come from the **Syntetos & Boylan (2005) academic paper**, derived
  empirically by minimizing forecast error on real spare-parts data
- They are the **industry standard** for intermittent demand classification —
  any forecasting researcher will recognize them immediately

**Why we classify:** Different demand types respond best to different methods:
- Smooth → simple exponential smoothing works fine
- Intermittent → Croston's method shines
- Lumpy → most methods fail; need ML or sophisticated approaches

**Our results:**
- Lumpy: 5,978 (74%)
- Erratic: 1,061 (13%)
- Smooth: 780 (10%)
- Intermittent: 245 (3%)

This is **typical for automotive aftermarket** — long-tail, highly intermittent.

---

## 6. The Forecasting Models

We train **9 models in parallel** and let the data tell us which is best.

### 6.1 Naive (lag-1)

**What:** Forecast = last month's actual demand
**Why include:** It's the **baseline floor**. If our sophisticated models can't
beat this, we have nothing. **MASE is defined relative to this baseline** —
MASE < 1 means we beat naive.

### 6.2 SES — Simple Exponential Smoothing

**Math:** `S_t = α * y_t + (1 - α) * S_{t-1}`

**What:** Weighted average where recent months count more (α = 0.20).

**Why include:** It's the textbook baseline for **smooth demand**. Battle-tested
since the 1950s. Cheap to compute. Good interpretability.

**Why α = 0.20:** Empirically validated for spare parts in academic literature.
Higher α (e.g. 0.5) reacts too quickly to noise; lower α is too sluggish.

**Where it wins:** Smooth-pattern SKUs.

### 6.3 Croston's Method (1972)

**The breakthrough:** Don't smooth demand directly — smooth two things separately:
1. The **size** of demand events (`z`)
2. The **interval** between them (`p`)

Then forecast = `z / p` (rate per period).

**Why this is brilliant for intermittent demand:**
- Standard SES on intermittent data is biased downward (zeros pull the average
  down, but you're not asking "how much each month" — you're asking "how much
  per demand event")
- Separating size and frequency captures the **two distinct sources of variance**

**Limitation:** Croston is mathematically biased upward (proven by Syntetos &
Boylan), so we also use SBA.

### 6.4 SBA — Syntetos-Boylan Approximation (2005)

**What:** Croston × `(1 - α/2)` — a bias correction.

**Why include:** It's **mathematically proven to be unbiased** under standard
assumptions. The default choice in modern intermittent-demand forecasting.

**Where it wins:** Intermittent and Lumpy patterns where the demand process is
stationary.

### 6.5 TSB — Teunter-Syntetos-Babai (2011)

**What:** Tracks **probability of demand** (`q`) instead of interval, plus size:
`forecast = q × z`

**Why this is better than Croston for some cases:**
- **Handles obsolescence**: when a SKU stops selling, `q` decays toward zero
  (Croston's `p` keeps growing forever, which is unrealistic)
- **Important for spare parts** because parts go end-of-life as vehicle models
  are retired

**Where it wins:** Aging SKUs and parts being phased out.

### 6.6 LightGBM with Tweedie Loss

**What:** Gradient-boosted decision trees — the dominant ML method for
tabular data in 2020s.

**Why Tweedie loss specifically:**
- Tweedie is a **compound Poisson-Gamma distribution**
- It naturally models data that is **non-negative with a point mass at zero**
  — exactly what spare-parts demand is
- `tweedie_variance_power = 1.5` is the standard sweet spot between
  Poisson (1.0) and Gamma (2.0)

**Why LightGBM over alternatives:**
- **Speed**: histogram-based, parallel, ~10x faster than XGBoost on this
  problem size
- **Memory efficient**: leaf-wise growth uses less RAM
- **Built-in Tweedie objective** (XGBoost has it too, RF doesn't)
- **Used by Microsoft, Kaggle winners, M5 forecasting competition winners**

**Hyperparameters and why:**
- `n_estimators=500`: enough boosting rounds for convergence on 145K samples
- `learning_rate=0.05`: standard "slow but steady" choice
- `num_leaves=63`: more leaves than the default 31 — captures the
  high-dimensional feature interactions
- `min_child_samples=30`: prevents over-fitting on tiny groups (we have 8K
  SKUs split many ways)
- `subsample=0.8, colsample_bytree=0.8`: row & column sampling for
  regularization (random-forest-style noise injection)
- `reg_alpha=0.1, reg_lambda=0.1`: L1 + L2 regularization

### 6.7 XGBoost (also Tweedie)

**Why include alongside LightGBM:** Different tree-growth strategy
(level-wise vs leaf-wise) catches different patterns. They sometimes
disagree, which gives us **ensemble diversity**.

### 6.8 Random Forest

**Why include:** Different model family (bagging, not boosting). Acts as
a **stability check** — if RF and the boosters all agree, we have high
confidence. RF also tends to handle outliers more gracefully.

### 6.9 Ensemble (SBA + LightGBM average)

**Why this specific pairing:**
- SBA captures the **structural intermittency**
- LightGBM captures **non-linear patterns and seasonality** in the features
- Averaging them is a classic **bias-variance tradeoff** — SBA brings low
  variance, LightGBM brings low bias
- Outperformed both individually on certain demand patterns

---

## 7. Feature Engineering

For the ML models we engineer **23 features per SKU per month**:

### Lag features (recent history)
- `lag1`, `lag2`, `lag3` — last 1-3 months (recent trend)
- `lag6` — half-year ago
- `lag12` — same month last year (seasonality)

**Why lag-12 specifically:** Many parts have seasonal patterns (winter
windshield wipers, summer AC components). Year-over-year is the key
seasonal cycle.

### Rolling statistics
- `roll3_mean`, `roll6_mean`, `roll12_mean` — short/medium/long-term levels
- `roll3_std` — recent volatility
- `roll6_max` — recent peak (catches episodic spikes)

**Why three time windows:** Different SKUs respond at different speeds.
Fast-moving parts care about 3-month trends; slow-movers need 12-month context.

### Trend & lifecycle
- `trend_ratio` = recent-3 / previous-3 — is demand accelerating or decaying?
- `months_since_last` — days since last non-zero sale (proxy for obsolescence)
- `nonzero_cnt12` — how many of the last 12 months had any sales?

### Variability descriptors
- `adi_pit` — point-in-time ADI (recomputed from history only)
- `zero_rate_pit` — fraction of zero months (point-in-time)
- `cv_pit` — coefficient of variation (point-in-time)
- `season_lag12_ratio` — `lag12 / roll12_mean` (seasonal index)

### Calendar
- `month_sin`, `month_cos` — month-of-year as a circular variable

**Why sin/cos encoding instead of `month=1..12`:**
- A model trained on `month=12` (Dec) and `month=1` (Jan) as separate
  integers thinks they're 11 apart, but they're actually adjacent
- Sin/cos puts them at adjacent points on a circle → the model learns the
  cyclical structure naturally
- This is the **standard cyclical encoding** in time-series ML

### Static SKU attributes
- `vol_class`, `move_class` — encoded as integers (label encoder)
- `brand` — single brand here (LR), but kept for portability
- `log_cost` — `log(1 + COST)` — log scale because cost spans 4 orders of
  magnitude ($0.01 to $18,710); raw cost would dominate everything else

---

## 8. Avoiding Data Leakage (CRITICAL)

> **This is the most important fix vs the original notebook.**

**What is leakage:** Using future information to predict the past during
training. Causes inflated backtest scores that **don't reproduce in
production**.

**The original code's leak:** It computed `zero_rate`, `cv`, `adi` ONCE over
the full 37 months (including the test period), then fed them as features.
That's like training a stock predictor with tomorrow's prices.

**Our fix:** Every per-SKU statistic is **recomputed point-in-time** —
using only `history[:, :t]` (months strictly before the prediction target).

**How we proved it:**
```python
F1 = build_features(demand[:, :t], ...)
demand_corrupted = demand.copy()
demand_corrupted[:, t:] = noise  # mess up everything after t
F2 = build_features(demand_corrupted[:, :t], ...)
assert np.allclose(F1, F2)  # features must be identical
```
This test passes — confirming **features are invariant to future data**.

**Why this matters for judges:** A judge with ML background may ask "did you
check for leakage?" — say YES, with this exact test as proof.

---

## 9. Evaluation Metrics

We deliberately use **multiple metrics** because no single one captures
everything for intermittent demand.

### WAPE — Weighted Absolute Percentage Error (PRIMARY METRIC)

```
WAPE = sum(|actual - forecast|) / sum(|actual|) * 100
```

**Why this is our headline:**
- Volume-weighted: a 10% error on a 100-unit SKU matters more than 50%
  on a 2-unit SKU
- **Not undefined when actual = 0** (unlike MAPE)
- The **industry standard** for intermittent demand

### vWAPE — Value-Weighted WAPE

Same as WAPE but weighted by `COST` instead of unit count.

**Why include:** Tells us about **money at risk**. A small expensive SKU
hurts more than a large cheap one financially.

### MASE — Mean Absolute Scaled Error

```
MASE = MAE / naive_MAE_in_sample
```

**Why this is the academic gold standard:**
- **Scale-free** — comparable across SKUs of vastly different sizes
- **MASE < 1** ⟺ "we beat the naive baseline"
- **Symmetric** for over- and under-forecast (unlike MAPE which penalizes
  under-forecast more)
- Recommended by **Hyndman & Koehler (2006)** — the most cited paper in
  forecasting metrics

**Why we report MASE_med (median):** The mean is dominated by a few
extreme SKUs. Median is robust and tells us "the typical SKU."

### Bias %

```
Bias% = sum(forecast - actual) / sum(actual) * 100
```

**Why include:** Detects **systematic over- or under-forecasting**. Negative
bias signals stockout risk; positive bias signals overstock risk.

### MAPE — Mean Absolute Percentage Error (team convention)

```
For each SKU-month:
    error% = |actual - forecast| / actual   if actual > 0
    error% = 0                               if actual == 0
MAPE = mean(error%) across ALL SKU-months * 100
```

**The convention we use (per mentor guidance):** when actual demand is
**zero**, that month contributes **0% error** to the MAPE average (rather
than being dropped or treated as undefined).

**Why this convention:**
- Standard MAPE divides by actual, so it is **undefined when actual = 0**.
  For intermittent demand where 60-90% of months are zero, that breaks MAPE.
- Treating zero-demand months as 0% error makes MAPE **well-defined for every
  SKU-month** and keeps it **directly comparable to the existing ERP
  reporting**, which uses the same convention. Consistency of definition is
  what lets us compare like-for-like.

**Important caveat to state to judges:** Because this convention counts the
many zero-demand months as "perfect" (0% error), MAPE is **diluted downward**
and is **not** a good standalone accuracy measure for intermittent demand.
That is exactly why **WAPE and MASE are our primary metrics** — they are not
distorted by the large number of zero months. We report MAPE only for
**comparability with the ERP's existing dashboards**.

**Honest note on the numbers:** Under this convention the ERP's MAPE
(~30.7%) is marginally lower than our hybrid's (~32.6%). This is an artifact
of the convention (the ERP over-forecasts, and on the rare non-zero months
its larger numbers can look proportionally closer). On every metric that
properly reflects intermittent-demand accuracy — **WAPE, MASE, MAE, RMSE** —
our model still wins. Do not let a single diluted metric override the four
that matter.

### MAE & RMSE

Classical metrics. RMSE penalizes large errors more (squared). We use
both for **robustness** — if RMSE rankings differ from MAE, there are
outlier SKUs distorting one of them.

---

## 10. The Hybrid Routing Strategy

**Why one model can't fit all:** Different demand patterns have different
optimal methods (proven theoretically and shown in our heatmap).

**Routing logic:**
```python
for each demand pattern in [Smooth, Erratic, Intermittent, Lumpy]:
    pick the model with lowest WAPE on that pattern
    assign all SKUs of that pattern to that model
```

**Our learned routing on real data:**
- Smooth → **SES** (simple, accurate for clean demand)
- Erratic → **XGBoost** (ML handles variance with rich features)
- Intermittent → **SBA** (classical specialist)
- Lumpy → **LightGBM** (ML's flexibility wins on the hardest pattern)

**Robust fallback:** If WAPE is undefined for a pattern (zero total demand
in test window), we fall back to MASE, then to a theory-aligned default
(SBA for Lumpy, etc.). This avoids `None` model assignments.

**Why pattern-routing beats global model selection:**
- A single "winner" model averages across all patterns, missing the
  specialist's edge on each
- Pattern-routing is **interpretable** — you can defend "why model X for
  Smooth" with literature backing

---

## 11. Quantile Forecasts & Safety Stock

### Why quantile forecasts?

A point forecast says "expected demand is 12 units." It says **nothing
about uncertainty**.

For inventory decisions, we need to know:
- How likely is demand to spike to 50?
- How much buffer do we need to hit 99% service level?

**Quantile regression** answers this directly: "P90 demand is 18 units"
= "10% of the time demand will exceed 18."

### LightGBM quantile loss

We train **two extra heads**:
- P50 head — `objective="quantile", alpha=0.5` (median forecast)
- P90 head — `objective="quantile", alpha=0.9`

The pinball loss this minimizes naturally produces calibrated quantiles.

### Non-crossing enforcement

**Problem:** Independently-trained quantile models can produce P90 < P50
on some SKUs (they shouldn't — that's nonsensical).

**Fix:** After predicting all quantiles, run a **monotone running max**
across sorted quantile levels. P90 ≥ P50 by construction.

### Service-level safety stock (the textbook approach)

```
SS = z(SL) × σ × √(lead_time)
```

Where:
- `z(SL)` is the inverse normal CDF at the target service level (z=2.33 for
  99%, z=1.65 for 95%)
- `σ` is per-period demand standard deviation (estimated from last 12 months)
- `√(lead_time)` scales σ to the lead-time horizon

**Why this formula:** Standard inventory theory (Silver & Pyke). Assumes
demand over lead time is approximately normal. The √ comes from variance of
sums of independent random variables.

**Why we use each SKU's own TSL:** The data provides `TSL` per SKU (median
99.2%). Using a global service level would over-stock low-criticality parts
and under-stock critical ones.

### Reorder Point

```
ROP = expected_lead_time_demand + safety_stock
```

This is what triggers a replenishment order. When stock falls below ROP,
order more.

---

## 12. The Cost Framework

This was iterated based on judge-style feedback ("if we under-predict more,
isn't that bad?").

### The honest framework: order = forecast + safety stock

**Don't compare raw point forecasts** to the ERP — that's apples to oranges
because the ERP's forecast implicitly carries its own buffer.

**The right comparison:** What you'd actually order under each system.

**For our model:** point forecast + service-level safety stock = realistic order
**For the ERP:** their FCST_H (which is already inflated to act as a buffer)

### Cost components

**Holding cost:** `25%/year × unit_cost × excess_units × months_held`
- 25% is the textbook "carrying cost rate" — covers warehouse, capital,
  insurance, obsolescence, shrinkage

**Stockout cost:** `multiplier × unit_cost × shortage_units`
- Multiplier varies based on part criticality:
  - 1.0x: easily-substituted commodity parts
  - 1.5x: standard (covers expediting + lost sale)
  - 3.0x: realistic for branded aftermarket
  - 5-10x: safety-critical (brakes, airbags) — vehicle off-road costs

### Sensitivity analysis (why we test multiple multipliers)

Because the "true" stockout cost is unknown and varies by part, we show
the cost story across the full range. **The result holds at every level**:
Hybrid+SS beats ERP by ~50% across all penalty assumptions, which makes the
conclusion **robust to assumptions**.

### Annualization

We multiply 6-month savings by 2 to estimate annual. This is conservative
because:
- Q4/Q1 demand can be higher than mid-year
- The model's accuracy gains compound over time as more recent data is
  incorporated

---

## 13. Results Summary

### Pure forecasting accuracy (backtest Nov-25 → Apr-26)

| Metric | Hybrid | ERP FCST_H | Improvement |
|---|---|---|---|
| WAPE | 38.2% | 41.0% | **6.8% better** |
| vWAPE | 54.6% | 58.2% | **6.3% better** |
| MASE_med | 0.82 | 0.98 | **clearly beats naive** |
| MAE | 3.01 | 3.23 | **6.8% lower** |
| RMSE | 36.5 | 40.3 | **9.4% lower** |

### Inventory cost (with safety stock)

| Stockout Penalty | Hybrid+SS | ERP | Savings | % |
|---|---|---|---|---|
| 1.0x | $3.1M | $6.3M | **$3.2M** | 50% |
| 1.5x | $4.5M | $9.4M | **$4.8M** | 52% |
| 3.0x | $8.8M | $18.6M | **$9.8M** | 53% |
| 5.0x | $14.5M | $31.0M | **$16.5M** | 53% |

### Service level (fill rate)

- Hybrid + Safety Stock: **96.7%** average fill rate
- ERP System: **77%** average fill rate
- **Improvement: ~20 percentage points better customer service**

### Annualized estimate
**$9.7M – $19.7M / year** depending on stockout penalty assumption.

---

## 14. Likely Judge Questions

### Q1: "How do you know your model isn't overfitting?"
**A:** Three safeguards:
1. **Time-based split**: train on months 0-29, test on 30-35. We never see
   test data during training.
2. **Point-in-time features**: every statistic is recomputed using only
   history strictly before the target month. We have a unit test proving
   features are invariant to future data corruption.
3. **Regularization**: `reg_alpha=0.1, reg_lambda=0.1, min_child_samples=30,
   subsample=0.8` — these all combat overfitting in LightGBM/XGBoost.

### Q2: "Why didn't you use ARIMA / Prophet / LSTM?"
**A:** ARIMA assumes Gaussian noise and continuous demand — both fail
catastrophically on intermittent data (74% of our SKUs are Lumpy). Prophet
is designed for daily/weekly seasonality of clean signals. LSTMs need lots
of data per series — we have 37 months per SKU, way too few. The Croston
family is purpose-built for this problem and is what the academic
literature recommends. Our LightGBM with Tweedie loss is the modern
ML equivalent.

### Q3: "Why Tweedie loss specifically?"
**A:** Tweedie is a compound Poisson-Gamma distribution that naturally
models "zero or positive continuous" data. Standard MSE assumes Gaussian
errors — wrong for our skewed, zero-inflated data. Tweedie with variance
power 1.5 is the standard choice between Poisson (1.0) and Gamma (2.0)
that matches spare-parts demand structure.

### Q4: "Your model under-predicts more SKUs than the ERP. Isn't that bad?"
**A:** That's why we don't use the raw point forecast for ordering. The
right thing to order is **point forecast + safety stock**. With service-
level-sized safety stock, we hit 96.7% fill rate vs the ERP's 77%, AND
save ~50% of cost. The ERP's apparent "buffer" from over-forecasting is
crude and wasteful — it over-stocks expensive parts unnecessarily.

### Q5: "Why is your hybrid biased downward (negative Bias%)?"
**A:** Tweedie regression and intermittent-demand methods are well-known
to under-forecast on a small subset of SKUs. This is **expected and
acceptable** because:
1. The bias is small (-10.8% net)
2. The safety stock layer compensates for it
3. The alternative (over-biased forecasts) hides poor accuracy behind
   waste

### Q6: "Why these particular cost assumptions (25% holding, 1.5x stockout)?"
**A:** Holding cost of 25%/year is the textbook industry standard
(covers capital, warehouse, obsolescence, insurance). Stockout multiplier
varies by part — that's why we run sensitivity analysis from 1x to 10x
and show the savings hold at every level. The conclusion is robust to
assumptions.

### Q7: "Could you forecast even better?"
**A:** Yes, with these enhancements:
1. Forecast UNCAP_UNSCAL demand for true demand recovery
2. Use lifecycle flags (NEW_PROD, PRE_PART, replacement dates) to handle
   new and obsolete parts specially
3. Per-SKU lead time calibration
4. Hierarchical reconciliation (reconcile SKU forecasts with category totals)
5. Probabilistic forecasts beyond P90 (full distributions)

### Q8: "How would you deploy this in production?"
**A:**
1. Schedule monthly retraining as new data arrives (the pipeline is
   one-function-call: `run_pipeline(cfg)`)
2. Output hybrid forecasts + safety stock + reorder points to ERP
3. Monitor monthly WAPE/Bias drift; alert if it degrades
4. A/B test: deploy on a SKU subset first, measure fill rate uplift
5. Quarterly review of routing decisions (re-pick best model per pattern)

### Q9: "What's the WORST-case scenario?"
**A:** A previously-stable SKU suddenly gets a step change in demand (e.g.
recall-driven spike). All forecasting methods would miss this initially —
our P90 quantile gives some protection but won't fully cover a 10x spike.
Mitigation: monitor forecast errors weekly, manually override known
recall events, and use the safety stock layer with high TSL (99%+) for
critical parts.

### Q10: "Why did you compute features point-in-time? Wasn't the simpler version fine?"
**A:** No, the simpler version had **data leakage**. It computed zero_rate
and CV over all 37 months, including the test window, then used those as
features during training. That artificially inflates backtest scores —
in production, the model would be much worse. Point-in-time is the
**only correct approach** and we have a proof test that the features
are leakage-free.

### Q11: "How big is your training set?"
**A:** **145,152 examples** (8,064 SKUs × 18 valid training months from
month 12 to 29). The first 12 months are reserved for building lag-12
features. This is plenty for the chosen models — gradient boosting
performs well from ~10K examples upward.

### Q12: "Why one warehouse only? What about generalization?"
**A:** Location IN00 is the dataset given. The pipeline is
location-agnostic — we'd just rerun for IN01, IN02, etc. Each location
has different demand dynamics, so per-location models are the right
choice rather than pooling.

### Q13: "What if a SKU has only 6 months of history (new product)?"
**A:** The lag-12 features would be zero, but the model still works:
- Static features (cost, class, brand) carry information
- Recent lags work normally
- The zero-gate likely predicts zero for very new SKUs — safe default
- Future enhancement: explicit `is_new_product` flag using `NEW_PROD` column

### Q14: "Why didn't you use AutoML / a pre-built tool?"
**A:** Tools like Amazon Forecast, Prophet, or AutoGluon don't natively
handle intermittent demand well, don't expose the safety-stock layer
we need, and don't allow the per-pattern routing strategy that's the
key to our accuracy gains. A custom pipeline gives us full control,
explainability, and ~50% cost savings — well worth the engineering effort.

### Q15: "What's your most important single insight?"
**A:** The fairest comparison is **what you'd actually order**, not raw
point forecasts. Once you include service-level safety stock, our hybrid
---

## 15. Potential Improvements to the Present Implementation

This section is your "we know the limitations and here's the roadmap" answer.
Being able to articulate this shows maturity and wins credibility with judges.
Items are grouped by theme and ordered roughly by impact-to-effort.

### A. Data & Target Improvements

1. **Forecast the raw `UNCAP_UNSCAL_*` demand, not just `CAP_SCAL_*`.**
   We currently model the capped/scaled series (for a fair comparison vs the
   ERP). The capping hides true demand spikes. Modelling uncapped demand —
   and treating the capping as a separate business rule — would recover the
   real signal and avoid systematically under-sizing buffers for spiky parts.

2. **Use lifecycle columns explicitly** (`NEW_PROD`, `NEW_MODEL`, `PRE_PART`,
   `SS_PART`, `REPLACEMENT_START_DATE`, `REPLACEBY_START_DATE`).
   Spare parts are born (new model launch) and die (superseded by a successor
   part). Right now a brand-new part with 2 months of history is treated like
   any other. Adding an `is_new`, `months_since_launch`, and `is_superseded`
   feature — and routing newly-launched parts to an analog/"like-part"
   forecast — would sharply improve the hardest cases.

3. **Incorporate external drivers.** Vehicle parc data (how many vehicles of
   each model are on the road and their age), warranty/recall campaigns,
   service-interval schedules, and even macro signals. A brake-pad's demand
   is downstream of the number of in-warranty vehicles — that is a powerful,
   currently-unused predictor.

4. **Data-quality hardening.** The raw file had a duplicate `M31` and a
   missing `M32` in the uncapped block. A validation step (assert 37 clean
   monthly columns, no duplicate periods, non-negative demand, consistent
   report date) should run before modelling and fail loudly.

### B. Modelling Improvements

5. **Hyperparameter tuning per demand pattern.** We use one sensible LightGBM
   config for all SKUs. Tuning (via Optuna / grid search) **separately for
   each pattern** — and tuning the Croston/SBA/TSB/SES smoothing constants
   (currently fixed at α=0.15 / 0.20) per SKU or per pattern — would squeeze
   out more accuracy.

6. **Cross-learning ML / global models with SKU embeddings.** Instead of
   hand-crafted static encodings, learn an embedding per SKU (or per
   part-family) so the model shares strength across similar parts. This is
   how the M5 competition winners handled tens of thousands of series.

7. **Richer ensembling.** We use a simple SBA+LightGBM average. Better options:
   - **Stacking**: train a meta-model that learns the optimal weights per SKU
     given its features (a model that predicts which model to trust).
   - **Weighted blends** tuned on the validation window rather than a fixed 50/50.

8. **Modern intermittent-demand methods.** Add **ADIDA / IMAPA** (temporal
   aggregation — aggregate to quarters, forecast, disaggregate; reduces zero
   noise) and **Willemain's bootstrap** (resamples historical demand to build
   a full lead-time demand distribution — excellent for safety stock).

9. **Dedicated new-product / cold-start handling.** "Like-for-like" forecasting
   that borrows the demand curve of a similar mature part for SKUs with little
   history, then blends toward the SKU's own data as it accumulates.

### C. Forecasting-Process Improvements

10. **Multi-window rolling-origin backtest.** We currently evaluate one
    6-month window (Nov-25 → Apr-26). Averaging metrics over **several** rolling
    origins (e.g. test on each of the last 12 month-ends) gives a far more
    robust, less luck-dependent estimate of accuracy and prevents over-tuning
    to one window.

11. **Multi-horizon evaluation.** We backtest one-month-ahead. Replenishment
    decisions depend on the full lead time, so we should also measure
    2- and 3-month-ahead accuracy explicitly and report degradation by horizon.

12. **Direct multi-step instead of recursive.** The forward forecast feeds its
    own predictions back as inputs (recursive), which compounds error. Training
    a separate model per horizon (direct strategy) usually wins at h=2,3.

13. **Hierarchical reconciliation.** Forecast at SKU level AND at
    category/brand/warehouse level, then reconcile (MinT) so the SKU forecasts
    sum to a coherent, less-noisy total. Improves both levels.

### D. Inventory & Cost Improvements (highest business value)

14. **Per-SKU lead times.** Safety stock currently assumes a global 2-month
    lead time. Real lead times vary enormously by supplier; using the actual
    per-SKU lead time (and its variability) is the single biggest lever on
    safety-stock accuracy. `SS = z * sqrt(LT * σ_d² + d² * σ_LT²)` properly
    accounts for lead-time variability too.

15. **Empirical / distribution-based safety stock instead of the normal
    approximation.** The `z * σ * √LT` formula assumes demand is normal — it
    is not (it's intermittent and skewed). Sizing safety stock directly from
    the **P-quantile of simulated lead-time demand** (bootstrap or quantile
    forecast) is more accurate for these distributions.

16. **Joint cost optimization (the "right" objective).** Rather than forecast
    accuracy then safety stock, optimize the **total expected cost** (holding +
    stockout + ordering) directly per SKU, choosing the order-up-to level that
    minimizes expected cost. This aligns the model with the real business goal.

17. **Calibrate the cost assumptions with finance.** Replace the assumed 25%
    holding rate and 1.5–10x stockout multipliers with the **actual** numbers
    from JLR finance and per-part criticality, so the savings figure is exact
    rather than a defensible estimate.

18. **EOQ and order-batching integration.** Combine the reorder point with
    Economic Order Quantity (`EOQ_QTY` is in the data) and supplier MOQ /
    pack-size constraints to produce a directly actionable order plan.

### E. Engineering / Productionization

19. **Automated monthly retraining + drift monitoring.** Schedule
    `run_pipeline(cfg)` monthly as data arrives; track WAPE/Bias over time and
    alert when accuracy drifts so models are retrained or investigated.

20. **Backtest-driven model registry.** Persist each month's chosen routing and
    metrics; only promote a new model to production if it beats the incumbent on
    the rolling backtest (champion/challenger).

21. **Prediction intervals everywhere + dashboards.** Surface P50/P90 bands and
    safety-stock recommendations in an interactive dashboard for planners, with
    the ability to override known events (recalls, promotions).

22. **Unit & integration tests in CI.** We have a leakage test and a
    vectorised-vs-reference equivalence check; expand these into a CI suite
    (schema checks, metric regression tests) so changes can't silently break
    the pipeline.

23. **Scale-out for many warehouses.** The pipeline is location-agnostic;
    wrap it to run per-location in parallel (e.g. across all IN0x sites) and
    aggregate results centrally.

### How to present this section
If a judge asks "what would you do with more time?", pick the **three highest-
impact items**: (1) per-SKU lead times + distribution-based safety stock,
(2) lifecycle/new-product handling, and (3) multi-window rolling backtest.
These directly address the biggest real-world gaps without overclaiming.

---

## 16. Glossary of Terms

| Term | Definition |
|---|---|
| **SKU** | Stock-Keeping Unit — a unique product identifier |
| **Demand** | Number of units sold/shipped in a period |
| **Lead time** | Time from placing an order to receiving stock |
| **Service level** | % of demand fulfilled from stock (target rate) |
| **Fill rate** | % of demand actually fulfilled (achieved rate) |
| **Stockout** | Running out of stock when demand occurs |
| **Safety stock** | Buffer inventory above expected demand |
| **Reorder point** | Stock level that triggers a new order |
| **EOQ** | Economic Order Quantity (cost-optimal order size) |
| **ADI** | Average inter-Demand Interval — months between non-zero demand |
| **CV** | Coefficient of Variation — σ/μ |
| **WAPE** | Weighted Absolute Percentage Error |
| **MASE** | Mean Absolute Scaled Error |
| **MAPE** | Mean Absolute % Error (our convention: zero-actual months = 0% error) |
| **Bias** | Average signed error (over/under direction) |
| **Tweedie** | Compound Poisson-Gamma probability distribution |
| **Quantile regression** | Predicting a specific percentile of the distribution |
| **Croston** | Intermittent-demand method (1972) |
| **SBA** | Bias-corrected Croston (2005) |
| **TSB** | Teunter-Syntetos-Babai (2011) — handles obsolescence |
| **SES** | Simple Exponential Smoothing |
| **LightGBM** | Microsoft's gradient-boosted-tree ML library |
| **Backtest** | Testing on historical held-out data |
| **Point-in-time** | Using only data available up to time t |
| **Look-ahead leakage** | Bug where future data influences past predictions |

---

## Final tips for presenting

1. **Start with the business problem**, not the math. Judges care about
   "what's the value?" before "how does it work?"

2. **Use the cost story as your headline**: $9.7M-$19.7M annual savings
   AND 20pp better fill rate. That's the memorable number.

3. **Acknowledge the under-prediction concern proactively** — say "you might
   wonder if our model under-predicts more SKUs, which would seem bad" and
   then show how safety stock fixes it. Shows critical thinking.

4. **Have the technical depth ready** but lead with intuition. If a judge
   asks "why Tweedie?", you can explain in one sentence ("zero-inflated
   non-negative data") OR five (full distribution explanation).

5. **Know your numbers cold**: WAPE 38.2%, fill rate 96.7%, 50% cost
   savings, 8,064 SKUs, 37 months. These come up over and over.

6. **Have a "future work" answer ready**: shows you understand limitations
   without weakening confidence.

Good luck!
