# Handover — Fix the Asymmetric Cost Comparison

> **Goal:** Replace the unfair "our forecast + safety stock vs ERP's bare forecast"
> comparison with a **symmetric** comparison so the savings number is defensible
> against any judge.
>
> This document contains everything needed to complete the fix in a new session.

---

## 1. Where the work lives

| What | Where |
|---|---|
| Branch with the 80-cell notebook (cost framework) | `improve/forecasting-v2` (commit `cd5e3db` or `398566e` had 80 cells) |
| Branch tip currently | only **51 cells** (cost sections were dropped). You need the 80-cell version. |
| Data file | `IN00_ST_DATA_3Years.xlsx` — uploaded to GitHub (on `main` branch). If missing locally, run: `git checkout origin/main -- IN00_ST_DATA_3Years.xlsx` |
| Forecast pipeline module | `improved/forecast_pipeline.py` |
| Notebook to fix | `improved/final_v2.ipynb` |
| Presentation guide to update | `improved/PRESENTATION_GUIDE.md` |
| Slide deck to update | `improved/JLR_Forecasting_Presentation.pptx` |

### How to restore the 80-cell notebook
```bash
cd /projects/sandbox/gic
git checkout improve/forecasting-v2
git checkout origin/main -- IN00_ST_DATA_3Years.xlsx   # restore data file
# Pull the 80-cell notebook from history and overwrite the current 51-cell one:
git checkout cd5e3db -- improved/final_v2.ipynb
```
After that, sections 16–26 (cost framework) are back.

---

## 2. The mistake — stated clearly

In the current 80-cell notebook, **Section 22 (Corrected Cost Framework)** does this:

```python
# Our model: forecast + safety stock buffer
hybrid_augmented = hybrid_preds + ss_per_month

# ERP: just the raw forecast (NO buffer added)
sys_test = fcst_h[:, TEST_INDICES]

# Compute holding + stockout cost on each, compare -> claim ~50% saving
```

This is **asymmetric and therefore invalid**:

1. We added a service-level safety buffer to **our** forecast but **not** to the ERP forecast.
2. The ERP's `FCST_H` is the bare forecast — the ERP system applies its own safety stock **separately** (the dataset has dedicated columns `SAFTY` and `REORDER_POINT` precisely for this).
3. So the comparison was effectively "our forecast + buffer" vs "ERP forecast - its buffer."
4. Adding any buffer to either side will crush stockouts and look like a huge win, especially because we assumed stockout cost = 1.5x-10x unit cost. That made stockouts ~98% of total cost.

**Result:** The "~50% savings / $4.8M-$9.8M annualised" claim is inflated and won't survive a sharp judge's questioning.

**What's still trustworthy:** The pure **accuracy** numbers (no cost assumptions, no buffering): WAPE 38.2 vs 41.0, MASE 0.82 vs 0.98, MAE 3.01 vs 3.23, RMSE 36.5 vs 40.3. Lead with these.

---

## 3. The correct symmetric comparison

There are two clean options. **Pick Option A** — it's the most defensible because it uses the ERP's *actual* policy as written in the data, not a recreated proxy.

### Option A (preferred) — use the ERP's real `REORDER_POINT` / `SAFTY` columns

The ERP already publishes its inventory position. Compare:
- Our side: hybrid forecast + service-level safety stock (`SS = z(TSL)*sigma*sqrt(LT)`)
- ERP side: the actual `REORDER_POINT` column (which is ERP_forecast + ERP_safety_stock)

This is the cleanest "what would each system actually order?" comparison.

### Option B — apply the same safety-stock formula to both forecasts

- Our side:   `hybrid + z(TSL)*sigma*sqrt(LT) / LT` per month
- ERP side:   `FCST_H + z(TSL)*sigma*sqrt(LT) / LT` per month   <- *the missing line*

Same buffer, same TSL, same sigma for both -> any difference is purely from forecast accuracy.

---

## 4. Drop-in replacement for Section 22 (Option B — symmetric buffering)

Replace the existing Section 22 cells with this. Keep Sections 16–21 (raw comparison) — that's the honest "no buffer either side" story; it stays.

````python
# ============================================================================
# CORRECTED COST FRAMEWORK — symmetric safety stock on BOTH sides
# ============================================================================
# Why: previously we buffered only OUR forecast and compared against the ERP's
# bare FCST_H. That's asymmetric. Here we apply the SAME service-level safety
# stock (z(TSL)*sigma*sqrt(LT), amortised per month) to BOTH forecasts. Any
# remaining saving is purely from our forecast being more accurate (less
# buffer needed to hit the same service level).
from scipy.stats import norm

HOLDING_RATE = 0.25 / 12          # monthly
LEAD_TIME    = 2.0                # months — same assumption as before
                                  # (calibrate to real lead times before deployment)

# Per-SKU service-level safety stock (same formula for both forecasts)
sigma_12  = demand[:, -12:].std(axis=1)
sigma_lt  = sigma_12 * np.sqrt(LEAD_TIME)

if "TSL" in inv.columns:
    sl = inv["TSL"].values / 100.0
    sl = np.where(np.isfinite(sl) & (sl > 0), sl, 0.95)
else:
    sl = np.full(len(demand), 0.95)
z_scores     = norm.ppf(np.clip(sl, 0.5, 0.999))
ss_per_month = (z_scores * sigma_lt) / LEAD_TIME    # amortised monthly buffer

# Augment BOTH forecasts with the same buffer  <- the fix
sys_test          = fcst_h[:, TEST_INDICES]
hybrid_augmented  = hybrid_preds + ss_per_month.reshape(-1, 1)
sys_augmented     = sys_test     + ss_per_month.reshape(-1, 1)   # <- NEW

hybrid_aug_errors = hybrid_augmented - actual_test
sys_aug_errors    = sys_augmented    - actual_test

def total_cost(errors, cost_arr, holding_rate_mo, stockout_mult):
    h_per_mo  = cost_arr * holding_rate_mo
    s_per_unit= cost_arr * stockout_mult
    holding   = (np.maximum( errors, 0) * h_per_mo.reshape(-1, 1)).sum()
    stockout  = (np.maximum(-errors, 0) * s_per_unit.reshape(-1, 1)).sum()
    return holding, stockout, holding + stockout

print("=" * 90)
print("SYMMETRIC COMPARISON — both forecasts get the same service-level safety stock")
print("=" * 90)
print(f"\n{'Stockout mult':<14} {'Hybrid+SS($)':<18} {'ERP+SS($)':<18} {'Savings($)':<15} {'Savings%':<10}")
print("-" * 80)
for mult in [1.0, 1.5, 2.0, 3.0, 5.0]:
    _, _, h_t = total_cost(hybrid_aug_errors, cost, HOLDING_RATE, mult)
    _, _, s_t = total_cost(sys_aug_errors,    cost, HOLDING_RATE, mult)
    sav = s_t - h_t
    pct = sav / s_t * 100 if s_t > 0 else 0
    print(f"{mult:<14.1f} ${h_t:<16,.0f} ${s_t:<16,.0f} ${sav:<13,.0f} {pct:+.1f}%")

# Fill rate comparison (both with safety stock)
print("\nFill rate (% of actual demand met) — both buffered the same way:")
for j, t_idx in enumerate(TEST_INDICES):
    mo = cal_months[t_idx].strftime("%b-%y")
    h_fill = np.minimum(hybrid_augmented[:, j], actual_test[:, j]).sum() / max(actual_test[:, j].sum(), 1) * 100
    s_fill = np.minimum(sys_augmented[:, j],    actual_test[:, j]).sum() / max(actual_test[:, j].sum(), 1) * 100
    print(f"  {mo}: Hybrid+SS = {h_fill:.1f}% | ERP+SS = {s_fill:.1f}%")
````

### Optional — Option A cell (using the ERP's real REORDER_POINT)
Add this as an extra cell so you can show both framings:

````python
# ============================================================================
# OPTION A — vs the ERP's ACTUAL published reorder policy
# ============================================================================
# The dataset has REORDER_POINT (= ERP forecast + ERP safety stock). Use it as
# the ERP's real position rather than rebuilding a proxy.
if "REORDER_POINT" in inv.columns:
    erp_rop_monthly = (inv["REORDER_POINT"].values / LEAD_TIME).reshape(-1, 1)
    erp_aug         = np.broadcast_to(erp_rop_monthly, hybrid_augmented.shape)
    erp_aug_errors  = erp_aug - actual_test

    print("\nVs ERP's published REORDER_POINT (its real policy):")
    print(f"{'Stockout':<10} {'Hybrid+SS($)':<18} {'ERP_real($)':<18} {'Savings($)':<15} {'Savings%':<10}")
    for mult in [1.5, 3.0, 5.0]:
        _, _, h_t = total_cost(hybrid_aug_errors, cost, HOLDING_RATE, mult)
        _, _, e_t = total_cost(erp_aug_errors,    cost, HOLDING_RATE, mult)
        sav = e_t - h_t
        pct = sav / e_t * 100 if e_t > 0 else 0
        print(f"{mult:<10.1f} ${h_t:<16,.0f} ${e_t:<16,.0f} ${sav:<13,.0f} {pct:+.1f}%")
````

---

## 5. What to expect from the corrected numbers

You will see:
- **Saving drops a lot** — likely **single-digit % to low-double-digit %** instead of 50%.
- **Fill rate of the ERP+SS will jump close to the hybrid's** (because both now have a safety buffer).
- **The remaining gap is purely from forecast accuracy** — which is the *honest, defensible* story.

That smaller saving is the right number to claim. It's the outcome of "more accurate forecast -> less buffer needed for the same service level."

---

## 6. Other places that must be updated after the fix

After you re-run Section 22 (and Option A), update these to remove the inflated numbers:

### 6.1 Section 23 — Scenario analysis at varying penalties
The table in Section 23 used `hybrid_aug_errors` vs `sys_errors` (asymmetric).
Change `sys_errors` -> `sys_aug_errors`. Same loop, just the symmetric inputs.

### 6.2 Section 24 — Fill rate chart
Replace the ERP curve to use `sys_augmented` rather than bare `sys_test`.

### 6.3 Section 25 — Over/under prediction "before and after SS"
The "ERP system (as-is)" row should become "ERP + SS" with the buffered numbers.

### 6.4 Section 26 — Executive summary
Replace the headline cost-saving numbers and the annualised estimate with the
symmetric-comparison output. Suggested replacement text:

> **What you'd actually order, both sides buffered to the same service level:**
> - Hybrid + SS vs ERP + SS at the same TSL and lead time
> - Saving of about **X%** (~ $Y over 6 months / $2Y annualised) at a 1.5x stockout multiplier
> - Saving holds across stockout multipliers from 1x to 5x
> - The saving comes from **lower forecast error -> less buffer needed for the same fill rate**

### 6.5 Slide deck — `JLR_Forecasting_Presentation.pptx`
- **Slide 12** ("Results — Cost & Service") currently shows "~50% lower cost". Replace with the corrected symmetric number.
- Reword the rebuttal box to: *"You never order the raw point forecast — both we AND the ERP add safety stock. With the same buffer policy on both sides, our advantage comes purely from being more accurate, so we need less buffer for the same service level."*

### 6.6 Presentation guide — `PRESENTATION_GUIDE.md`
- Section 12 ("The Cost Framework"): rewrite to describe the symmetric methodology.
- Section 13 ("Results Summary"): replace the cost-saving table with the corrected numbers.
- Q&A:
  - **Update Q4** ("Your model under-predicts more SKUs — isn't that bad?") to the symmetric framing.
  - **Add a new Q**: *"Doesn't the ERP already include its own safety stock?"* -> "Yes — that's exactly why our cost comparison applies the same safety-stock formula to both forecasts (or alternatively uses the ERP's published REORDER_POINT). Any remaining saving comes purely from the forecast accuracy gap, not from a buffering trick."

---

## 7. End-to-end execution checklist

1. `git checkout improve/forecasting-v2`
2. `git checkout origin/main -- IN00_ST_DATA_3Years.xlsx` (restore data file)
3. `git checkout cd5e3db -- improved/final_v2.ipynb` (restore 80-cell notebook)
4. Open `improved/final_v2.ipynb`. In Section 22, replace the cell as in section 4 above.
5. Update Sections 23, 24, 25, 26 per section 6.
6. Re-execute the whole notebook top to bottom on the real data.
7. Update `PRESENTATION_GUIDE.md` per section 6.6.
8. Update `JLR_Forecasting_Presentation.pptx` slide 12 per section 6.5.
9. Commit with a message like:
   `"Fix asymmetric cost comparison: apply same safety-stock policy to both forecasts"`
10. Push.

### To re-execute the notebook from the command line
```bash
pip install -q nbconvert jupyter ipykernel matplotlib seaborn lightgbm xgboost \
    scikit-learn openpyxl scipy pandas numpy
cd /projects/sandbox/gic/improved
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 final_v2.ipynb
```

---

## 8. What to claim safely after the fix

- **Accuracy (rock-solid, no assumptions):**
  WAPE 38.2% vs 41.0% (-6.8%), MASE 0.82 vs 0.98, MAE 3.01 vs 3.23, RMSE 36.5 vs 40.3.
- **Cost (after symmetric correction):**
  *Replace with the actual number from the rerun. Expected: single-digit to low-double-digit %.*
- **Inventory:**
  Per-SKU service-level safety stock (`z(TSL)*sigma*sqrt(LT)`) makes our forecast
  directly actionable as a reorder policy. The reduction comes from needing
  less buffer to hit the same target service level.

---

## 9. One-line summary

> The 50% savings claim was inflated because we added safety stock only to our
> forecast. The fix is to add the same safety stock to the ERP's forecast (or
> use its published REORDER_POINT), then the saving reduces to a smaller but
> defensible number driven purely by forecast accuracy.
