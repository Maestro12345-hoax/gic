"""
Financial comparison: Hybrid vs ERP (FCST_H)
   * Stockout multiplier = 0.2x  (per user request)
   * Holding rate         = 25% / yr -> 25/12 % per month  (matches notebook default)
   * No safety stock      (per user request)

Inputs come from the already-produced backtest workbook
    improved/IN00_forecast_results_v2.xlsx
sheet `SKU_Forecasts`, which has:
   - actual_<MONTH>   : ground-truth demand
   - hybrid_<MONTH>   : pattern-routed hybrid model forecast
   - sys_<MONTH>      : ERP FCST_H forecast
   - COST             : per-SKU unit cost

Cost model (no safety stock on either side):
   error_units    = forecast - actual
   over_units     = max(error_units,  0)   # we ordered too many -> sit in stock
   under_units    = max(-error_units, 0)   # short -> stockout
   holding_cost   = over_units  * COST * (HOLDING_RATE / 12)   # carrying cost
   stockout_cost  = under_units * COST * STOCKOUT_MULT         # lost margin
   total_cost     = holding_cost + stockout_cost
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
WORKBOOK = HERE / "IN00_forecast_results_v2.xlsx"

STOCKOUT_MULT = 0.2          # user request
HOLDING_RATE = 0.25 / 12     # 25%/yr / 12 = monthly holding rate
SAFETY_STOCK = 0             # user request: no safety stock

TEST_MONTHS = ["Nov25", "Dec25", "Jan26", "Feb26", "Mar26", "Apr26"]


def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="SKU_Forecasts")
    return df


def stack_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Build (n_skus, n_months) array from columns named e.g. hybrid_Nov25."""
    cols = [f"{prefix}_{m}" for m in TEST_MONTHS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in workbook: {missing}")
    return df[cols].to_numpy(dtype=float)


def compute_costs(forecast: np.ndarray, actual: np.ndarray, cost: np.ndarray,
                  stockout_mult: float, holding_rate: float, safety_stock: float = 0.0):
    """Per-(SKU, month) holding and stockout dollar cost arrays."""
    forecast = np.maximum(forecast, 0.0)
    actual = np.maximum(actual, 0.0)
    order_qty = forecast + safety_stock
    over = np.maximum(order_qty - actual, 0.0)
    under = np.maximum(actual - order_qty, 0.0)

    cost_col = cost.reshape(-1, 1)
    holding = over * cost_col * holding_rate
    stockout = under * cost_col * stockout_mult
    return holding, stockout, over, under


def per_sku_total(arr: np.ndarray) -> np.ndarray:
    return arr.sum(axis=1)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    df = load_forecasts(WORKBOOK)
    n = len(df)
    cost = df["COST"].to_numpy(dtype=float)

    actual = stack_matrix(df, "actual")
    hybrid = stack_matrix(df, "hybrid")
    erp = stack_matrix(df, "sys")

    h_hold, h_stock, h_over_u, h_under_u = compute_costs(
        hybrid, actual, cost, STOCKOUT_MULT, HOLDING_RATE, SAFETY_STOCK)
    e_hold, e_stock, e_over_u, e_under_u = compute_costs(
        erp, actual, cost, STOCKOUT_MULT, HOLDING_RATE, SAFETY_STOCK)

    h_total = h_hold + h_stock
    e_total = e_hold + e_stock

    line = "=" * 90
    print(line)
    print("FINANCIAL COMPARISON: Hybrid vs ERP (FCST_H)")
    print(f"  Stockout multiplier : {STOCKOUT_MULT}x  (cost per shortage unit = {STOCKOUT_MULT} x COST)")
    print(f"  Holding rate        : {HOLDING_RATE * 12:.0%}/yr ({HOLDING_RATE * 100:.4f}%/mo)")
    print(f"  Safety stock        : {SAFETY_STOCK} units (none)")
    print(f"  SKUs                : {n:,}")
    print(f"  Test period         : Nov-2025 -> Apr-2026 (6 months)")
    print(line)

    summary = pd.DataFrame({
        "Method": ["Hybrid", "ERP (FCST_H)"],
        "Over_units": [h_over_u.sum(), e_over_u.sum()],
        "Short_units": [h_under_u.sum(), e_under_u.sum()],
        "Holding_$": [h_hold.sum(), e_hold.sum()],
        "Stockout_$": [h_stock.sum(), e_stock.sum()],
        "Total_$": [h_total.sum(), e_total.sum()],
    })
    pd.options.display.float_format = "{:,.2f}".format
    print("\nTotal financial impact over the 6 test months:")
    print(summary.to_string(index=False))

    savings = e_total.sum() - h_total.sum()
    pct = savings / e_total.sum() * 100 if e_total.sum() > 0 else 0.0
    print(f"\n>>> Hybrid vs ERP: savings = {fmt_money(savings)}  ({pct:+.2f}%)")
    print(f">>> Annualised (x2 for 12 months): {fmt_money(savings * 2)}")

    # Per-month breakdown
    print("\nPer-month breakdown:")
    monthly = pd.DataFrame({
        "month": TEST_MONTHS,
        "actual_units": actual.sum(axis=0).round(0),
        "hybrid_holding_$": h_hold.sum(axis=0).round(2),
        "hybrid_stockout_$": h_stock.sum(axis=0).round(2),
        "hybrid_total_$": h_total.sum(axis=0).round(2),
        "erp_holding_$": e_hold.sum(axis=0).round(2),
        "erp_stockout_$": e_stock.sum(axis=0).round(2),
        "erp_total_$": e_total.sum(axis=0).round(2),
    })
    monthly["savings_$"] = (monthly["erp_total_$"] - monthly["hybrid_total_$"]).round(2)
    monthly["savings_%"] = ((monthly["savings_$"] / monthly["erp_total_$"]) * 100).round(2)
    print(monthly.to_string(index=False))

    # Per-MOVE_CLASS breakdown
    if "MOVE_CLASS" in df.columns:
        rows = []
        for cls in sorted(df["MOVE_CLASS"].dropna().unique()):
            mask = (df["MOVE_CLASS"].values == cls)
            h_t = h_total[mask].sum()
            e_t = e_total[mask].sum()
            rows.append({
                "MOVE_CLASS": cls,
                "n_sku": int(mask.sum()),
                "hybrid_total_$": round(h_t, 2),
                "erp_total_$": round(e_t, 2),
                "savings_$": round(e_t - h_t, 2),
                "savings_%": round((e_t - h_t) / e_t * 100, 2) if e_t > 0 else 0.0,
            })
        print("\nBy MOVE_CLASS:")
        print(pd.DataFrame(rows).to_string(index=False))

    # Per-demand_type breakdown
    if "demand_type" in df.columns:
        rows = []
        for patt in sorted(df["demand_type"].dropna().unique()):
            mask = (df["demand_type"].values == patt)
            h_t = h_total[mask].sum()
            e_t = e_total[mask].sum()
            rows.append({
                "demand_type": patt,
                "n_sku": int(mask.sum()),
                "hybrid_total_$": round(h_t, 2),
                "erp_total_$": round(e_t, 2),
                "savings_$": round(e_t - h_t, 2),
                "savings_%": round((e_t - h_t) / e_t * 100, 2) if e_t > 0 else 0.0,
            })
        print("\nBy demand_type:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
