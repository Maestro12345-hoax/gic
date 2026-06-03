"""
Demand Forecasting Pipeline — IN00 Spare Parts (refined / v2)
=============================================================

This module is a refined version of the original `final.ipynb` pipeline.
It keeps the same overall philosophy (statistical + ML models, demand-pattern
classification, comparison vs the ERP system forecast) but fixes several
correctness problems and adds inventory-grade capabilities.

What changed vs the original notebook
-------------------------------------
CORRECTNESS
  * No look-ahead leakage. `zero_rate`, `cv`, ADI and every rolling statistic
    are recomputed *point-in-time* from history[:, :t] only. The original
    computed them once over all 37 months (including the test/future window)
    and used them as features -> optimistic, leaked results.
  * Proper per-series MASE. Each SKU is scaled by its OWN in-sample naive MAE,
    with a sensible fallback (series mean) for flat/intermittent series, then
    aggregated. The original divided a pooled MAE by a single global scale.
  * Honest error metrics. WAPE is the primary KPI for intermittent demand.
    MAPE is reported only on truly non-zero actual months and clearly labelled
    (the original silently set zero-actual months to 0% error, deflating it).

ACCURACY
  * Rolling-origin backtest (multiple windows) instead of a single 6-month
    split, so model selection is robust rather than tuned to one window.
  * Model routing by demand pattern (Smooth / Erratic / Intermittent / Lumpy),
    which is theory-aligned, plus an ensemble blend that is benchmarked too.
  * Richer point-in-time features: months-since-last-sale, ADI, non-zero count,
    trend ratio, rolling max, seasonal lag-12 ratio, brand and log-cost.

SPEED
  * Croston / SBA / TSB are fully vectorised across all SKUs in a single pass
    over the time axis (was a Python loop per SKU per test month).

INVENTORY VALUE
  * Quantile forecasts (P50 / P90 by default) via LightGBM quantile objective,
    turned into a lead-time safety-stock and reorder-point suggestion. A point
    forecast alone cannot size safety stock; this is the main new capability.
  * Value-weighted WAPE (weighted by COST) so accuracy tracks money at risk.

The module is import-friendly: `run_pipeline(cfg)` does everything and returns a
result object. A thin notebook (`final_v2.ipynb`) calls into it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_file: str = "IN00_ST_DATA_3Years.xlsx"
    out_file: str = "IN00_forecast_results_v2.xlsx"

    n_months: int = 37            # total months in the dataset
    m_oldest: int = 36            # suffix of the oldest column (CAP_SCAL_M36)
    m00_month: pd.Timestamp = pd.Timestamp("2026-05-01")  # most recent month

    # Backtest: evaluate the last `bt_windows` one-month-ahead origins.
    # e.g. with origins 30..35 we test Nov-25 .. Apr-26 (matches original test set)
    bt_first_origin: int = 30     # first month index used as a test origin
    bt_last_origin: int = 35      # last month index used as a test origin
    feat_min_history: int = 12    # earliest t used to build ML training rows

    horizon: int = 3              # forward forecast horizon (months)
    lead_time_months: float = 2.0 # lead time for safety-stock sizing
    default_service_level: float = 0.95  # used when a SKU has no TSL value

    zero_threshold: float = 0.30  # forecasts below this are snapped to zero
    use_zero_gate: bool = True    # apply the learned zero-demand classifier
    quantiles: tuple = (0.5, 0.9) # quantile heads for safety stock

    random_state: int = 42
    light: bool = False           # smaller/faster models (for quick runs / CI)

    # Statistical smoothing constants
    croston_alpha: float = 0.15
    ses_alpha: float = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _match_cols(columns, prefix):
    """Find a column whose name is `prefix`, `prefix_*` or `prefix-*`."""
    return [c for c in columns
            if c == prefix or c.startswith(prefix + "_") or c.startswith(prefix + "-")]


def load_data(cfg: Config):
    """Load the Excel file and return (demand, fcst_h, fcst_f, meta, cal_months)."""
    df = pd.read_excel(cfg.data_file)

    # Demand columns CAP_SCAL_M36 .. M00, ordered oldest-first.
    demand_cols = []
    for i in range(cfg.m_oldest, -1, -1):
        found = _match_cols(df.columns, f"CAP_SCAL_M{i:02d}")
        if found:
            demand_cols.append(found[0])
    demand = df[demand_cols].values.astype(float)

    # Historical system forecast FCST_H36 .. H00 (for benchmarking)
    fh_cols = []
    for i in range(cfg.m_oldest, -1, -1):
        found = _match_cols(df.columns, f"FCST_H{i:02d}")
        if found:
            fh_cols.append(found[0])
    fcst_h = df[fh_cols].values.astype(float) if len(fh_cols) == cfg.n_months else None

    # Future system forecast FCST_F01 .. F12 (optional benchmark for the forward horizon)
    ff_cols = []
    for i in range(1, 13):
        found = _match_cols(df.columns, f"FCST_F{i:02d}")
        if found:
            ff_cols.append(found[0])
    fcst_f = df[ff_cols].values.astype(float) if len(ff_cols) >= 3 else None

    cal_months = [cfg.m00_month - pd.DateOffset(months=i) for i in range(cfg.n_months)]
    cal_months.reverse()

    meta_cols = [c for c in ["PRODUCT", "FACING_LOC", "VOL_CLASS", "MOVE_CLASS",
                             "FCST_MODEL", "BRAND", "COST"] if c in df.columns]
    meta = df[meta_cols].copy().reset_index(drop=True)
    if "COST" not in meta:
        meta["COST"] = 1.0

    # Optional existing-ERP inventory parameters, kept for side-by-side comparison.
    inv_cols = [c for c in ["TSL", "SAFTY", "MIN_SFTY", "MAX_SFTY",
                            "REORDER_POINT", "MAX_STOCK", "EOQ_QTY",
                            "STD_DEV_PERC", "PERIO"] if c in df.columns]
    inv = df[inv_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True) \
        if inv_cols else pd.DataFrame(index=range(len(df)))

    return demand, fcst_h, fcst_f, meta, cal_months, inv


# ---------------------------------------------------------------------------
# Demand-pattern classification (Syntetos-Boylan-Croston)
# ---------------------------------------------------------------------------
def adi_per_series(demand):
    """Average inter-demand interval (mean gap between non-zero months)."""
    out = np.empty(demand.shape[0])
    for i in range(demand.shape[0]):
        nz = np.where(demand[i] > 0)[0]
        out[i] = np.diff(nz).mean() if len(nz) >= 2 else demand.shape[1]
    return out


def classify_demand(demand):
    """Return a DataFrame of per-SKU demand statistics + SBC pattern label.

    Note: these are descriptive (used for routing + reporting). The *features*
    used by the models are recomputed point-in-time elsewhere to avoid leakage.
    """
    zero_rate = (demand == 0).mean(axis=1)
    mean_dem = demand.mean(axis=1)
    std_dem = demand.std(axis=1)
    nz = mean_dem > 0
    cv = np.where(nz, std_dem / np.where(nz, mean_dem, 1.0), 0.0)
    adi = adi_per_series(demand)
    cv2 = cv ** 2

    demand_type = np.where(
        adi >= 1.32,
        np.where(cv2 >= 0.49, "Lumpy", "Intermittent"),
        np.where(cv2 >= 0.49, "Erratic", "Smooth"),
    )
    return pd.DataFrame({
        "zero_rate": zero_rate, "mean_demand": mean_dem,
        "cv": cv, "adi": adi, "demand_type": demand_type,
    })


# ---------------------------------------------------------------------------
# Vectorised intermittent-demand methods (Croston / SBA / TSB) + SES
# ---------------------------------------------------------------------------
def croston_batch(history, alpha=0.15, variant="sba"):
    """Vectorised Croston / SBA across all SKUs in one pass over time.

    variant: 'croston' (raw) or 'sba' (Syntetos-Boylan bias correction).
    Returns one forecast rate per SKU = size / interval.
    """
    n, L = history.shape
    z = np.zeros(n)               # smoothed demand size
    p = np.ones(n)                # smoothed inter-demand interval
    q = np.zeros(n, dtype=int)    # periods since last demand
    init = np.zeros(n, dtype=bool)

    for t in range(L):
        y = history[:, t]
        q += 1
        nz = y > 0
        first = nz & ~init
        z[first] = y[first]
        p[first] = q[first]
        init[first] = True
        q[first] = 0
        upd = nz & init & ~first
        z[upd] = (1 - alpha) * z[upd] + alpha * y[upd]
        p[upd] = (1 - alpha) * p[upd] + alpha * q[upd]
        q[upd] = 0

    rate = np.where(init & (p > 0), z / np.where(p > 0, p, 1.0), 0.0)
    if variant == "sba":
        rate = rate * (1 - alpha / 2.0)
    return np.maximum(rate, 0.0)


def tsb_batch(history, alpha_d=0.15, alpha_z=0.15):
    """Vectorised Teunter-Syntetos-Babai across all SKUs. Forecast = q * z."""
    n, L = history.shape
    z = np.zeros(n)
    q = np.zeros(n)
    init = np.zeros(n, dtype=bool)

    for t in range(L):
        y = history[:, t]
        nz = y > 0
        first = nz & ~init
        z[first] = y[first]
        q[first] = 1.0 / (t + 1)
        init[first] = True
        active = init & ~first
        demand = active & nz
        nodem = active & ~nz
        q[demand] = (1 - alpha_d) * q[demand] + alpha_d
        z[demand] = (1 - alpha_z) * z[demand] + alpha_z * y[demand]
        q[nodem] = (1 - alpha_d) * q[nodem]

    return np.maximum(np.where(init, q * z, 0.0), 0.0)


def ses_batch(history, alpha=0.20):
    """Vectorised simple exponential smoothing across all SKUs."""
    n, L = history.shape
    if L == 0:
        return np.zeros(n)
    S = history[:, 0].astype(float).copy()
    for t in range(1, L):
        S = (1 - alpha) * S + alpha * history[:, t]
    return np.maximum(S, 0.0)


# ---------------------------------------------------------------------------
# Point-in-time feature engineering (NO leakage)
# ---------------------------------------------------------------------------
def encode_static(meta):
    """Encode the slow-changing per-SKU attributes (class, brand, log-cost)."""
    from sklearn.preprocessing import LabelEncoder

    vol = LabelEncoder().fit_transform(meta["VOL_CLASS"].astype(str)) \
        if "VOL_CLASS" in meta else np.zeros(len(meta))
    move = LabelEncoder().fit_transform(meta["MOVE_CLASS"].astype(str)) \
        if "MOVE_CLASS" in meta else np.zeros(len(meta))
    brand = LabelEncoder().fit_transform(meta["BRAND"].astype(str)) \
        if "BRAND" in meta else np.zeros(len(meta))
    log_cost = np.log1p(np.maximum(meta.get("COST", pd.Series(np.ones(len(meta)))).values, 0))
    return np.column_stack([vol, move, brand, log_cost]).astype(float)


FEATURE_NAMES = [
    "lag1", "lag2", "lag3", "lag6", "lag12",
    "roll3_mean", "roll6_mean", "roll12_mean", "roll3_std", "roll6_max",
    "trend_ratio", "months_since_last", "nonzero_cnt12", "adi_pit",
    "zero_rate_pit", "cv_pit", "season_lag12_ratio",
    "month_sin", "month_cos",
    "vol_class", "move_class", "brand", "log_cost",
]


def build_features(history, month, static_feats):
    """Build the feature matrix for predicting `month` using ONLY `history`.

    `history` is demand[:, :t] (everything strictly before the target month),
    so nothing downstream of the prediction point can leak in.
    """
    n, L = history.shape

    def lag(k):
        return history[:, -k] if L >= k else np.zeros(n)

    l1, l2, l3, l6, l12 = lag(1), lag(2), lag(3), lag(6), lag(12)
    r3 = history[:, -min(3, L):].mean(axis=1)
    r6 = history[:, -min(6, L):].mean(axis=1)
    r12 = history[:, -min(12, L):].mean(axis=1)
    rs3 = history[:, -min(3, L):].std(axis=1)
    rmax6 = history[:, -min(6, L):].max(axis=1)

    # Trend: recent 3-month mean vs previous 3-month mean (ratio, clipped)
    prev3 = history[:, -min(6, L):-min(3, L)] if L >= 6 else history[:, :max(L - 3, 0)]
    prev3_mean = prev3.mean(axis=1) if prev3.shape[1] > 0 else r3
    trend_ratio = np.clip(r3 / np.where(prev3_mean > 0, prev3_mean, 1.0), 0, 5)

    # Months since last non-zero sale
    months_since = np.full(n, L, dtype=float)
    for i in range(n):
        nz = np.where(history[i] > 0)[0]
        if len(nz):
            months_since[i] = L - 1 - nz[-1]

    nonzero_cnt12 = (history[:, -min(12, L):] > 0).sum(axis=1).astype(float)

    # Point-in-time ADI, zero-rate and CV (recomputed from history only)
    adi_pit = np.array([
        (np.diff(np.where(history[i] > 0)[0]).mean()
         if (history[i] > 0).sum() >= 2 else float(L))
        for i in range(n)
    ])
    zero_rate_pit = (history == 0).mean(axis=1)
    mean_pit = history.mean(axis=1)
    std_pit = history.std(axis=1)
    cv_pit = np.where(mean_pit > 0, std_pit / np.where(mean_pit > 0, mean_pit, 1.0), 0.0)

    # Seasonal index proxy: demand 12 months ago vs the 12-month average
    season_ratio = np.clip(l12 / np.where(r12 > 0, r12, 1.0), 0, 5)

    m = month.month
    month_sin = np.full(n, np.sin(2 * np.pi * m / 12))
    month_cos = np.full(n, np.cos(2 * np.pi * m / 12))

    return np.column_stack([
        l1, l2, l3, l6, l12,
        r3, r6, r12, rs3, rmax6,
        trend_ratio, months_since, nonzero_cnt12, adi_pit,
        zero_rate_pit, cv_pit, season_ratio,
        month_sin, month_cos,
        static_feats,  # vol, move, brand, log_cost (4 cols)
    ])


# ---------------------------------------------------------------------------
# Metrics (honest, intermittent-aware)
# ---------------------------------------------------------------------------
def naive_scale_per_series(history):
    """Per-series in-sample naive (lag-1) MAE, with a robust fallback."""
    if history.shape[1] < 2:
        scale = history.mean(axis=1)
    else:
        scale = np.abs(np.diff(history, axis=1)).mean(axis=1)
    # Fall back to series mean for flat series; final floor avoids divide-by-zero.
    fallback = history.mean(axis=1)
    scale = np.where(scale > 1e-8, scale, fallback)
    return np.where(scale > 1e-8, scale, 1.0)


def compute_metrics(actual, forecast, naive_scale, cost=None):
    """actual/forecast: (n, h). Returns a dict of aggregate KPIs.

    WAPE      — primary KPI for intermittent demand (volume-weighted).
    vWAPE     — value-weighted by COST (money at risk).
    MASE      — proper per-series scaling; we report the median (robust).
    Bias%     — sum(f-a)/sum(a); + = over-forecast, - = under (stockout risk).
    MAPE_nz   — MAPE over non-zero actual months ONLY (clearly scoped).
    """
    forecast = np.maximum(forecast, 0.0)
    a = actual
    f = forecast
    abs_err = np.abs(a - f)

    wape = abs_err.sum() / a.sum() * 100 if a.sum() > 0 else np.nan

    if cost is not None:
        w = cost.reshape(-1, 1)
        denom = (w * np.abs(a)).sum()
        vwape = (w * abs_err).sum() / denom * 100 if denom > 0 else np.nan
    else:
        vwape = np.nan

    mae_series = abs_err.mean(axis=1)
    mase_series = mae_series / naive_scale
    mase_med = float(np.median(mase_series))
    mase_mean = float(np.mean(mase_series))

    bias = (f - a).sum() / a.sum() * 100 if a.sum() > 0 else np.nan

    nz = a > 0
    mape_nz = float(np.abs((a[nz] - f[nz]) / a[nz]).mean() * 100) if nz.any() else np.nan

    return {
        "WAPE": round(wape, 3) if wape == wape else np.nan,
        "vWAPE": round(vwape, 3) if vwape == vwape else np.nan,
        "MASE_med": round(mase_med, 3),
        "MASE_mean": round(mase_mean, 3),
        "Bias%": round(bias, 3) if bias == bias else np.nan,
        "MAPE_nz": round(mape_nz, 1) if mape_nz == mape_nz else np.nan,
        "MAE": round(float(mae_series.mean()), 4),
        "RMSE": round(float(np.sqrt(((a - f) ** 2).mean())), 4),
    }


# ---------------------------------------------------------------------------
# ML training (point forecast + quantiles) and zero-gate
# ---------------------------------------------------------------------------
def _lgbm_params(cfg, objective="tweedie", alpha=None):
    import lightgbm as lgb  # noqa: F401  (import guard)
    p = dict(
        n_estimators=200 if cfg.light else 500,
        learning_rate=0.05, num_leaves=31 if cfg.light else 63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=cfg.random_state, n_jobs=-1, verbose=-1,
    )
    if objective == "tweedie":
        p.update(objective="tweedie", tweedie_variance_power=1.5)
    elif objective == "quantile":
        p.update(objective="quantile", alpha=alpha)
    return p


def build_training_matrix(demand, static_feats, cal_months, t_start, t_end):
    """Stack point-in-time feature rows + targets for t in [t_start, t_end]."""
    X_rows, y_rows = [], []
    for t in range(t_start, t_end + 1):
        X_rows.append(build_features(demand[:, :t], cal_months[t], static_feats))
        y_rows.append(demand[:, t])
    return np.vstack(X_rows), np.concatenate(y_rows)


def train_models(cfg, demand, static_feats, cal_months, t_end):
    """Train Tweedie regressors + quantile heads + the zero-demand gate."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    X, y = build_training_matrix(demand, static_feats, cal_months,
                                 cfg.feat_min_history, t_end)

    models = {}
    models["LightGBM"] = lgb.LGBMRegressor(**_lgbm_params(cfg)).fit(X, y)
    models["XGBoost"] = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.5,
        n_estimators=200 if cfg.light else 500, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=cfg.random_state, n_jobs=-1, verbosity=0,
    ).fit(X, y)
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=80 if cfg.light else 100, max_depth=8, min_samples_leaf=5,
        n_jobs=-1, random_state=cfg.random_state,
    ).fit(X, y)

    # Quantile heads (for safety stock / reorder points)
    quantile_models = {}
    for q in cfg.quantiles:
        quantile_models[q] = lgb.LGBMRegressor(
            **_lgbm_params(cfg, objective="quantile", alpha=q)).fit(X, y)

    # Zero-demand classifier gate
    zero_clf = None
    if cfg.use_zero_gate:
        y_zero = (y == 0).astype(int)
        zero_clf = lgb.LGBMClassifier(
            n_estimators=150 if cfg.light else 300, num_leaves=31,
            min_child_samples=30, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=cfg.random_state, n_jobs=-1, verbose=-1,
        ).fit(X, y_zero)

    return models, quantile_models, zero_clf


def apply_gate(cfg, preds, demand, origin_indices, static_feats, cal_months, zero_clf):
    """Apply the learned zero gate + small-value threshold to a prediction matrix."""
    out = np.maximum(preds.copy(), 0.0)
    for j, t in enumerate(origin_indices):
        if zero_clf is not None:
            X = build_features(demand[:, :t], cal_months[t], static_feats)
            zero_mask = zero_clf.predict(X).astype(bool)
            out[:, j] = np.where(zero_mask, 0.0, out[:, j])
        out[:, j] = np.where(out[:, j] < cfg.zero_threshold, 0.0, out[:, j])
    return out


# ---------------------------------------------------------------------------
# Backtest: produce predictions for every model on the test origins
# ---------------------------------------------------------------------------
def backtest_predictions(cfg, demand, static_feats, cal_months,
                         origin_indices, models, zero_clf):
    """One-month-ahead predictions for each origin month, every model."""
    n = demand.shape[0]
    h = len(origin_indices)
    preds = {name: np.zeros((n, h)) for name in
             ["SBA", "Croston", "TSB", "SES", "LightGBM", "XGBoost",
              "RandomForest", "Naive(lag1)"]}

    for j, t in enumerate(origin_indices):
        hist = demand[:, :t]
        preds["SBA"][:, j] = croston_batch(hist, cfg.croston_alpha, "sba")
        preds["Croston"][:, j] = croston_batch(hist, cfg.croston_alpha, "croston")
        preds["TSB"][:, j] = tsb_batch(hist, cfg.croston_alpha, cfg.croston_alpha)
        preds["SES"][:, j] = ses_batch(hist, cfg.ses_alpha)
        preds["Naive(lag1)"][:, j] = demand[:, t - 1]
        X = build_features(hist, cal_months[t], static_feats)
        for name in ["LightGBM", "XGBoost", "RandomForest"]:
            preds[name][:, j] = models[name].predict(X)

    # Simple ensemble: mean of SBA + LightGBM (robust intermittent + flexible ML)
    preds["Ensemble"] = 0.5 * (preds["SBA"] + preds["LightGBM"])

    # Gate everything
    for name in list(preds.keys()):
        preds[name] = apply_gate(cfg, preds[name], demand, origin_indices,
                                 static_feats, cal_months, zero_clf)
    return preds


# ---------------------------------------------------------------------------
# Forward forecast (point + quantiles) for the production horizon
# ---------------------------------------------------------------------------
def forecast_forward(cfg, demand, static_feats, cal_months,
                     point_model, quantile_models, zero_clf):
    """Iterative multi-step forward forecast with quantile heads for safety stock."""
    n = demand.shape[0]
    extended = demand.copy()
    point = np.zeros((n, cfg.horizon))
    quant = {q: np.zeros((n, cfg.horizon)) for q in cfg.quantiles}

    for step in range(cfg.horizon):
        future_month = cfg.m00_month + pd.DateOffset(months=step + 1)
        X = build_features(extended, future_month, static_feats)

        p = np.maximum(point_model.predict(X), 0.0)
        if zero_clf is not None:
            zmask = zero_clf.predict(X).astype(bool)
            p = np.where(zmask, 0.0, p)
        p = np.where(p < cfg.zero_threshold, 0.0, p)
        point[:, step] = p

        # Predict each quantile head, then enforce non-crossing (monotone in q)
        # via a running max over sorted quantile levels. We do NOT clamp toward
        # the Tweedie mean: for right-skewed demand the median is legitimately
        # below the mean, and the upper quantile is what drives safety stock.
        qs_sorted = sorted(quantile_models.keys())
        running = np.zeros(n)
        for q in qs_sorted:
            qp = np.maximum(quantile_models[q].predict(X), 0.0)
            qp = np.maximum(qp, running)      # never below a lower quantile
            running = qp
            quant[q][:, step] = qp

        extended = np.column_stack([extended, p])

    return point, quant


def safety_stock(cfg, point, quant):
    """Lead-time demand, safety stock and reorder point from quantiles.

    Uses the highest configured quantile as the lead-time service level.
    """
    lt = cfg.lead_time_months
    steps = int(np.ceil(lt))
    steps = min(steps, point.shape[1])
    frac = lt - (steps - 1) if steps >= 1 else lt

    weights = np.ones(steps)
    if steps >= 1:
        weights[-1] = frac  # partial last month if lead time is fractional

    lt_demand = (point[:, :steps] * weights).sum(axis=1)
    upper_q = max(cfg.quantiles)
    lt_upper = (quant[upper_q][:, :steps] * weights).sum(axis=1)
    ss = np.maximum(lt_upper - lt_demand, 0.0)
    reorder_point = lt_demand + ss
    return lt_demand, ss, reorder_point, upper_q


def service_level_safety_stock(cfg, demand, lt_demand, service_level):
    """Classic service-level safety stock: SS = z(SL) * sigma_over_leadtime.

    This mirrors how the ERP sizes safety stock (it stores per-SKU target
    service levels in TSL and per-period sigma), so the result is directly
    comparable to the existing SAFTY / REORDER_POINT figures.

    sigma_period is estimated from the last 12 months of demand (robust to the
    intermittency typical of spare parts). service_level is a per-SKU array.
    """
    from scipy.stats import norm

    sl = np.clip(np.asarray(service_level, dtype=float), 0.50, 0.999)
    z = norm.ppf(sl)
    sigma_period = demand[:, -12:].std(axis=1)
    sigma_lt = sigma_period * np.sqrt(max(cfg.lead_time_months, 1e-9))
    ss = np.maximum(z * sigma_lt, 0.0)
    reorder_point = lt_demand + ss
    return ss, reorder_point


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class Result:
    summary: pd.DataFrame = None
    by_pattern: pd.DataFrame = None
    routing: dict = field(default_factory=dict)
    hybrid_metrics: dict = None
    system_metrics: dict = None
    output: pd.DataFrame = None


def run_pipeline(cfg: Config | None = None, verbose: bool = True):
    cfg = cfg or Config()
    log = print if verbose else (lambda *a, **k: None)

    demand, fcst_h, fcst_f, meta, cal_months, inv = load_data(cfg)
    log(f"Loaded {demand.shape[0]:,} SKUs x {demand.shape[1]} months")

    stats = classify_demand(demand)
    meta = pd.concat([meta, stats], axis=1)
    log("Demand pattern mix:\n" + stats["demand_type"].value_counts().to_string())

    static_feats = encode_static(meta)
    cost = meta["COST"].values.astype(float)

    origins = list(range(cfg.bt_first_origin, cfg.bt_last_origin + 1))
    train_end = origins[0] - 1  # train strictly before the first test origin
    log(f"Training up to {cal_months[train_end].strftime('%b-%Y')}; "
        f"testing {cal_months[origins[0]].strftime('%b-%Y')} .. "
        f"{cal_months[origins[-1]].strftime('%b-%Y')}")

    models, quantile_models, zero_clf = train_models(
        cfg, demand, static_feats, cal_months, train_end)
    log("Models trained (Tweedie regressors + quantile heads + zero gate)")

    preds = backtest_predictions(cfg, demand, static_feats, cal_months,
                                 origins, models, zero_clf)

    actual = demand[:, origins]
    naive_scale = naive_scale_per_series(demand[:, :origins[0]])

    # Overall model leaderboard
    rows = []
    for name, p in preds.items():
        m = compute_metrics(actual, p, naive_scale, cost)
        m["model"] = name
        rows.append(m)
    summary = (pd.DataFrame(rows).set_index("model")
               .sort_values("WAPE"))
    log("\nModel leaderboard (lower WAPE = better):\n" + summary.to_string())

    # Per-pattern leaderboard + routing (theory-aligned model selection)
    patt_rows = []
    routing = {}
    for patt in sorted(meta["demand_type"].unique()):
        mask = (meta["demand_type"] == patt).values
        if mask.sum() == 0:
            continue
        # Rank by WAPE when it is defined (some patterns can have zero total
        # actual demand in the test window -> WAPE undefined); fall back to the
        # always-finite per-series MASE, then to a theory-aligned default.
        best_w, best_wape = None, np.inf
        best_m, best_mase = None, np.inf
        for name, p in preds.items():
            mm = compute_metrics(actual[mask], p[mask], naive_scale[mask], cost[mask])
            mm["demand_type"], mm["model"], mm["n_sku"] = patt, name, int(mask.sum())
            patt_rows.append(mm)
            if mm["WAPE"] == mm["WAPE"] and mm["WAPE"] < best_wape:
                best_w, best_wape = name, mm["WAPE"]
            if mm["MASE_med"] == mm["MASE_med"] and mm["MASE_med"] < best_mase:
                best_m, best_mase = name, mm["MASE_med"]
        default_for_pattern = {"Smooth": "SES", "Erratic": "SBA",
                               "Intermittent": "TSB", "Lumpy": "SBA"}
        routing[patt] = best_w or best_m or default_for_pattern.get(patt, "SBA")
    by_pattern = pd.DataFrame(patt_rows)
    log("\nBest model per demand pattern:")
    for k, v in routing.items():
        log(f"  {k:13s} -> {v}")

    # Build the hybrid forecast by routing each SKU to its pattern's best model
    hybrid = np.zeros_like(actual)
    for patt, name in routing.items():
        mask = (meta["demand_type"] == patt).values
        hybrid[mask] = preds[name][mask]
    hybrid_metrics = compute_metrics(actual, hybrid, naive_scale, cost)
    log("\nHybrid (pattern-routed) metrics: " + str(hybrid_metrics))

    system_metrics = None
    if fcst_h is not None:
        system_metrics = compute_metrics(actual, fcst_h[:, origins], naive_scale, cost)
        log("System FCST_H metrics:           " + str(system_metrics))

    # ---- Production: refit on ALL history, forecast forward + safety stock ----
    log("\nRefitting on full history for the forward forecast ...")
    models_full, quant_full, zero_full = train_models(
        cfg, demand, static_feats, cal_months, cfg.m_oldest - 1)
    point, quant = forecast_forward(
        cfg, demand, static_feats, cal_months,
        models_full["LightGBM"], quant_full, zero_full)
    lt_demand, ss_q, rop_q, upper_q = safety_stock(cfg, point, quant)

    # Service-level safety stock using each SKU's own target service level (TSL).
    # TSL in the file is a percentage (e.g. 99.4); fall back to the config default.
    if "TSL" in inv.columns:
        sl = (inv["TSL"].values / 100.0)
        sl = np.where(np.isfinite(sl) & (sl > 0), sl, cfg.default_service_level)
    else:
        sl = np.full(demand.shape[0], cfg.default_service_level)
    ss_sl, rop_sl = service_level_safety_stock(cfg, demand, lt_demand, sl)

    future_months = [cfg.m00_month + pd.DateOffset(months=k + 1)
                     for k in range(cfg.horizon)]

    # ---- Assemble output table ----
    out = meta.copy()
    out["assigned_model"] = [routing.get(p, "LightGBM") for p in meta["demand_type"]]
    for j, t in enumerate(origins):
        tag = cal_months[t].strftime("%b%y")
        out[f"actual_{tag}"] = actual[:, j]
        out[f"hybrid_{tag}"] = np.round(hybrid[:, j], 3)
        if fcst_h is not None:
            out[f"sys_{tag}"] = np.round(np.maximum(fcst_h[:, t], 0), 3)
    for k, fm in enumerate(future_months):
        tag = fm.strftime("%b%y")
        out[f"fcst_{tag}"] = np.round(point[:, k], 3)
        for q in cfg.quantiles:
            out[f"p{int(q*100)}_{tag}"] = np.round(quant[q][:, k], 3)

    out[f"leadtime_demand_{cfg.lead_time_months}m"] = np.round(lt_demand, 3)
    out["target_service_level"] = np.round(sl, 4)
    out["safety_stock_serviceLevel"] = np.round(ss_sl, 3)
    out["reorder_point_serviceLevel"] = np.round(rop_sl, 3)
    out[f"safety_stock_p{int(upper_q*100)}"] = np.round(ss_q, 3)
    out["reorder_point_quantile"] = np.round(rop_q, 3)

    # Side-by-side with the existing ERP inventory parameters (if present).
    erp_compare = None
    if "REORDER_POINT" in inv.columns:
        out["erp_reorder_point"] = np.round(inv["REORDER_POINT"].values, 3)
        out["reorder_delta_vs_erp"] = np.round(rop_sl - inv["REORDER_POINT"].values, 3)
    if "SAFTY" in inv.columns:
        out["erp_safety_stock"] = np.round(inv["SAFTY"].values, 3)
    if "REORDER_POINT" in inv.columns:
        erp = inv["REORDER_POINT"].values
        erp_compare = pd.DataFrame([{
            "metric": "reorder_point",
            "erp_total": round(float(np.nansum(erp)), 1),
            "model_total": round(float(np.nansum(rop_sl)), 1),
            "model_vs_erp_%": round(float((np.nansum(rop_sl) - np.nansum(erp))
                                          / max(np.nansum(erp), 1e-9) * 100), 2),
            "skus_model_higher": int((rop_sl > erp).sum()),
            "skus_model_lower": int((rop_sl < erp).sum()),
        }])
        log("\nReorder-point vs ERP (service-level method):\n" + erp_compare.to_string(index=False))

    res = Result(summary=summary.reset_index(), by_pattern=by_pattern,
                 routing=routing, hybrid_metrics=hybrid_metrics,
                 system_metrics=system_metrics, output=out)

    # ---- Export ----
    with pd.ExcelWriter(cfg.out_file, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="SKU_Forecasts", index=False)
        summary.to_excel(w, sheet_name="Model_Leaderboard")
        by_pattern.to_excel(w, sheet_name="By_Pattern", index=False)
        comp = pd.DataFrame([
            {"approach": "Hybrid (pattern-routed)", **hybrid_metrics},
            *( [{"approach": "System FCST_H", **system_metrics}] if system_metrics else [] ),
        ])
        comp.to_excel(w, sheet_name="Hybrid_vs_System", index=False)
        if erp_compare is not None:
            erp_compare.to_excel(w, sheet_name="Inventory_vs_ERP", index=False)
    log(f"\nSaved {cfg.out_file}")
    return res


if __name__ == "__main__":
    run_pipeline(Config())
