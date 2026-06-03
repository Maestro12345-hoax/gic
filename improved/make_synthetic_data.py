"""
Generate a synthetic IN00-style spare-parts dataset for testing the pipeline.

The real file (IN00_ST_DATA_3Years.xlsx) is not in the repo, so this produces a
file with the SAME schema and realistic intermittent demand so we can validate
the pipeline end-to-end. Replace with the real Excel for production runs.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_SKU = 600
N_MONTHS = 37          # M36 .. M00
M00 = pd.Timestamp("2026-05-01")

VOL = ["A", "B", "C", "D", "E"]
MOVE = ["A", "B", "C", "D", "E", "S0"]
FCST_MODELS = ["A1", "B1", "C1", "G1", "S0"]
BRANDS = ["Jaguar", "LandRover", "RangeRover", "Defender"]


def make_series(pattern, n=N_MONTHS):
    """Create one SKU's monthly demand following a demand-pattern archetype."""
    months = np.arange(n)
    season = 1 + 0.3 * np.sin(2 * np.pi * (months % 12) / 12)
    if pattern == "Smooth":
        base = RNG.uniform(20, 120)
        s = base * season * RNG.normal(1, 0.12, n)
    elif pattern == "Erratic":
        base = RNG.uniform(10, 60)
        s = base * season * RNG.normal(1, 0.6, n)
    elif pattern == "Intermittent":
        p = RNG.uniform(0.3, 0.6)
        size = RNG.uniform(2, 15)
        s = RNG.binomial(1, p, n) * size * RNG.normal(1, 0.2, n)
    else:  # Lumpy
        p = RNG.uniform(0.15, 0.4)
        s = RNG.binomial(1, p, n) * RNG.gamma(1.2, 12, n)
    return np.clip(np.round(s), 0, None)


rows = []
patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
for i in range(N_SKU):
    patt = RNG.choice(patterns, p=[0.15, 0.2, 0.35, 0.3])
    demand = make_series(patt)

    # A plausible (biased, laggy) "system" forecast = smoothed + noise + bias
    sysf = np.convolve(demand, np.ones(3) / 3, mode="same") * RNG.normal(1.1, 0.15, N_MONTHS)
    sysf = np.clip(np.round(sysf, 2), 0, None)
    # System future forecast: flat-ish continuation
    fut = np.clip(np.round(np.full(12, demand[-6:].mean()) * RNG.normal(1.1, 0.1, 12), 2), 0, None)

    row = {
        "PRODUCT": f"SKU{i:05d}",
        "FACING_LOC": "IN00",
        "VOL_CLASS": RNG.choice(VOL),
        "MOVE_CLASS": RNG.choice(MOVE),
        "FCST_MODEL": RNG.choice(FCST_MODELS),
        "BRAND": RNG.choice(BRANDS),
        "COST": round(float(RNG.uniform(5, 5000)), 2),
        "REP_RUN_DATE": pd.Timestamp("2026-05-12"),
    }
    # CAP_SCAL_M36 (oldest) .. M00 (newest). demand[0] is oldest.
    for k in range(N_MONTHS):
        suffix = N_MONTHS - 1 - k  # k=0 -> M36, k=36 -> M00
        name = f"CAP_SCAL_M{suffix:02d}"
        if suffix == 0:
            name = "CAP_SCAL_M00-MAY-26"   # exercise the date-suffix matcher
        elif suffix == 1:
            name = "CAP_SCAL_M01_APRIL26"
        row[name] = demand[k]
        row[f"FCST_H{suffix:02d}"] = sysf[k]
    for f in range(1, 13):
        row[f"FCST_F{f:02d}"] = fut[f - 1]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_excel("IN00_ST_DATA_3Years.xlsx", index=False)
print(f"Wrote IN00_ST_DATA_3Years.xlsx  ({df.shape[0]} SKUs, {df.shape[1]} cols)")
