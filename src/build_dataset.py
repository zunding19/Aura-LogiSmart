"""Build the Aura training file from the raw supply chain risk file.

Run from the project root:
    python src/build_dataset.py

Reads  data/raw/global_supply_chain_risk_2026.csv
Writes data/processed/supply_chain_risk_clean.csv   (recorded columns + quality flags)
Writes data/processed/aura_shipments.csv            (the file to train on)

Recorded in the raw file and kept unchanged:
    Shipment_ID, Date, origin, destination, Distance_km, Transport_Mode,
    Product_Category, Weather_Condition, Lead_Time_Days, Fuel_Price_Index,
    Geopolitical_Risk_Score, Carrier_Reliability_Score, and the target
    Disruption_Occurred. Weight_MT is kept as Stock_Volume (metric tonnes).

Simulated here, because the raw file does not have them:
    Transit_Duration_Hours, Storage_standard, Buffer_of_Package,
    Cargo_Value_USD, Hospital_Urgency_Tier.

Rules for the simulated columns:
- They are built only from recorded inputs plus seeded randomness.
- They never read Disruption_Occurred, so they cannot leak the answer.
- Each range below names its source, or says plainly that it is an assumption.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_dataset import FLAG_COLUMNS, RAW_PATH, TARGET, clean
from clean_dataset import OUT_PATH as CLEAN_PATH

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "aura_shipments.csv"
SEED = 42

# --- Storage standard ------------------------------------------------------
# Only temperature-sensitive goods get a cold-chain standard.
# Pharmaceuticals: 2-8°C is 62% to 70% of the pharma cold chain market
# (Meticulous Research and PMarketResearch, 2025). The split of the remainder
# between -20°C and deep frozen is an assumption.
# Perishables: the chilled / frozen split is an assumption.
STORAGE_MIX = {
    "Pharmaceuticals": {"2°C to 8°C": 0.65, "-20°C": 0.25, "Deep Frozen / Dry Ice": 0.10},
    "Perishables": {"2°C to 8°C": 0.60, "-20°C": 0.40},
}
AMBIENT = "Ambient"

# --- Packaging hold time ---------------------------------------------------
# Air freight uses passive insulated shippers, pre-qualified for a set number
# of hours: 24, 48, 72, 96 "and more" (FedEx healthcare packaging; ThermoSafe
# E48 is rated 48 hours). 120 is the top class used here.
PASSIVE_CLASSES_HOURS = np.array([24, 48, 72, 96, 120])
# Sea, rail and road use powered refrigerated containers. The buffer is how
# long the cargo stays in range if power is lost: about 1 to 2 hours for
# chilled cargo and about 8 hours for frozen (PIL reefer cargo handling guide).
# Dry-ice units hold for up to about 100 hours (Wikipedia, Refrigerated container).
REEFER_HOLD_HOURS = {
    "2°C to 8°C": (1.0, 2.0),
    "-20°C": (6.0, 10.0),
    "Deep Frozen / Dry Ice": (72.0, 100.0),
}

# --- Cargo value -----------------------------------------------------------
# USD per kg when flown. Pharmaceuticals and Electronics are from US air export
# values per ton (US Bureau of Transportation Statistics, 2000 report: about
# $148,000 and $481,000 per ton). The table is old, so read these as rough
# sizes. The other three have no source and are assumptions.
AIR_VALUE_USD_PER_KG = {
    "Pharmaceuticals": 148.0,
    "Electronics": 481.0,
    "Automotive": 12.0,
    "Textiles": 10.0,
    "Perishables": 4.0,
}
# Goods sent by surface are far less valuable per kg than goods that are flown.
# The factor of 50 is an assumption.
SURFACE_VALUE_DIVISOR = 50.0

# --- Hospital urgency ------------------------------------------------------
# Only pharmaceuticals go to hospitals. Urgent orders are more likely to fly.
# All shares are assumptions.
URGENCY_TIERS = ["Routine Restock", "Elevated Procedure", "Critical Emergency"]
URGENCY_MIX = {"Air": [0.35, 0.40, 0.25], "Surface": [0.70, 0.25, 0.05]}
NOT_APPLICABLE = "Not Applicable"

COLUMN_ORDER = [
    # Identifiers and route
    "Shipment_ID",
    "Date",
    "Place_of_Origin",
    "Place_of_Destination",
    "Distance_km",
    # Operations and risk
    "Transport_Mode",
    "Lead_Time_Days",
    "Transit_Duration_Hours",
    "Stock_Volume",
    "Geopolitical_Risk_Score",
    "Fuel_Price_Index",
    "Carrier_Reliability_Score",
    "Weather_Condition",
    "Product_Category",
    # Cold-chain physics
    "Buffer_of_Package",
    "Storage_standard",
    # Financial and urgency
    "Cargo_Value_USD",
    "Hospital_Urgency_Tier",
    # Quality flags and target
    *FLAG_COLUMNS,
    "Needs_Review",
    TARGET,
]
SIMULATED_COLUMNS = [
    "Transit_Duration_Hours",
    "Storage_standard",
    "Buffer_of_Package",
    "Cargo_Value_USD",
    "Hospital_Urgency_Tier",
]


def _pick(rng: np.random.Generator, mix: dict, size: int) -> np.ndarray:
    return rng.choice(list(mix), size=size, p=list(mix.values()))


def build(cleaned: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(cleaned)
    df = cleaned.rename(
        columns={
            "Origin_Port": "Place_of_Origin",
            "Destination_Port": "Place_of_Destination",
            "Weight_MT": "Stock_Volume",  # recorded weight in metric tonnes
        }
    ).copy()

    mode = df["Transport_Mode"].to_numpy()
    product = df["Product_Category"].to_numpy()
    by_air = mode == "Air"
    lead_hours = df["Lead_Time_Days"].to_numpy() * 24

    # --- Transit duration: hours on the road so far -------------------------
    # Each row is a shipment caught part-way through its journey, somewhere
    # between 10% and 95% of its lead time. It can never exceed the lead time.
    progress = rng.uniform(0.10, 0.95, n)
    elapsed = lead_hours * progress
    df["Transit_Duration_Hours"] = elapsed

    # --- Storage standard, from what the cargo is ---------------------------
    storage = np.full(n, AMBIENT, dtype=object)
    for category, mix in STORAGE_MIX.items():
        rows = product == category
        storage[rows] = _pick(rng, mix, rows.sum())
    df["Storage_standard"] = storage
    cold = storage != AMBIENT

    # --- Buffer: thermal hold hours left ------------------------------------
    # Air: smallest passive class that covers the lead time with a 20% margin,
    # minus the hours already used. Zero means the packaging is exhausted.
    wanted = lead_hours * 1.2
    index = np.searchsorted(PASSIVE_CLASSES_HOURS, wanted).clip(
        max=len(PASSIVE_CLASSES_HOURS) - 1
    )
    passive_left = (PASSIVE_CLASSES_HOURS[index] - elapsed).clip(min=0)
    # Surface: hold time of a refrigerated container if its power is lost.
    reefer_left = np.zeros(n)
    for standard, (low, high) in REEFER_HOLD_HOURS.items():
        rows = storage == standard
        reefer_left[rows] = rng.uniform(low, high, rows.sum())
    buffer = np.where(by_air, passive_left, reefer_left)
    df["Buffer_of_Package"] = np.where(cold, buffer, np.nan)  # blank for ambient cargo

    # --- Cargo value: weight times value per kg -----------------------------
    per_kg = df["Product_Category"].map(AIR_VALUE_USD_PER_KG).to_numpy()
    per_kg = np.where(by_air, per_kg, per_kg / SURFACE_VALUE_DIVISOR)
    spread = rng.lognormal(mean=0.0, sigma=0.35, size=n)  # shipments differ in value
    df["Cargo_Value_USD"] = df["Stock_Volume"].to_numpy() * 1000 * per_kg * spread

    # --- Hospital urgency: pharmaceuticals only -----------------------------
    urgency = np.full(n, NOT_APPLICABLE, dtype=object)
    pharma = product == "Pharmaceuticals"
    for rows, shares in [
        (pharma & by_air, URGENCY_MIX["Air"]),
        (pharma & ~by_air, URGENCY_MIX["Surface"]),
    ]:
        urgency[rows] = rng.choice(URGENCY_TIERS, size=rows.sum(), p=shares)
    df["Hospital_Urgency_Tier"] = urgency

    rounding = {"Transit_Duration_Hours": 1, "Buffer_of_Package": 1, "Cargo_Value_USD": 0}
    return df.round(rounding)[COLUMN_ORDER]


if __name__ == "__main__":
    cleaned = clean(pd.read_csv(RAW_PATH))
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False, encoding="utf-8", date_format="%Y-%m-%d")

    shipments = build(cleaned)
    shipments.to_csv(OUT_PATH, index=False, encoding="utf-8", date_format="%Y-%m-%d")

    print(f"Wrote {len(cleaned)} rows to {CLEAN_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(shipments)} rows and {shipments.shape[1]} columns to {OUT_PATH.relative_to(ROOT)}")
    print(f"Disruption rate: {shipments[TARGET].mean():.1%}")
    print(f"Simulated columns: {', '.join(SIMULATED_COLUMNS)}")
