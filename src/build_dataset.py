"""Clean the raw shipment file and add simulated outcomes.

Run from the project root:
    python src/build_dataset.py

Reads  data/raw/global_supply_chain_pharma.csv
Writes data/processed/aura_shipments.csv

Kept from the raw file: Shipment_ID, origin, destination, Stock_Volume,
Geopolitical_Risk_Score, Fuel_Price_Index, Carrier_Reliability_Score,
Storage_standard, Cargo_Value_USD, Hospital_Urgency_Tier, and Transport_Mode
where the route allows it.

Rebuilt because the raw values were physically impossible: Distance_km,
Lead_Time_Days, Transit_Duration_Hours, Buffer_of_Package.

Added: Planned_Transit_Hours, Handover_Count, and the outcome columns
(Actual_Transit_Hours, Late_Delivery, Temperature_Excursion, Handling_Damage,
Shipment_Failure). Outcomes are SIMULATED. They are only known after delivery,
so never feed them to the model as inputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "global_supply_chain_pharma.csv"
OUT_PATH = ROOT / "data" / "processed" / "aura_shipments.csv"
SEED = 42

# City -> (latitude, longitude)
CITY_COORDS = {
    "Singapore": (1.35, 103.99),
    "Rotterdam": (51.92, 4.48),
    "Busan": (35.18, 129.08),
    "Shanghai": (31.23, 121.47),
    "Antwerp": (51.22, 4.40),
    "Dubai": (25.25, 55.36),
    "Los Angeles": (33.94, -118.41),
    "Hamburg": (53.55, 9.99),
    "Marseille": (43.30, 5.37),
}
# The only cities joined by road in this network.
EUROPE = {"Rotterdam", "Antwerp", "Hamburg", "Marseille"}

OUTCOME_COLUMNS = [
    "Actual_Transit_Hours",
    "Late_Delivery",
    "Temperature_Excursion",
    "Handling_Damage",
    "Shipment_Failure",
]

# Model inputs, known while the shipment is still moving.
INPUT_COLUMNS = [
    "Shipment_ID",
    "Place_of_Origin",
    "Place_of_Destination",
    "Distance_km",
    "Transport_Mode",
    "Handover_Count",
    "Lead_Time_Days",
    "Planned_Transit_Hours",
    "Transit_Duration_Hours",
    "Stock_Volume",
    "Geopolitical_Risk_Score",
    "Fuel_Price_Index",
    "Carrier_Reliability_Score",
    "Buffer_of_Package",
    "Storage_standard",
    "Cargo_Value_USD",
    "Hospital_Urgency_Tier",
]


def _city(place: pd.Series) -> pd.Series:
    """'Los Angeles Medical Centre Depot (synthetic)' -> 'Los Angeles'."""
    return place.str.replace(
        r"\s+(Pharmaceutical Logistics|Medical Centre) Depot.*$", "", regex=True
    )


def _great_circle_km(origin: pd.Series, dest: pd.Series) -> np.ndarray:
    lat1, lon1 = np.radians(np.array([CITY_COORDS[c] for c in origin]).T)
    lat2, lon2 = np.radians(np.array([CITY_COORDS[c] for c in dest]).T)
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build(raw: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(raw)
    df = raw.copy()

    # --- Tidy labels -------------------------------------------------------
    origin, dest = _city(df["Place_of_Origin"]), _city(df["Place_of_Destination"])
    df["Place_of_Origin"] = df["Place_of_Origin"].str.replace(" (synthetic)", "")
    df["Place_of_Destination"] = df["Place_of_Destination"].str.replace(
        " (synthetic)", ""
    )
    df["Hospital_Urgency_Tier"] = df["Hospital_Urgency_Tier"].str.replace(
        " / Direct Patient Treatment", ""
    )

    # --- Route: one real distance per city pair, road only inside Europe ----
    air_km = _great_circle_km(origin, dest)
    by_road = (origin.isin(EUROPE) & dest.isin(EUROPE)).to_numpy()
    mode = df["Transport_Mode"].to_numpy().copy()
    mode[~by_road & (mode == "Land")] = "Multi-Modal Air-to-Land"
    mode[by_road & (air_km < 600)] = "Land"  # nobody flies Antwerp to Rotterdam
    df["Transport_Mode"] = mode
    is_land, is_multi = mode == "Land", mode == "Multi-Modal Air-to-Land"

    # --- Planned schedule from distance and mode ---------------------------
    connection = (air_km > 9500) & ~is_land  # long haul needs one stopover
    flight_h = air_km * 1.03 / 820 + 1.5 + np.where(connection, 5.0, 0.0)
    last_mile_h = np.where(is_multi, rng.uniform(3, 9, n), 0.0)
    air_plan_h = flight_h + 5.0 + last_mile_h + np.where(is_multi, 2.0, 0.0)

    road_km = air_km * 1.3
    drive_h = road_km / 55
    land_plan_h = drive_h + 2.0 + 9.0 * np.floor(drive_h / 9)  # driver rest stops

    planned_h = np.where(is_land, land_plan_h, air_plan_h) * rng.uniform(0.95, 1.08, n)
    df["Planned_Transit_Hours"] = planned_h
    df["Distance_km"] = np.where(is_land, road_km, air_km * 1.03 + last_mile_h * 45)
    df["Handover_Count"] = np.select(
        [is_land, is_multi],
        [rng.integers(1, 3, n), rng.integers(3, 5, n) + connection],
        default=2 + connection,
    )

    # --- Delivery window: tighter for urgent orders -------------------------
    tier = df["Hospital_Urgency_Tier"].to_numpy()
    slack = np.select(
        [tier == "Critical Emergency", tier == "Elevated Procedure"],
        [rng.uniform(1.15, 1.40, n), rng.uniform(1.25, 1.60, n)],
        default=rng.uniform(1.40, 2.00, n),
    )
    df["Lead_Time_Days"] = planned_h * slack / 24

    # --- Current ETA: plan plus the delay already known ---------------------
    geo = df["Geopolitical_Risk_Score"].to_numpy() / 10
    unreliable = 1 - df["Carrier_Reliability_Score"].to_numpy()
    known_delay = rng.exponential(0.02 + 0.05 * geo + 0.20 * unreliable)
    df["Transit_Duration_Hours"] = planned_h * (1 + known_delay)

    # --- Packaging hold time, chosen for the planned trip -------------------
    classes = np.array([24, 48, 72, 96, 120])
    wanted = planned_h * rng.uniform(1.4, 2.2, n)
    idx = np.searchsorted(classes, wanted).clip(max=len(classes) - 1)
    idx = np.where(rng.random(n) < 0.07, idx - 1, idx).clip(min=0)  # under-specified
    storage = df["Storage_standard"].to_numpy()
    hold_factor = np.select(  # dry ice and frozen packs lose hold time faster
        [storage == "Deep Frozen / Dry Ice", storage == "-20°C"], [0.85, 0.92], 1.0
    )
    df["Buffer_of_Package"] = classes[idx] * hold_factor * rng.uniform(0.90, 1.00, n)

    # --- Simulated outcomes (what happened after the ETA was set) ----------
    handovers = df["Handover_Count"].to_numpy()
    extra_h = planned_h * rng.exponential(0.01 + 0.03 * geo + 0.12 * unreliable)
    extra_h += rng.gamma(handovers, 0.2 + 0.5 * unreliable)  # dwell at each handover
    disrupted = rng.random(n) < 0.01 + 0.03 * geo + 0.08 * unreliable
    extra_h += disrupted * rng.uniform(8, 40, n)  # strike, customs hold, missed flight
    actual_h = df["Transit_Duration_Hours"].to_numpy() + extra_h

    exposure = rng.uniform(0, 0.04, n) * handovers  # hold time lost on the tarmac
    df["Actual_Transit_Hours"] = actual_h
    df["Late_Delivery"] = (actual_h > df["Lead_Time_Days"] * 24).astype(int)
    df["Temperature_Excursion"] = (
        actual_h > df["Buffer_of_Package"] * (1 - exposure)
    ).astype(int)
    df["Handling_Damage"] = (rng.random(n) < 0.005 + 0.003 * handovers).astype(int)
    df["Shipment_Failure"] = df[
        ["Late_Delivery", "Temperature_Excursion", "Handling_Damage"]
    ].max(axis=1)

    rounding = {
        "Distance_km": 0,
        "Planned_Transit_Hours": 1,
        "Transit_Duration_Hours": 1,
        "Actual_Transit_Hours": 1,
        "Lead_Time_Days": 2,
        "Buffer_of_Package": 1,
    }
    return df.round(rounding)[INPUT_COLUMNS + OUTCOME_COLUMNS]


if __name__ == "__main__":
    clean = build(pd.read_csv(RAW_PATH))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(clean)} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Failure rate: {clean['Shipment_Failure'].mean():.1%}")
