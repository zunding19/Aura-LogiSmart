"""Clean the raw supply chain risk file without changing any recorded value.

Run from the project root:
    python src/clean_dataset.py

Reads  data/raw/global_supply_chain_risk_2026.csv
Writes data/processed/supply_chain_risk_clean.csv

What it does:
1. Checks the file is complete: no blanks, no duplicate rows or IDs.
2. Fixes types: Date becomes a real date, text is trimmed, numbers are numbers.
3. Checks every value is in its allowed range or category list.
4. Adds flag columns for rows that are physically doubtful. No row is deleted
   and no original value is rewritten, so every decision can be undone later.

The target column is Disruption_Occurred. It came with the file.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "global_supply_chain_risk_2026.csv"
OUT_PATH = ROOT / "data" / "processed" / "supply_chain_risk_clean.csv"

TARGET = "Disruption_Occurred"
TEXT_COLUMNS = [
    "Shipment_ID",
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Weather_Condition",
]
NUMERIC_COLUMNS = [
    "Distance_km",
    "Weight_MT",
    "Fuel_Price_Index",
    "Geopolitical_Risk_Score",
    "Carrier_Reliability_Score",
    "Lead_Time_Days",
]

# Allowed values. Anything else stops the script with a clear message.
ALLOWED = {
    "Transport_Mode": {"Air", "Sea", "Road", "Rail"},
    "Product_Category": {
        "Electronics",
        "Textiles",
        "Perishables",
        "Pharmaceuticals",
        "Automotive",
    },
    "Weather_Condition": {"Clear", "Rain", "Fog", "Storm", "Hurricane"},
}
# (lowest, highest) allowed for each number
RANGES = {
    "Distance_km": (0, 25_000),
    "Weight_MT": (0, 10_000),
    "Fuel_Price_Index": (0, 10),
    "Geopolitical_Risk_Score": (0, 10),
    "Carrier_Reliability_Score": (0, 1),
    "Lead_Time_Days": (0, 365),
}

# Fastest believable door-to-door speed for each mode, in km/h
SPEED_LIMIT_KMH = {"Air": 950, "Rail": 120, "Road": 100, "Sea": 50}
LEAD_TIME_MINIMUM_DAYS = 0.5  # the file never goes below this value

FLAG_COLUMNS = [
    "Flag_Land_Route_Across_Ocean",
    "Flag_Lead_Time_At_Minimum",
    "Flag_Speed_Impossible",
]


def check(df: pd.DataFrame) -> None:
    """Stop with a clear message if the file breaks a basic rule."""
    problems = []
    if df.isna().any().any():
        problems.append(f"{int(df.isna().sum().sum())} missing or unreadable values")
    if df.duplicated().any():
        problems.append(f"{int(df.duplicated().sum())} duplicate rows")
    if df["Shipment_ID"].duplicated().any():
        problems.append("duplicate Shipment_ID values")
    if (df["Origin_Port"] == df["Destination_Port"]).any():
        problems.append("rows where origin and destination are the same")
    for column, allowed in ALLOWED.items():
        unknown = set(df[column]) - allowed
        if unknown:
            problems.append(f"{column} has unknown values {sorted(unknown)}")
    for column, (low, high) in RANGES.items():
        outside = ~df[column].between(low, high)
        if outside.any():
            problems.append(f"{column} has {int(outside.sum())} values outside {low} to {high}")
    if not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET} has values other than 0 and 1")
    if problems:
        raise ValueError("Raw file failed checks: " + "; ".join(problems))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # --- Types -------------------------------------------------------------
    for column in TEXT_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    check(df)
    df[TARGET] = df[TARGET].astype(int)

    # --- Flags: doubtful rows are marked, never removed --------------------
    by_land = df["Transport_Mode"].isin(["Road", "Rail"])
    touches_los_angeles = (df["Origin_Port"] == "Los Angeles") | (
        df["Destination_Port"] == "Los Angeles"
    )
    # Every other port is across an ocean from Los Angeles.
    df["Flag_Land_Route_Across_Ocean"] = (by_land & touches_los_angeles).astype(int)

    # 0.5 days is a floor in the file, so the true lead time may be shorter.
    df["Flag_Lead_Time_At_Minimum"] = (
        df["Lead_Time_Days"] <= LEAD_TIME_MINIMUM_DAYS
    ).astype(int)

    speed_kmh = df["Distance_km"] / (df["Lead_Time_Days"] * 24)
    df["Flag_Speed_Impossible"] = (
        speed_kmh > df["Transport_Mode"].map(SPEED_LIMIT_KMH)
    ).astype(int)

    df["Needs_Review"] = df[FLAG_COLUMNS].max(axis=1)

    return df.sort_values("Shipment_ID").reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False, encoding="utf-8", date_format="%Y-%m-%d")

    print(f"Read  {len(raw)} rows from {RAW_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(cleaned)} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Disruption rate: {cleaned[TARGET].mean():.1%}")
    for column in FLAG_COLUMNS + ["Needs_Review"]:
        print(f"{column}: {int(cleaned[column].sum())} rows")
