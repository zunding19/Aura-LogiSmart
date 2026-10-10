"""Keep supplied disruption labels; add optional, explicitly simulated context."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean_dataset import clean, RAW_PATH, OUT_PATH as CLEAN_PATH
from src.schema import (
    CONTEXT_COLUMNS,
    DATASET_VERSION,
    CONTEXT_SOURCE,
    TARGET_COLUMN,
    OUTCOME_COLUMNS,
)

OUT_PATH = ROOT / "data" / "processed" / "aura_shipments.csv"
SEED = 42
TARGET = TARGET_COLUMN
SIMULATED_COLUMNS = CONTEXT_COLUMNS

STORAGE_MIX = {
    "Pharmaceuticals": {
        "2°C to 8°C": 0.65,
        "-20°C": 0.25,
        "Deep Frozen / Dry Ice": 0.10,
    },
    "Perishables": {"2°C to 8°C": 0.60, "-20°C": 0.40},
}
PASSIVE_CLASSES_HOURS = np.array([24, 48, 72, 96, 120])
REEFER_HOLD_HOURS = {
    "2°C to 8°C": (1.0, 2.0),
    "-20°C": (6.0, 10.0),
    "Deep Frozen / Dry Ice": (72.0, 100.0),
}
# Load weights are whole container loads, so the value per shipment is the
# reference value per kg scaled down. Air loads are assumed twice as valuable
# as surface loads of the same weight.
REFERENCE_VALUE_USD_PER_KG = {
    "Pharmaceuticals": 148.0,
    "Electronics": 481.0,
    "Automotive": 12.0,
    "Textiles": 10.0,
    "Perishables": 4.0,
}
VALUE_SCALE_DIVISOR = {"Air": 25.0, "Surface": 50.0}
URGENCY_TIERS = [
    "Routine Restock", "Elevated Procedure", "Critical Emergency",
]
URGENCY_MIX = {
    "Air": [0.35, 0.40, 0.25],
    "Surface": [0.70, 0.25, 0.05],
}

CONTEXT_PROVENANCE = {
    "Elapsed_Transit_Hours_Simulated": (
        "Assumed 10%-95% of planned lead time; "
        "not recorded elapsed telemetry or an ETA."
    ),
    "Storage_Standard_Simulated": (
        "Assumed seeded mix by product; "
        "Ambient for non-temperature-sensitive categories."
    ),
    "Remaining_Buffer_Hours_Simulated": (
        "Assumed passive-air hold minus simulated elapsed time, "
        "or surface hold after power loss; blank for Ambient. "
        "Not measured temperature."
    ),
    "Cargo_Value_USD_Simulated": (
        "Assumed value/kg times recorded metric tonnes times 1000 "
        "with seeded variation; divided by 25 for air and 50 for surface "
        "so a shipment is worth thousands to a few million USD. "
        "Not measured cargo value."
    ),
    "Hospital_Urgency_Tier_Simulated": (
        "Assumed seeded mix for Pharmaceuticals only; "
        "Not Applicable otherwise. Not a recorded clinical condition."
    ),
}


def _pick(rng, mix, size):
    return rng.choice(
        list(mix),
        size=size,
        p=list(mix.values()),
    )


def build(raw, seed=SEED):
    """Context is target-independent; modes, weights and labels survive."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("Context seed must be a nonnegative integer.")

    df = clean(raw)
    rng = np.random.default_rng(seed)
    n = len(df)

    by_air = df["Transport_Mode"].eq("Air").to_numpy()
    product = df["Product_Category"].to_numpy()
    lead_hours = df["Lead_Time_Days"].to_numpy() * 24

    elapsed = lead_hours * rng.uniform(0.10, 0.95, n)
    df["Elapsed_Transit_Hours_Simulated"] = elapsed

    storage = np.full(n, "Ambient", dtype=object)
    for category, mix in STORAGE_MIX.items():
        rows = product == category
        storage[rows] = _pick(rng, mix, rows.sum())

    df["Storage_Standard_Simulated"] = storage
    cold = storage != "Ambient"

    index = np.searchsorted(
        PASSIVE_CLASSES_HOURS,
        lead_hours * 1.2,
    ).clip(max=len(PASSIVE_CLASSES_HOURS) - 1)

    passive_left = (
        PASSIVE_CLASSES_HOURS[index] - elapsed
    ).clip(min=0)

    reefer_left = np.zeros(n)
    for standard, (low, high) in REEFER_HOLD_HOURS.items():
        rows = storage == standard
        reefer_left[rows] = rng.uniform(low, high, rows.sum())

    buffer = np.where(by_air, passive_left, reefer_left)
    df["Remaining_Buffer_Hours_Simulated"] = np.where(
        cold, buffer, np.nan
    )

    per_kg = df["Product_Category"].map(
        REFERENCE_VALUE_USD_PER_KG
    ).to_numpy()
    per_kg = np.where(
        by_air,
        per_kg / VALUE_SCALE_DIVISOR["Air"],
        per_kg / VALUE_SCALE_DIVISOR["Surface"],
    )
    spread = rng.lognormal(mean=0.0, sigma=0.35, size=n)

    df["Cargo_Value_USD_Simulated"] = (
        df["Weight_MT"].to_numpy() * 1000 * per_kg * spread
    )

    urgency = np.full(n, "Not Applicable", dtype=object)
    pharma = product == "Pharmaceuticals"

    for rows, shares in [
        (pharma & by_air, URGENCY_MIX["Air"]),
        (pharma & ~by_air, URGENCY_MIX["Surface"]),
    ]:
        urgency[rows] = rng.choice(
            URGENCY_TIERS,
            size=rows.sum(),
            p=shares,
        )

    df["Hospital_Urgency_Tier_Simulated"] = urgency

    df = df.round({
        "Elapsed_Transit_Hours_Simulated": 1,
        "Remaining_Buffer_Hours_Simulated": 1,
        "Cargo_Value_USD_Simulated": 0,
    })

    df["Context_Source"] = CONTEXT_SOURCE
    df["Context_Provenance"] = json.dumps(
        CONTEXT_PROVENANCE,
        ensure_ascii=False,
        sort_keys=True,
    )
    df["Context_Seed"] = int(seed)
    df["Dataset_Version"] = DATASET_VERSION
    df["Weight_MT_Unit"] = "metric tonnes"

    return df


if __name__ == "__main__":
    from src.data_loader import read_raw_shipments

    recorded = clean(read_raw_shipments(RAW_PATH))
    result = build(recorded)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recorded.to_csv(
        CLEAN_PATH, index=False, date_format="%Y-%m-%d"
    )
    result.to_csv(
        OUT_PATH, index=False, date_format="%Y-%m-%d"
    )

    print(
        f"Wrote {len(result)} rows; supplied disruption rate: "
        f"{result[TARGET].mean():.1%}"
    )
    print(
        "Simulated display-only context:",
        ", ".join(CONTEXT_COLUMNS),
    )