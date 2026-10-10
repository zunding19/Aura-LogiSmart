"""Reproduce the shipment simulation and label it explicitly."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    SYNTHETIC_SOURCE,
    read_raw_shipments,
)
from src.preprocessing import (
    clean_raw_shipment_data,
)
from src.schema import (
    DATASET_VERSION,
    OUTCOME_COLUMNS,
)


RAW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "global_supply_chain_pharma.csv"
)

OUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "aura_shipments.csv"
)

SEED = 42

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

EUROPE = {
    "Rotterdam",
    "Antwerp",
    "Hamburg",
    "Marseille",
}

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


def _city(place):
    return place.str.replace(
        r"\s+(Pharmaceutical Logistics|Medical Centre) Depot.*$",
        "",
        regex=True,
    )


def _great_circle_km(origin, dest):
    unknown = (
        set(origin) | set(dest)
    ) - set(CITY_COORDS)

    if unknown:
        raise ValueError(
            "Unknown simulation cities: "
            f"{sorted(unknown)}. "
            "Add authoritative coordinates before rebuilding."
        )

    lat1, lon1 = np.radians(
        np.array([
            CITY_COORDS[city]
            for city in origin
        ]).T
    )

    lat2, lon2 = np.radians(
        np.array([
            CITY_COORDS[city]
            for city in dest
        ]).T
    )

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    return (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )


def build(raw, seed=SEED):
    """Retain simulation equations and RNG order."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError(
            "Simulation seed must be "
            "a nonnegative integer."
        )

    df = clean_raw_shipment_data(
        raw,
        require_processed=False,
    ).rename(
        columns={
            "Distance": "Distance_km",
            "Lead-Time in days": "Lead_Time_Days",
            "Transit_Duration": "Transit_Duration_Hours",
            "Geopolitical_Risk": "Geopolitical_Risk_Score",
        }
    )

    rng = np.random.default_rng(seed)
    n = len(df)

    origin = _city(df["Place_of_Origin"])
    dest = _city(df["Place_of_Destination"])

    df["Place_of_Origin"] = (
        df["Place_of_Origin"].str.replace(
            " (synthetic)",
            "",
            regex=False,
        )
    )

    df["Place_of_Destination"] = (
        df["Place_of_Destination"].str.replace(
            " (synthetic)",
            "",
            regex=False,
        )
    )

    air_km = _great_circle_km(
        origin, dest
    )

    by_road = (
        origin.isin(EUROPE)
        & dest.isin(EUROPE)
    ).to_numpy()

    mode = (
        df["Transport_Mode"]
        .to_numpy()
        .copy()
    )

    mode[
        ~by_road & (mode == "Land")
    ] = "Multi-Modal Air-to-Land"

    mode[
        by_road & (air_km < 600)
    ] = "Land"

    df["Transport_Mode"] = mode

    is_land = mode == "Land"
    is_multi = (
        mode == "Multi-Modal Air-to-Land"
    )
    connection = (
        (air_km > 9500) & ~is_land
    )

    flight_h = (
        air_km * 1.03 / 820
        + 1.5
        + np.where(
            connection, 5.0, 0.0
        )
    )

    last_mile_h = np.where(
        is_multi,
        rng.uniform(3, 9, n),
        0.0,
    )

    air_plan_h = (
        flight_h
        + 5.0
        + last_mile_h
        + np.where(
            is_multi, 2.0, 0.0
        )
    )

    road_km = air_km * 1.3
    drive_h = road_km / 55

    land_plan_h = (
        drive_h
        + 2.0
        + 9.0 * np.floor(drive_h / 9)
    )

    planned_h = (
        np.where(
            is_land,
            land_plan_h,
            air_plan_h,
        )
        * rng.uniform(
            0.95, 1.08, n
        )
    )

    df["Planned_Transit_Hours"] = (
        planned_h
    )

    df["Distance_km"] = np.where(
        is_land,
        road_km,
        air_km * 1.03
        + last_mile_h * 45,
    )

    df["Handover_Count"] = np.select(
        [is_land, is_multi],
        [
            rng.integers(1, 3, n),
            rng.integers(3, 5, n)
            + connection,
        ],
        default=2 + connection,
    )

    tier = (
        df["Hospital_Urgency_Tier"]
        .to_numpy()
    )

    slack = np.select(
        [
            tier == "Critical Emergency",
            tier == "Elevated Procedure",
        ],
        [
            rng.uniform(1.15, 1.40, n),
            rng.uniform(1.25, 1.60, n),
        ],
        default=rng.uniform(
            1.40, 2.00, n
        ),
    )

    df["Lead_Time_Days"] = (
        planned_h * slack / 24
    )

    geo = (
        df["Geopolitical_Risk_Score"]
        .to_numpy() / 10
    )

    unreliable = (
        1
        - df["Carrier_Reliability_Score"]
        .to_numpy()
    )

    known_delay = rng.exponential(
        0.02
        + 0.05 * geo
        + 0.20 * unreliable
    )

    df["Transit_Duration_Hours"] = (
        planned_h * (1 + known_delay)
    )

    classes = np.array([
        24, 48, 72, 96, 120
    ])

    wanted = (
        planned_h
        * rng.uniform(1.4, 2.2, n)
    )

    idx = np.searchsorted(
        classes, wanted
    ).clip(
        max=len(classes) - 1
    )

    idx = np.where(
        rng.random(n) < 0.07,
        idx - 1,
        idx,
    ).clip(min=0)

    storage = (
        df["Storage_standard"]
        .to_numpy()
    )

    hold_factor = np.select(
        [
            storage == "Deep Frozen / Dry Ice",
            storage == "-20°C",
        ],
        [0.85, 0.92],
        1.0,
    )

    df["Buffer_of_Package"] = (
        classes[idx]
        * hold_factor
        * rng.uniform(
            0.90, 1.00, n
        )
    )

    handovers = (
        df["Handover_Count"]
        .to_numpy()
    )

    extra_h = (
        planned_h
        * rng.exponential(
            0.01
            + 0.03 * geo
            + 0.12 * unreliable
        )
    )

    extra_h += rng.gamma(
        handovers,
        0.2 + 0.5 * unreliable,
    )

    disrupted = (
        rng.random(n)
        < (
            0.01
            + 0.03 * geo
            + 0.08 * unreliable
        )
    )

    extra_h += (
        disrupted
        * rng.uniform(8, 40, n)
    )

    actual_h = (
        df["Transit_Duration_Hours"]
        .to_numpy()
        + extra_h
    )

    exposure = (
        rng.uniform(0, 0.04, n)
        * handovers
    )

    df["Actual_Transit_Hours"] = (
        actual_h
    )

    df["Late_Delivery"] = (
        actual_h
        > df["Lead_Time_Days"] * 24
    ).astype(int)

    df["Temperature_Excursion"] = (
        actual_h
        > df["Buffer_of_Package"]
        * (1 - exposure)
    ).astype(int)

    df["Handling_Damage"] = (
        rng.random(n)
        < 0.005 + 0.003 * handovers
    ).astype(int)

    df["Shipment_Failure"] = df[
        [
            "Late_Delivery",
            "Temperature_Excursion",
            "Handling_Damage",
        ]
    ].max(axis=1)

    rounding = {
        "Distance_km": 0,
        "Planned_Transit_Hours": 1,
        "Transit_Duration_Hours": 1,
        "Actual_Transit_Hours": 1,
        "Lead_Time_Days": 2,
        "Buffer_of_Package": 1,
    }

    result = df.round(
        rounding
    )[
        INPUT_COLUMNS + OUTCOME_COLUMNS
    ].copy()

    result["Data_Source"] = (
        SYNTHETIC_SOURCE
    )
    result["Simulation_Seed"] = int(seed)
    result["Dataset_Version"] = (
        DATASET_VERSION
    )

    return result


if __name__ == "__main__":
    result = build(
        read_raw_shipments(RAW_PATH)
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Wrote {len(result)} synthetic rows "
        f"to {OUT_PATH.relative_to(ROOT)}"
    )
    print(
        "Simulated failure rate: "
        f"{result['Shipment_Failure'].mean():.1%}"
    )