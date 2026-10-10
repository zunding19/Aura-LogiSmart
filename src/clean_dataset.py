"""Preserve recorded values and labels; quality flags never delete rows."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_raw_shipment_data
from src.schema import (
    FLAG_COLUMNS,
    TARGET_COLUMN,
    DATA_SOURCE,
    LABEL_PROVENANCE,
)

RAW_PATH = ROOT / "data" / "raw" / "global_supply_chain_risk_2026.csv"
OUT_PATH = ROOT / "data" / "processed" / "supply_chain_risk_clean.csv"
TARGET = TARGET_COLUMN

SPEED_LIMIT_KMH = {"Air": 950, "Rail": 120, "Road": 100, "Sea": 50}
LEAD_TIME_MINIMUM_DAYS = 0.5


def check(df):
    clean_raw_shipment_data(df, require_target=True)

    if df.duplicated().any() or df["Shipment_ID"].duplicated().any():
        raise ValueError(
            "Duplicate shipment rows or identifiers; review the raw dataset."
        )

    if df["Origin_Port"].eq(df["Destination_Port"]).any():
        raise ValueError(
            "Origin and destination coincide; review the recorded route."
        )


def clean(raw):
    df = clean_raw_shipment_data(raw, require_target=True)
    check(df)

    land = df["Transport_Mode"].isin(["Road", "Rail"])
    touches_la = (
        df["Origin_Port"].eq("Los Angeles")
        | df["Destination_Port"].eq("Los Angeles")
    )
    df["Flag_Land_Route_Across_Ocean"] = (
        land & touches_la
    ).astype(int)

    df["Flag_Lead_Time_At_Minimum"] = (
        df["Lead_Time_Days"].le(LEAD_TIME_MINIMUM_DAYS)
    ).astype(int)

    speed = df["Distance_km"] / (df["Lead_Time_Days"] * 24)
    df["Flag_Speed_Impossible"] = speed.gt(
        df["Transport_Mode"].map(SPEED_LIMIT_KMH)
    ).astype(int)

    df["Needs_Review"] = df[FLAG_COLUMNS].max(axis=1)

    # Review heuristics, not proof of an error; recorded values are preserved.
    if "Data_Source" not in df:
        df["Data_Source"] = DATA_SOURCE
    if "Label_Provenance" not in df:
        df["Label_Provenance"] = LABEL_PROVENANCE

    return df.sort_values("Shipment_ID").reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import read_raw_shipments

    result = clean(read_raw_shipments(RAW_PATH))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Wrote {len(result)} rows; supplied disruption rate: "
        f"{result[TARGET].mean():.1%}"
    )
    print({
        column: int(result[column].sum())
        for column in FLAG_COLUMNS + ["Needs_Review"]
    })