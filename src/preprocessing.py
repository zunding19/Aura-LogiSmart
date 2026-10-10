"""Validate inputs without inventing telemetry or outcomes."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src.schema import (
    NUMERIC_COLUMNS,
    OUTCOME_COLUMNS,
    RAW_INPUT_COLUMNS,
    SHIPMENT_COLUMNS,
    TARGET_COLUMN,
    TRANSPORT_MODES,
    STORAGE_STANDARDS,
    URGENCY_TIERS,
)


class ShipmentValidationError(ValueError):
    pass


def _key(value, keep_minus=False):
    text = unicodedata.normalize(
        "NFKC", str(value)
    ).casefold()

    for dash in ("−", "–", "—"):
        text = text.replace(dash, "-")

    pattern = (
        r"[^a-z0-9-]+"
        if keep_minus
        else r"[^a-z0-9]+"
    )
    return re.sub(pattern, "", text)


ALIASES = {
    "Shipment_ID": [
        "shipment_identifier",
        "consignment_id",
    ],
    "Place_of_Origin": [
        "origin",
        "origin_location",
        "shipment_origin",
    ],
    "Place_of_Destination": [
        "destination",
        "destination_location",
        "shipment_destination",
    ],
    "Distance": [
        "distance_km",
        "route_distance_km",
        "distance_in_km",
    ],
    "Transport_Mode": [
        "transportation_mode",
        "shipping_mode",
    ],
    "Lead-Time in days": [
        "lead_time_days",
        "lead_days",
        "lead_time_in_days",
    ],
    "Transit_Duration": [
        "transit_duration_hours",
        "transit_hours",
        "estimated_transit_hours",
        "eta_hours",
    ],
    "Stock_Volume": [
        "shipment_volume",
        "volume",
    ],
    "Geopolitical_Risk": [
        "geopolitical_risk_score",
        "geo_risk",
    ],
    "Fuel_Price_Index": [
        "fuel_index",
    ],
    "Carrier_Reliability_Score": [
        "carrier_reliability",
        "reliability_score",
    ],
    "Buffer_of_Package": [
        "buffer_hours",
        "package_buffer_hours",
        "packaging_hold_hours",
        "total_hold_hours",
    ],
    "Storage_standard": [
        "storage_requirement",
        "storage_requirements",
    ],
    "Cargo_Value_USD": [
        "cargo_value",
        "shipment_value_usd",
    ],
    "Hospital_Urgency_Tier": [
        "urgency_tier",
        "hospital_urgency",
        "clinical_urgency",
    ],
    "Planned_Transit_Hours": [
        "planned_transit_duration_hours",
        "planned_hours",
    ],
    "Handover_Count": [
        "handover_count",
        "handovers",
    ],
    TARGET_COLUMN: [
        "failure",
        "failure_label",
        "observed_failure",
    ],
}

COLUMN_LOOKUP = {
    _key(alias): canonical
    for canonical in SHIPMENT_COLUMNS + OUTCOME_COLUMNS
    for alias in [
        canonical,
        *ALIASES.get(canonical, []),
    ]
}


def suggest_column_mapping(columns):
    return {
        str(column): COLUMN_LOOKUP[_key(column)]
        for column in columns
        if _key(column) in COLUMN_LOOKUP
    }


def normalize_column_names(df):
    if not isinstance(df, pd.DataFrame):
        raise ShipmentValidationError(
            "Provide a DataFrame or read a CSV "
            "with read_raw_shipments()."
        )

    if df.columns.duplicated().any():
        raise ShipmentValidationError(
            "Duplicate input headers; keep one column "
            "for each field."
        )

    result = df.rename(
        columns={
            column: COLUMN_LOOKUP.get(
                _key(column), column
            )
            for column in df
        }
    ).copy()

    duplicates = result.columns[
        result.columns.duplicated()
    ].tolist()

    if duplicates:
        raise ShipmentValidationError(
            "Multiple columns map to the same field: "
            f"{duplicates}."
        )

    return result


def _categories(names, extras, keep_minus=False):
    return {
        _key(alias, keep_minus): name
        for name in names
        for alias in [
            name,
            *extras.get(name, []),
        ]
    }


CATEGORY_LOOKUP = {
    "Transport_Mode": _categories(
        TRANSPORT_MODES,
        {
            "Air": [
                "air freight",
                "airfreight",
            ],
            "Land": [
                "road",
                "ground",
                "road freight",
            ],
            "Multi-Modal Air-to-Land": [
                "air to land",
                "multimodal",
                "multi modal",
                "air-land",
            ],
        },
    ),
    "Storage_standard": _categories(
        STORAGE_STANDARDS,
        {
            "2°C to 8°C": [
                "2-8 C",
                "2 to 8 C",
                "2-8°C",
                "2°C-8°C",
                "2 to 8°C",
                "refrigerated",
                "chilled",
            ],
            "-20°C": [
                "-20 C",
                "-20",
                "minus 20 C",
                "frozen -20 C",
                "negative 20 C",
            ],
            "Deep Frozen / Dry Ice": [
                "deep frozen",
                "dry ice",
                "deep-frozen dry-ice",
            ],
        },
        keep_minus=True,
    ),
    "Hospital_Urgency_Tier": _categories(
        URGENCY_TIERS,
        {
            "Routine Restock": [
                "routine",
                "restock",
            ],
            "Elevated Procedure": [
                "elevated",
            ],
            "Critical Emergency": [
                "critical",
                "emergency",
                "Critical Emergency / Direct Patient Treatment",
            ],
        },
    ),
}


def _numbers(series):
    unsupported = series.map(
        lambda value: isinstance(
            value,
            (
                bool,
                np.bool_,
                complex,
                np.complexfloating,
            ),
        )
    )

    return pd.to_numeric(
        series.where(~unsupported, np.nan),
        errors="coerce",
    ).astype(float)


def _rows(mask):
    positions = (
        np.flatnonzero(
            mask.to_numpy(dtype=bool)
        )[:5] + 1
    )
    return ", ".join(map(str, positions))


def clean_raw_shipment_data(
    df,
    require_target=False,
    require_processed=True,
):
    """Transit is TOTAL-trip ETA; buffer is TOTAL hold capacity."""
    result = normalize_column_names(df)

    required = list(
        SHIPMENT_COLUMNS
        if require_processed
        else RAW_INPUT_COLUMNS
    )

    if require_target:
        required.append(TARGET_COLUMN)

    missing = [
        column
        for column in required
        if column not in result
    ]

    if missing:
        raise ShipmentValidationError(
            "Missing required shipment fields: "
            + ", ".join(missing)
            + ". Check schema and units."
        )

    if result.empty:
        raise ShipmentValidationError(
            "No shipment rows were supplied."
        )

    errors = []

    for column in [
        "Shipment_ID",
        "Place_of_Origin",
        "Place_of_Destination",
    ]:
        invalid = (
            result[column].isna()
            | result[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if invalid.any():
            errors.append(
                f"{column}: missing text "
                f"at data rows {_rows(invalid)}"
            )

        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

    numeric_fields = [
        column
        for column in NUMERIC_COLUMNS + [
            "Actual_Transit_Hours"
        ]
        if column in result
    ]

    for column in numeric_fields:
        values = _numbers(result[column])

        invalid = ~pd.Series(
            np.isfinite(values.to_numpy()),
            index=result.index,
        )

        if column in {
            "Lead-Time in days",
            "Fuel_Price_Index",
            "Planned_Transit_Hours",
        }:
            invalid |= values <= 0
            rule = "finite and greater than zero"

        elif column in {
            "Geopolitical_Risk",
            "Carrier_Reliability_Score",
        }:
            upper = (
                10
                if column == "Geopolitical_Risk"
                else 1
            )
            invalid |= (
                (values < 0)
                | (values > upper)
            )
            rule = f"finite, between 0 and {upper}"

        else:
            invalid |= values < 0
            rule = "finite and nonnegative"

        if column in {
            "Stock_Volume",
            "Handover_Count",
        }:
            invalid |= (
                (values % 1 != 0)
                | (values >= float(2**63))
            )
            rule += ", whole integers below 2**63"

        if invalid.any():
            errors.append(
                f"{column}: expected {rule}; "
                f"invalid data rows {_rows(invalid)}"
            )

        if (
            column in {"Stock_Volume", "Handover_Count"}
            and not invalid.any()
        ):
            result[column] = values.astype("int64")
        else:
            result[column] = values

    for column, lookup in CATEGORY_LOOKUP.items():
        values = result[column].map(
            lambda value: lookup.get(
                _key(
                    value,
                    column == "Storage_standard",
                )
            )
        )
        invalid = values.isna()

        if invalid.any():
            allowed = list(
                dict.fromkeys(lookup.values())
            )
            errors.append(
                f"{column}: expected one of {allowed}; "
                f"invalid data rows {_rows(invalid)}"
            )

        result[column] = values

    binary_fields = [
        column
        for column in OUTCOME_COLUMNS
        if (
            column != "Actual_Transit_Hours"
            and column in result
        )
    ]

    for column in binary_fields:
        values = _numbers(result[column])
        invalid = ~values.isin([0, 1])

        if invalid.any():
            errors.append(
                f"{column}: expected supplied binary labels "
                f"0/1 at rows {_rows(invalid)}; "
                "never invent missing outcomes"
            )
        else:
            result[column] = values.astype(int)

    if errors:
        raise ShipmentValidationError(
            "Shipment validation failed:\n- "
            + "\n- ".join(errors)
        )

    return result