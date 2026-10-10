from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.schema import (
    NUMERIC_COLUMNS,
    SHIPMENT_COLUMNS,
    STORAGE_STANDARDS,
    TARGET_COLUMN,
    TRANSPORT_MODES,
    URGENCY_TIERS,
)


class ShipmentValidationError(ValueError):
    pass


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub("[^a-z0-9]+", "", text)


_COLUMN_ALIASES = {
    "Shipment_ID": [
        "shipment id", "shipment_identifier", "consignment_id"
    ],
    "Place_of_Origin": [
        "origin", "origin_location", "shipment_origin"
    ],
    "Place_of_Destination": [
        "destination", "destination_location", "shipment_destination"
    ],
    "Distance": [
        "distance_km", "route_distance_km", "distance_in_km"
    ],
    "Transport_Mode": [
        "transport mode", "transportation_mode", "shipping_mode"
    ],
    "Lead-Time in days": [
        "lead_time_days", "lead_days",
        "lead_time_in_days", "leadtime_days"
    ],
    "Transit_Duration": [
        "transit_duration_hours", "transit_hours",
        "elapsed_transit_hours"
    ],
    "Stock_Volume": [
        "stock_volume", "shipment_volume", "volume"
    ],
    "Geopolitical_Risk": [
        "geopolitical_risk_score", "geo_risk"
    ],
    "Fuel_Price_Index": [
        "fuel_index", "fuel_price_index"
    ],
    "Carrier_Reliability_Score": [
        "carrier_reliability", "reliability_score"
    ],
    "Buffer_of_Package": [
        "buffer_hours", "package_buffer_hours",
        "remaining_buffer_hours", "packaging_hold_hours"
    ],
    "Storage_standard": [
        "storage_standard", "storage_requirement",
        "storage_requirements"
    ],
    "Cargo_Value_USD": [
        "cargo_value", "cargo_value_usd", "shipment_value_usd"
    ],
    "Hospital_Urgency_Tier": [
        "urgency_tier", "hospital_urgency", "clinical_urgency"
    ],
    TARGET_COLUMN: [
        "shipment_failure", "failure",
        "failure_label", "observed_failure"
    ],
}

_ALIAS_LOOKUP = {
    _normalized_name(alias): canonical
    for canonical, aliases in _COLUMN_ALIASES.items()
    for alias in [canonical, *aliases]
}


def suggest_column_mapping(
    columns: Iterable[object],
) -> dict[str, str]:
    return {
        str(column): _ALIAS_LOOKUP[_normalized_name(column)]
        for column in columns
        if _normalized_name(column) in _ALIAS_LOOKUP
    }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ShipmentValidationError(
            "Provide shipments as a pandas DataFrame "
            "or load a CSV with load_shipments()."
        )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ShipmentValidationError(
            f"Duplicate CSV columns: {duplicates}. "
            "Rename or remove duplicated headers."
        )

    renamed = df.rename(
        columns={
            column: _ALIAS_LOOKUP.get(
                _normalized_name(column), column
            )
            for column in df.columns
        }
    ).copy()

    if renamed.columns.duplicated().any():
        duplicates = renamed.columns[
            renamed.columns.duplicated()
        ].tolist()
        raise ShipmentValidationError(
            f"Multiple columns map to the same field: {duplicates}. "
            "Keep one source column for each field."
        )

    return renamed


def _normalized_category(
    value: object,
    preserve_minus: bool = False,
) -> str:
    if not preserve_minus:
        return _normalized_name(value)

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    for dash in ["−", "–", "—"]:
        text = text.replace(dash, "-")
    return re.sub("[^a-z0-9-]+", "", text)


def _category_aliases(
    canonical: list[str],
    extras: dict[str, list[str]],
    preserve_minus: bool = False,
) -> dict[str, str]:
    return {
        _normalized_category(
            alias, preserve_minus=preserve_minus
        ): name
        for name in canonical
        for alias in [name, *extras.get(name, [])]
    }


_CATEGORY_LOOKUPS = {
    "Transport_Mode": _category_aliases(
        TRANSPORT_MODES,
        {
            "Air": ["air freight", "airfreight"],
            "Land": ["road", "ground", "road freight"],
            "Multi-Modal Air-to-Land": [
                "air to land", "multimodal",
                "multi modal", "air-land"
            ],
        },
    ),
    "Storage_standard": _category_aliases(
        STORAGE_STANDARDS,
        {
            "2°C to 8°C": [
                "2-8 C", "2 to 8 C", "2-8°C",
                "2°C-8°C", "2 to 8°C", "refrigerated", "chilled"
            ],
            "-20°C": [
                "-20 C", "-20", "minus 20 C",
                "frozen -20 C", "negative 20 C"
            ],
            "Deep Frozen / Dry Ice": [
                "deep frozen", "dry ice", "deep-frozen dry-ice"
            ],
        },
        preserve_minus=True,
    ),
    "Hospital_Urgency_Tier": _category_aliases(
        URGENCY_TIERS,
        {
            "Routine Restock": ["routine", "restock"],
            "Elevated Procedure": ["elevated"],
            "Critical Emergency": ["critical", "emergency"],
        },
    ),
}


def _bad_rows(mask: pd.Series) -> str:
    positions = np.flatnonzero(
        np.asarray(mask, dtype=bool)
    )[:5] + 1
    suffix = " ..." if int(mask.sum()) > 5 else ""
    return ", ".join(
        str(int(position)) for position in positions
    ) + suffix


def clean_raw_shipment_data(
    df: pd.DataFrame,
    require_target: bool = False,
) -> pd.DataFrame:
    cleaned = normalize_column_names(df)
    required = SHIPMENT_COLUMNS + (
        [TARGET_COLUMN] if require_target else []
    )
    missing = [
        column for column in required
        if column not in cleaned.columns
    ]

    if missing:
        raise ShipmentValidationError(
            "Missing required shipment fields: "
            + ", ".join(missing)
            + ". Map your CSV headers to the documented schema; "
            "supply actual values with the documented units."
        )

    if cleaned.empty:
        raise ShipmentValidationError(
            "No shipment rows were supplied. "
            "Upload a CSV containing at least one shipment."
        )

    problems = []

    for column in [
        "Shipment_ID", "Place_of_Origin", "Place_of_Destination"
    ]:
        missing_text = (
            cleaned[column].isna()
            | cleaned[column].astype(str).str.strip().eq("")
        )
        if missing_text.any():
            problems.append(
                f"{column}: missing text at data rows "
                f"{_bad_rows(missing_text)}"
            )
        cleaned[column] = (
            cleaned[column].astype("string").str.strip()
        )

    for column in NUMERIC_COLUMNS:
        unsupported = cleaned[column].map(
            lambda value: isinstance(
                value,
                (bool, np.bool_, complex, np.complexfloating),
            )
        )
        values = pd.to_numeric(
            cleaned[column].where(~unsupported, np.nan),
            errors="coerce",
        )
        finite = pd.Series(
            np.isfinite(
                values.to_numpy(dtype=float, na_value=np.nan)
            ),
            index=cleaned.index,
        )
        invalid = ~finite

        if column in {"Lead-Time in days", "Fuel_Price_Index"}:
            invalid |= values <= 0
            expected = "a finite number greater than 0"
        elif column == "Geopolitical_Risk":
            invalid |= (values < 0) | (values > 10)
            expected = "a finite score from 0 to 10"
        elif column == "Carrier_Reliability_Score":
            invalid |= (values < 0) | (values > 1)
            expected = "a finite score from 0 to 1"
        else:
            invalid |= values < 0
            expected = "a finite number greater than or equal to 0"

        if invalid.any():
            problems.append(
                f"{column}: expected {expected}; "
                f"invalid data rows {_bad_rows(invalid)}"
            )

        cleaned[column] = values.astype(float)

    for column, lookup in _CATEGORY_LOOKUPS.items():
        mapped = cleaned[column].map(
            lambda value: lookup.get(
                _normalized_category(
                    value,
                    preserve_minus=column == "Storage_standard",
                )
            )
        )
        invalid = mapped.isna()

        if invalid.any():
            allowed = list(dict.fromkeys(lookup.values()))
            problems.append(
                f"{column}: choose one of {allowed}; "
                f"invalid data rows {_bad_rows(invalid)}"
            )

        cleaned[column] = mapped

    if TARGET_COLUMN in cleaned.columns:
        target = pd.to_numeric(
            cleaned[TARGET_COLUMN], errors="coerce"
        )
        invalid = ~target.isin([0, 1])

        if invalid.any():
            problems.append(
                f"{TARGET_COLUMN}: expected observed binary labels "
                f"0 or 1; invalid data rows {_bad_rows(invalid)}. "
                "For inference-only data, remove the target column "
                "rather than inventing labels"
            )
        else:
            cleaned[TARGET_COLUMN] = target.astype(int)

    if problems:
        raise ShipmentValidationError(
            "Shipment validation failed:\n- "
            + "\n- ".join(problems)
        )

    return cleaned