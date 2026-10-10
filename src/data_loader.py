"""Read CSVs safely; supplied disruption labels retain unverified provenance."""
from __future__ import annotations

import csv
import io
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
    normalize_column_names,
    suggest_column_mapping,
)
from src.schema import (
    SHIPMENT_COLUMNS,
    TARGET_COLUMN,
    DATA_SOURCE,
    LABEL_PROVENANCE,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "data" / "processed" / "aura_shipments.csv"

SAMPLE_SHIPMENT = {
    "Shipment_ID": "SC-10000",
    "Date": "2025-10-16",
    "Origin_Port": "Singapore",
    "Destination_Port": "Los Angeles",
    "Transport_Mode": "Rail",
    "Product_Category": "Textiles",
    "Distance_km": 5930.83,
    "Weight_MT": 197.42,
    "Fuel_Price_Index": 2.43,
    "Geopolitical_Risk_Score": 5.0,
    "Weather_Condition": "Hurricane",
    "Carrier_Reliability_Score": 0.865,
    "Lead_Time_Days": 41.39,
}


def read_raw_shipments(source, column_mapping=None):
    """Preserve leading zeros, literal NA, blank missingness and upload position."""
    position = None

    try:
        if isinstance(source, (str, os.PathLike)):
            path = Path(source).expanduser()
            if path.suffix.lower() != ".csv":
                raise ShipmentValidationError(
                    "Export data as a UTF-8 .csv file."
                )
            payload = path.read_bytes()
        else:
            if not hasattr(source, "read"):
                raise ShipmentValidationError(
                    "Supply a CSV path or text/binary upload object."
                )
            if hasattr(source, "seek") and hasattr(source, "tell"):
                try:
                    position = source.tell()
                    source.seek(0)
                except (OSError, ValueError):
                    position = None
            payload = source.read()

        if isinstance(payload, bytes):
            text = payload.decode("utf-8-sig")
        elif isinstance(payload, str):
            text = payload.removeprefix("\ufeff")
        else:
            raise ShipmentValidationError(
                "CSV content must be UTF-8 text or bytes."
            )

        for number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith(("<<<<<<<", "=======", ">>>>>>>")):
                raise ShipmentValidationError(
                    f"Unresolved Git merge marker at line {number}; "
                    "regenerate the CSV from the valid raw file."
                )

        stream = io.StringIO(text)
        reader = csv.reader(stream, strict=True)
        header = next((row for row in reader if row), None)

        if not header:
            raise ShipmentValidationError(
                "CSV is empty; include headers and shipment rows."
            )

        duplicates = [
            name for name, count in Counter(header).items() if count > 1
        ]
        if duplicates:
            raise ShipmentValidationError(
                f"Duplicate CSV headers: {duplicates}."
            )

        for number, row in enumerate(reader, 2):
            if row and len(row) != len(header):
                raise ShipmentValidationError(
                    f"CSV record {number} has {len(row)} fields; "
                    f"expected {len(header)}."
                )

        stream.seek(0)
        raw = pd.read_csv(stream, dtype=str, keep_default_na=False)

    except ShipmentValidationError:
        raise
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        raise ShipmentValidationError(
            f"Cannot read CSV: {exc}. Check path, UTF-8 encoding "
            "and comma-separated fields."
        ) from exc
    finally:
        if position is not None:
            try:
                source.seek(position)
            except (OSError, ValueError):
                pass

    if column_mapping is not None:
        if not isinstance(column_mapping, Mapping):
            raise ShipmentValidationError(
                "Column mapping must map source headers to canonical fields."
            )

        for old, new in column_mapping.items():
            if (
                not isinstance(old, str)
                or not isinstance(new, str)
                or old not in raw
            ):
                raise ShipmentValidationError(
                    f"Invalid or missing mapping source: {old!r}."
                )
            if new not in SHIPMENT_COLUMNS + [TARGET_COLUMN]:
                raise ShipmentValidationError(
                    f"Unknown canonical mapping target: {new!r}."
                )

        raw = raw.rename(columns=dict(column_mapping))

    raw = normalize_column_names(
        raw.replace(r"^\s*$", pd.NA, regex=True)
    )

    fallbacks = {
        "Data_Source": DATA_SOURCE,
        "Label_Provenance": (
            LABEL_PROVENANCE
            if TARGET_COLUMN in raw
            else "No outcome labels supplied; inference only"
        ),
    }
    for column, fallback in fallbacks.items():
        raw[column] = (
            raw[column].fillna(fallback)
            if column in raw
            else fallback
        )

    return raw


def load_shipments(source, column_mapping=None, require_target=False):
    return clean_raw_shipment_data(
        read_raw_shipments(source, column_mapping),
        require_target=require_target,
    )


def generate_demo_shipments(n=400, seed=42):
    """Sample the rebuilt dataset; this does not generate or alter labels."""
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
    ):
        raise ValueError("n must be a positive integer.")

    data = load_shipments(DEMO_PATH, require_target=True)
    if n > len(data):
        raise ValueError(
            f"n must be between 1 and {len(data)}; "
            "sampling is without replacement."
        )

    positions = np.random.default_rng(seed).choice(
        len(data), size=n, replace=False
    )
    return data.iloc[positions].reset_index(drop=True)


__all__ = [
    "SAMPLE_SHIPMENT",
    "read_raw_shipments",
    "load_shipments",
    "generate_demo_shipments",
    "suggest_column_mapping",
]