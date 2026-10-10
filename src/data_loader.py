"""CSV loading with preserved identifiers and explicit provenance."""
from __future__ import annotations

import csv
import hashlib
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
    OUTCOME_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]

DEMO_PATH = (
    ROOT / "data" / "processed" / "aura_shipments.csv"
)

SYNTHETIC_SOURCE = (
    "Synthetic shipments with simulated delivery outcomes"
)

# Exact fingerprint of the original uploaded processed CSV.
LEGACY_PROCESSED_SHA256 = (
    "7374ba5fb21ef732e802e1f8ac3b5de9372673def03fd45305df44d54a5b71a4"
)

SAMPLE_SHIPMENT = {
    "Shipment_ID": "SC-10000",
    "Place_of_Origin": (
        "Singapore Pharmaceutical Logistics Depot"
    ),
    "Place_of_Destination": (
        "Los Angeles Medical Centre Depot"
    ),
    "Distance": 14869.0,
    "Transport_Mode": "Multi-Modal Air-to-Land",
    "Lead-Time in days": 3.12,
    "Transit_Duration": 39.6,
    "Stock_Volume": 6241,
    "Geopolitical_Risk": 5.0,
    "Fuel_Price_Index": 2.43,
    "Carrier_Reliability_Score": 0.865,
    "Buffer_of_Package": 67.9,
    "Storage_standard": "2°C to 8°C",
    "Cargo_Value_USD": 28618.15,
    "Hospital_Urgency_Tier": "Routine Restock",
    "Planned_Transit_Hours": 38.0,
    "Handover_Count": 5,
}


def read_raw_shipments(
    source,
    column_mapping=None,
):
    """Preserve identifiers, literal NA, blanks and upload position."""
    is_path = isinstance(
        source, (str, os.PathLike)
    )
    position = None

    try:
        if is_path:
            path = Path(source).expanduser()

            if path.suffix.lower() != ".csv":
                raise ShipmentValidationError(
                    "Export the dataset as a UTF-8 .csv file."
                )

            payload = path.read_bytes()

        else:
            if not hasattr(source, "read"):
                raise ShipmentValidationError(
                    "Supply a CSV path "
                    "or text/binary upload object."
                )

            if (
                hasattr(source, "seek")
                and hasattr(source, "tell")
            ):
                try:
                    position = source.tell()
                    source.seek(0)
                except (OSError, ValueError):
                    position = None

            payload = source.read()

        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        if not isinstance(payload, bytes):
            raise ShipmentValidationError(
                "CSV content must be UTF-8 text or bytes."
            )

        digest = hashlib.sha256(
            payload
        ).hexdigest()

        stream = io.StringIO(
            payload.decode("utf-8-sig")
        )

        header = next(
            (
                row
                for row in csv.reader(
                    stream, strict=True
                )
                if row
            ),
            None,
        )

        if not header:
            raise ShipmentValidationError(
                "The CSV is empty; "
                "include headers and shipment rows."
            )

        duplicates = [
            name
            for name, count in Counter(header).items()
            if count > 1
        ]

        if duplicates:
            raise ShipmentValidationError(
                f"Duplicate CSV headers: {duplicates}."
            )

        stream.seek(0)

        raw = pd.read_csv(
            stream,
            dtype=str,
            keep_default_na=False,
        )

    except ShipmentValidationError:
        raise

    except (
        OSError,
        UnicodeError,
        ValueError,
        csv.Error,
    ) as exc:
        raise ShipmentValidationError(
            f"Cannot read CSV: {exc}. "
            "Check UTF-8 encoding, path, "
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
                "Column mapping must map source headers "
                "to canonical field names."
            )

        for old, new in column_mapping.items():
            if (
                not isinstance(old, str)
                or not isinstance(new, str)
                or old not in raw
            ):
                raise ShipmentValidationError(
                    "Invalid or missing mapping source: "
                    f"{old!r}."
                )

            if new not in (
                SHIPMENT_COLUMNS + OUTCOME_COLUMNS
            ):
                raise ShipmentValidationError(
                    "Unknown canonical mapping target: "
                    f"{new!r}."
                )

        raw = raw.rename(
            columns=dict(column_mapping)
        )

    raw = normalize_column_names(
        raw.replace(
            r"^\s*$", pd.NA, regex=True
        )
    )

    marked = pd.Series(
        digest == LEGACY_PROCESSED_SHA256,
        index=raw.index,
    )

    for column in [
        "Place_of_Origin",
        "Place_of_Destination",
    ]:
        if column in raw:
            marked |= (
                raw[column]
                .astype("string")
                .str.contains(
                    "(synthetic)",
                    case=False,
                    regex=False,
                    na=False,
                )
            )

    if marked.any():
        if "Data_Source" not in raw:
            raw["Data_Source"] = pd.NA

        raw.loc[
            marked, "Data_Source"
        ] = SYNTHETIC_SOURCE

    return raw


def load_shipments(
    source,
    column_mapping=None,
    require_target=False,
):
    return clean_raw_shipment_data(
        read_raw_shipments(
            source,
            column_mapping,
        ),
        require_target=require_target,
    )


def generate_demo_shipments(
    n=400,
    seed=42,
):
    """Sample the bundled dataset without a second simulation."""
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
    ):
        raise ValueError(
            "n must be a positive integer."
        )

    data = load_shipments(
        DEMO_PATH,
        require_target=True,
    )

    if n > len(data):
        raise ValueError(
            f"n must be between 1 and {len(data)}; "
            "demo sampling is without replacement."
        )

    positions = np.random.default_rng(
        seed
    ).choice(
        len(data),
        size=n,
        replace=False,
    )

    return data.iloc[
        positions
    ].reset_index(drop=True)


__all__ = [
    "SAMPLE_SHIPMENT",
    "read_raw_shipments",
    "load_shipments",
    "generate_demo_shipments",
    "suggest_column_mapping",
]