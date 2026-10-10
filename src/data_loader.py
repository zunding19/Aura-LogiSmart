from __future__ import annotations

import csv
import io
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    TRANSPORT_MODES,
    STORAGE_STANDARDS,
    URGENCY_TIERS,
)


SAMPLE_SHIPMENT = {
    "Shipment_ID": "SYNTHETIC-BIO-0001",
    "Place_of_Origin": "Demo Frankfurt CargoCity",
    "Place_of_Destination": "Demo Kuala Lumpur Hospital",
    "Distance": 9850.0,
    "Transport_Mode": "Multi-Modal Air-to-Land",
    "Lead-Time in days": 3.0,
    "Transit_Duration": 17.5,
    "Stock_Volume": 1200.0,
    "Geopolitical_Risk": 6.5,
    "Fuel_Price_Index": 1.15,
    "Carrier_Reliability_Score": 0.42,
    "Buffer_of_Package": 2.8,
    "Storage_standard": "2°C to 8°C",
    "Cargo_Value_USD": 85000.0,
    "Hospital_Urgency_Tier": "Critical Emergency",
}


def generate_demo_shipments(
    n: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
    ):
        raise ValueError("n must be a positive integer.")

    rng = np.random.default_rng(seed)
    lead_days = rng.uniform(0.5, 5.0, n)
    elapsed_ratio = rng.uniform(0.1, 1.3, n)
    transit = lead_days * 24 * elapsed_ratio
    remaining = np.maximum(lead_days * 24 - transit, 0)

    buffer = rng.uniform(0.2, 96, n)
    low_buffer = rng.random(n) < 0.2
    buffer[low_buffer] = rng.uniform(
        0.1, 5.0, int(low_buffer.sum())
    )

    reliability = rng.uniform(0.3, 0.99, n)
    geo_risk = rng.uniform(0, 10, n)
    modes = rng.choice(
        TRANSPORT_MODES, n, p=[0.35, 0.25, 0.4]
    )
    fuel = rng.uniform(0.8, 1.8, n)

    margin_deficit = np.clip(
        (remaining - buffer) / 48, -1.5, 1.5
    )
    latent = (
        -2.3
        + 1.35 * margin_deficit
        + 2.5 * (elapsed_ratio - 0.65)
        + 3.1 * (0.75 - reliability)
        + 0.1 * geo_risk
        + 0.35 * (modes == "Multi-Modal Air-to-Land")
        + 0.2 * (fuel - 1.0)
        + 1.3 * (buffer < 3.0)
    )
    synthetic_probability = 1.0 / (1.0 + np.exp(-latent))
    observed_synthetic_labels = rng.binomial(
        1, synthetic_probability
    )

    origins = np.array([
        "Demo Frankfurt Hub",
        "Demo Singapore Hub",
        "Demo Tokyo Hub",
        "Demo Penang Depot",
    ])
    destinations = np.array([
        "Demo Kuala Lumpur Hospital",
        "Demo Johor Hospital",
        "Demo Ipoh Clinic",
        "Demo Selangor Hospital",
    ])

    result = pd.DataFrame({
        "Shipment_ID": [
            f"SYNTHETIC-{int(seed)}-{i + 1:05d}"
            for i in range(n)
        ],
        "Place_of_Origin": rng.choice(origins, n),
        "Place_of_Destination": rng.choice(destinations, n),
        "Distance": rng.uniform(50, 12000, n).round(1),
        "Transport_Mode": modes,
        "Lead-Time in days": lead_days.round(4),
        "Transit_Duration": transit.round(3),
        "Stock_Volume": rng.integers(20, 2500, n),
        "Geopolitical_Risk": geo_risk.round(3),
        "Fuel_Price_Index": fuel.round(3),
        "Carrier_Reliability_Score": reliability.round(4),
        "Buffer_of_Package": buffer.round(3),
        "Storage_standard": rng.choice(STORAGE_STANDARDS, n),
        "Cargo_Value_USD": rng.uniform(
            2500, 150000, n
        ).round(2),
        "Hospital_Urgency_Tier": rng.choice(
            URGENCY_TIERS, n, p=[0.45, 0.35, 0.2]
        ),
        TARGET_COLUMN: observed_synthetic_labels,
        "Data_Source": (
            "Synthetic demonstration data — "
            "not real shipment records"
        ),
    })

    return clean_raw_shipment_data(
        result, require_target=True
    )


def read_raw_shipments(
    source: Any,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    is_path = isinstance(source, (str, os.PathLike))

    if is_path:
        path = Path(source).expanduser()
        if not path.is_file():
            raise ShipmentValidationError(
                f"CSV file does not exist: {path}. "
                "Check the dataset path."
            )
        if path.suffix.casefold() != ".csv":
            raise ShipmentValidationError(
                "Please provide a CSV file. "
                "Export spreadsheet data to .csv first."
            )
    elif not hasattr(source, "read"):
        raise ShipmentValidationError(
            "Provide a .csv filesystem path "
            "or an uploaded/file-like CSV object."
        )

    original_position = None
    if (
        not is_path
        and hasattr(source, "seek")
        and hasattr(source, "tell")
    ):
        try:
            original_position = source.tell()
            source.seek(0)
        except (OSError, ValueError):
            original_position = None

    try:
        if is_path:
            with path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                raw = _read_csv_stream(stream)
        else:
            payload = source.read()
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8-sig")
            if not isinstance(payload, str):
                raise ShipmentValidationError(
                    "The uploaded CSV must contain "
                    "UTF-8 text or bytes."
                )
            raw = _read_csv_stream(
                io.StringIO(payload.removeprefix("\ufeff"))
            )
    except ShipmentValidationError:
        raise
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeError,
        OSError,
        ValueError,
        csv.Error,
    ) as exc:
        raise ShipmentValidationError(
            f"Could not read the CSV: {exc}. "
            "Use a UTF-8 CSV with a header row "
            "and consistent comma-separated fields."
        ) from exc
    finally:
        if original_position is not None:
            try:
                source.seek(original_position)
            except (OSError, ValueError):
                pass

    if column_mapping is not None:
        if not isinstance(column_mapping, Mapping):
            raise ShipmentValidationError(
                "Column mapping must be an object mapping "
                "source headers to canonical field names."
            )

        allowed_targets = set(
            SHIPMENT_COLUMNS + [TARGET_COLUMN]
        )

        for source_name, target_name in column_mapping.items():
            if (
                not isinstance(source_name, str)
                or not isinstance(target_name, str)
            ):
                raise ShipmentValidationError(
                    "Column mapping keys and values "
                    "must be text column names."
                )
            if source_name not in raw.columns:
                raise ShipmentValidationError(
                    f"Mapped source column {source_name!r} "
                    "is missing from this CSV."
                )
            if target_name not in allowed_targets:
                raise ShipmentValidationError(
                    f"Unknown mapping target {target_name!r}. "
                    "Use the canonical shipment schema."
                )

        raw = raw.rename(columns=dict(column_mapping))

    return normalize_column_names(
        raw.replace(r"^\s*$", pd.NA, regex=True)
    )


def load_shipments(
    source: Any,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    return clean_raw_shipment_data(
        read_raw_shipments(
            source, column_mapping=column_mapping
        )
    )


def _read_csv_stream(stream: Any) -> pd.DataFrame:
    reader = csv.reader(stream, strict=True)
    header = next((row for row in reader if row), None)

    if header is None:
        raise ShipmentValidationError(
            "The CSV is empty. "
            "Supply a header row and at least one shipment."
        )

    duplicates = [
        name
        for name, count in Counter(header).items()
        if count > 1
    ]
    if duplicates:
        raise ShipmentValidationError(
            f"Duplicate CSV columns: {duplicates}. "
            "Keep one source column per field."
        )

    stream.seek(0)
    return pd.read_csv(
        stream, dtype=str, keep_default_na=False
    )


__all__ = [
    "SAMPLE_SHIPMENT",
    "generate_demo_shipments",
    "read_raw_shipments",
    "load_shipments",
    "suggest_column_mapping",
]