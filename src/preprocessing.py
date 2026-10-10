"""Normalize headers and validate recorded fields without filling telemetry."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.schema import (
    SHIPMENT_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    FLAG_COLUMNS,
    TRANSPORT_MODES,
    PRODUCT_CATEGORIES,
    WEATHER_CONDITIONS,
    RAW_MODEL_INPUT_COLUMNS,
    RAW_MODEL_NUMERIC_COLUMNS,
    TYPICAL_PACE,
)


class ShipmentValidationError(ValueError):
    pass


def _key(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        unicodedata.normalize("NFKC", str(value)).casefold(),
    )


ALIASES = {
    "Shipment_ID": ["shipment_identifier", "consignment_id"],
    "Origin_Port": ["Place_of_Origin", "origin", "origin_location"],
    "Destination_Port": [
        "Place_of_Destination", "destination", "destination_location",
    ],
    "Distance_km": ["Distance", "route_distance_km", "distance_in_km"],
    "Weight_MT": ["Stock_Volume", "weight_metric_tonnes"],
    "Lead_Time_Days": ["Lead-Time in days", "lead_days", "lead_time_in_days"],
    "Geopolitical_Risk_Score": ["Geopolitical_Risk", "geo_risk"],
    "Fuel_Price_Index": ["fuel_index"],
    "Carrier_Reliability_Score": ["carrier_reliability", "reliability_score"],
    TARGET_COLUMN: ["disruption_label"],
}

COLUMN_LOOKUP = {
    _key(alias): column
    for column in SHIPMENT_COLUMNS + [TARGET_COLUMN]
    for alias in [column, *ALIASES.get(column, [])]
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
            "Provide a DataFrame or load a CSV with read_raw_shipments()."
        )
    if df.columns.duplicated().any():
        raise ShipmentValidationError(
            "Duplicate input headers; keep one column per field."
        )

    result = df.rename(
        columns={
            column: COLUMN_LOOKUP.get(_key(column), column)
            for column in df
        }
    ).copy()

    duplicate = result.columns[result.columns.duplicated()].tolist()
    if duplicate:
        raise ShipmentValidationError(
            f"Multiple columns map to the same field: {duplicate}."
        )
    return result


def _numbers(series):
    unsupported = series.map(
        lambda value: isinstance(
            value, (bool, np.bool_, complex, np.complexfloating)
        )
    )
    return pd.to_numeric(
        series.astype(object).where(~unsupported, np.nan),
        errors="coerce",
    ).astype(float)


def _rows(mask):
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))[:5] + 1
    return ", ".join(map(str, positions))


def prepare_prediction_inputs(df, config=None):
    """Validate model inputs; leave missing values for notebook imputers."""
    result = normalize_column_names(df)
    config = config or {}

    for column in RAW_MODEL_INPUT_COLUMNS:
        if column not in result:
            result[column] = np.nan

    bounds = {
        "Distance_km": (0, 25000),
        "Lead_Time_Days": (0, 365),
        "Geopolitical_Risk_Score": (0, 10),
        "Carrier_Reliability_Score": (0, 1),
    }
    errors = []

    for column in RAW_MODEL_NUMERIC_COLUMNS:
        original = result[column]
        missing = original.isna() | original.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        values = _numbers(original)
        low, high = bounds[column]

        invalid = ~missing & (
            ~pd.Series(np.isfinite(values.to_numpy()), index=result.index)
            | (values < low)
            | (values > high)
        )

        if column in {"Distance_km", "Lead_Time_Days"}:
            invalid |= ~missing & (values <= 0)

        if invalid.any():
            errors.append(
                f"{column}: invalid supplied value; expected {low}..{high} "
                f"(distance and lead time must be positive); rows {_rows(invalid)}"
            )

        result[column] = values.where(~missing, np.nan)

    weather = result["Weather_Condition"]
    missing = weather.isna() | weather.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    bad_type = ~missing & ~weather.map(lambda value: isinstance(value, str))

    if bad_type.any():
        errors.append(
            f"Weather_Condition: expected text; rows {_rows(bad_type)}"
        )

    result["Weather_Condition"] = weather.map(
        lambda value: value.strip() if isinstance(value, str) else value
    ).astype(object).where(~missing, np.nan)

    modes = config.get("typical_pace", TYPICAL_PACE)
    lookup = {_key(value): value for value in modes}
    mode = result["Transport_Mode"]
    missing = mode.isna() | mode.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    values = mode.map(lambda value: lookup.get(_key(value), np.nan))
    invalid = ~missing & values.isna()

    if invalid.any():
        errors.append(
            f"Transport_Mode: expected one of {list(modes)}; "
            f"rows {_rows(invalid)}"
        )

    result["Transport_Mode"] = values.astype(object).where(
        ~missing, np.nan
    )

    if errors:
        raise ShipmentValidationError(
            "Prediction validation failed:\n- " + "\n- ".join(errors)
        )

    return result


def clean_raw_shipment_data(
    df,
    require_target=False,
    require_processed=True,
):
    """Both raw and context datasets require the same 13 recorded inputs."""
    result = normalize_column_names(df)

    required = SHIPMENT_COLUMNS + (
        [TARGET_COLUMN] if require_target else []
    )
    missing = [column for column in required if column not in result]

    if missing:
        raise ShipmentValidationError(
            "Missing required recorded fields: "
            + ", ".join(missing)
            + ". Do not substitute simulated values."
        )

    if result.empty:
        raise ShipmentValidationError("No shipment rows supplied.")

    errors = []

    for column in ["Shipment_ID", "Origin_Port", "Destination_Port"]:
        result[column] = result[column].astype("string").str.strip()
        invalid = result[column].isna() | result[column].eq("")

        if invalid.any():
            errors.append(
                f"{column}: missing text at rows {_rows(invalid)}"
            )

    date_values = result["Date"].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    date_types = date_values.map(
        lambda value: isinstance(
            value, (date, datetime, pd.Timestamp, np.datetime64)
        )
        or (
            isinstance(value, str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None
        )
    )
    dates = pd.to_datetime(
        date_values.where(date_types, pd.NaT),
        format="%Y-%m-%d",
        errors="coerce",
    )

    if dates.isna().any():
        errors.append(
            f"Date: expected YYYY-MM-DD at rows {_rows(dates.isna())}"
        )

    result["Date"] = dates.dt.normalize()

    bounds = {
        "Distance_km": (0, 25000),
        "Weight_MT": (0, 10000),
        "Fuel_Price_Index": (0, 10),
        "Geopolitical_Risk_Score": (0, 10),
        "Carrier_Reliability_Score": (0, 1),
        "Lead_Time_Days": (0, 365),
    }

    for column in NUMERIC_COLUMNS:
        values = _numbers(result[column])
        low, high = bounds[column]
        invalid = (
            ~pd.Series(
                np.isfinite(values.to_numpy()),
                index=result.index,
            )
            | (values < low)
            | (values > high)
        )

        if column in {
            "Distance_km", "Weight_MT", "Fuel_Price_Index", "Lead_Time_Days",
        }:
            invalid |= values <= 0

        if invalid.any():
            errors.append(
                f"{column}: expected finite values in documented bounds "
                f"{low}..{high}; rows {_rows(invalid)}"
            )

        result[column] = values

    for column, allowed in [
        ("Transport_Mode", TRANSPORT_MODES),
        ("Product_Category", PRODUCT_CATEGORIES),
        ("Weather_Condition", WEATHER_CONDITIONS),
    ]:
        lookup = {_key(value): value for value in allowed}
        values = result[column].map(
            lambda value: lookup.get(_key(value))
        )

        if values.isna().any():
            errors.append(
                f"{column}: expected one of {allowed}; "
                f"rows {_rows(values.isna())}"
            )

        result[column] = values

    binary_columns = [
        column
        for column in [TARGET_COLUMN, *FLAG_COLUMNS, "Needs_Review"]
        if column in result
    ]

    for column in binary_columns:
        values = _numbers(result[column])
        invalid = ~values.isin([0, 1])

        if invalid.any():
            errors.append(
                f"{column}: expected supplied binary 0/1 values at rows "
                f"{_rows(invalid)}; never generate missing outcomes"
            )
        else:
            result[column] = values.astype(int)

    if errors:
        raise ShipmentValidationError(
            "Shipment validation failed:\n- " + "\n- ".join(errors)
        )

    return result