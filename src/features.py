"""Exact notebook 03/04/05 arithmetic; use the mappings saved in notebook 05."""
import numpy as np
import pandas as pd

from src.preprocessing import ShipmentValidationError, prepare_prediction_inputs
from src.schema import (
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    FALLBACK_FEATURE_COLUMNS,
    WEATHER_SEVERITY,
    TYPICAL_PACE,
    HIGH_RISK_SCORE,
    LOW_RELIABILITY,
)


def add_features(shipments, config=None):
    """Missing inputs remain NaN for notebook imputers; unknown weather uses fallback."""
    config = config or {}
    out = prepare_prediction_inputs(shipments, config=config)

    severity = out["Weather_Condition"].map(
        config.get("weather_severity", WEATHER_SEVERITY)
    )
    hours = out["Lead_Time_Days"] * 24
    typical = out["Transport_Mode"].map(
        config.get("typical_pace", TYPICAL_PACE)
    )

    out["weather_severity"] = severity
    out["severe_weather_flag"] = (
        severity >= 3
    ).astype(float).where(severity.notna())

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out["pace_h_per_1000km"] = hours / (out["Distance_km"] / 1000)
        out["pace_vs_typical"] = out["pace_h_per_1000km"] / typical

    out["red_flag_count"] = (
        (severity >= 3).astype(int)
        + (
            out["Geopolitical_Risk_Score"]
            >= config.get("high_risk_score", HIGH_RISK_SCORE)
        ).astype(int)
        + (
            out["Carrier_Reliability_Score"]
            < config.get("low_reliability", LOW_RELIABILITY)
        ).astype(int)
    )

    # Exposure is display-only and never part of either saved model's features.
    elapsed = pd.to_numeric(
        out.get(
            "Elapsed_Transit_Hours_Simulated",
            pd.Series(np.nan, index=out.index),
        ),
        errors="coerce",
    )
    buffer = pd.to_numeric(
        out.get(
            "Remaining_Buffer_Hours_Simulated",
            pd.Series(np.nan, index=out.index),
        ),
        errors="coerce",
    )

    out["hours_to_go"] = hours - elapsed
    out["cold_gap_hours"] = buffer - out["hours_to_go"]

    if np.isinf(out[MODEL_NUMERIC_COLUMNS].to_numpy(dtype=float)).any():
        raise ShipmentValidationError(
            "Infinite engineered features; check numeric magnitudes "
            "and artifact mappings."
        )

    return out


def extract_features(shipments, config=None):
    return add_features(
        shipments, config=config
    ).loc[:, MODEL_FEATURE_COLUMNS]


def extract_fallback_features(shipments, config=None):
    return add_features(
        shipments, config=config
    ).loc[:, FALLBACK_FEATURE_COLUMNS]