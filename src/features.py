"""ETA features excluding outcomes, identifiers and financial urgency."""
import numpy as np

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import (
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
)


def extract_features(clean_df):
    df = clean_raw_shipment_data(
        clean_df
    )

    features = df[
        [
            column
            for column in (
                MODEL_NUMERIC_COLUMNS
                + MODEL_CATEGORICAL_COLUMNS
            )
            if column in df
        ]
    ].copy()

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        eta = df["Transit_Duration"]
        planned = df["Planned_Transit_Hours"]
        deadline = (
            df["Lead-Time in days"] * 24
        )
        hold = df["Buffer_of_Package"]

        features["known_delay_hours"] = (
            eta - planned
        )

        features["deadline_margin_hours"] = (
            deadline - eta
        )

        features["thermal_margin_hours"] = (
            hold - eta
        )

        features["eta_to_planned_ratio"] = (
            eta / planned.clip(lower=0.5)
        )

        features["eta_to_deadline_ratio"] = (
            eta / deadline.clip(lower=0.5)
        )

        features["buffer_to_eta_ratio"] = (
            hold / eta.clip(lower=0.5)
        )

        features["multimodal_flag"] = (
            df["Transport_Mode"]
            .eq("Multi-Modal Air-to-Land")
            .astype(int)
        )

    if not np.isfinite(
        features[
            MODEL_NUMERIC_COLUMNS
        ].to_numpy(dtype=float)
    ).all():
        raise ShipmentValidationError(
            "Derived features exceed finite arithmetic; "
            "check units and magnitudes."
        )

    return features.loc[
        :, MODEL_FEATURE_COLUMNS
    ]