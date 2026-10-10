from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import (
    MODEL_FEATURE_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
)


def extract_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    shipments = clean_raw_shipment_data(clean_df)

    features = shipments[
        [
            column
            for column in (
                MODEL_NUMERIC_COLUMNS
                + MODEL_CATEGORICAL_COLUMNS
            )
            if column in shipments.columns
        ]
    ].copy()

    with np.errstate(
        over="ignore", invalid="ignore", divide="ignore"
    ):
        sla_hours = shipments["Lead-Time in days"] * 24.0
        remaining = (
            sla_hours - shipments["Transit_Duration"]
        ).clip(lower=0.0)

        features["remaining_sla_hours"] = remaining
        features["thermal_margin_to_deadline"] = (
            shipments["Buffer_of_Package"] - remaining
        )
        features["elapsed_sla_ratio"] = (
            shipments["Transit_Duration"]
            / sla_hours.clip(lower=0.5)
        )
        features["buffer_to_deadline_ratio"] = (
            shipments["Buffer_of_Package"]
            / remaining.clip(lower=0.5)
        )
        features["multimodal_flag"] = (
            shipments["Transport_Mode"]
            .eq("Multi-Modal Air-to-Land")
            .astype(int)
        )

    if not np.isfinite(
        features[MODEL_NUMERIC_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ShipmentValidationError(
            "Shipment magnitudes exceed finite feature arithmetic. "
            "Check units and unrealistic numeric values."
        )

    return features.loc[:, MODEL_FEATURE_COLUMNS]