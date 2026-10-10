"""Features use recorded inputs only; Date is assumed to precede the outcome."""
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
    df = clean_raw_shipment_data(clean_df)

    result = df[
        [
            column
            for column in MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
            if column in df
        ]
    ].copy()

    angle = 2 * np.pi * (df["Date"].dt.month - 1) / 12
    result["month_sin"] = np.sin(angle)
    result["month_cos"] = np.cos(angle)

    if not np.isfinite(
        result[MODEL_NUMERIC_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ShipmentValidationError(
            "Nonfinite model features; check recorded inputs."
        )

    return result.loc[:, MODEL_FEATURE_COLUMNS]