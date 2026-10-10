import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import URGENCY_WEIGHTS


def calculate_triage_queue(
    shipments_df: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(shipments_df, pd.DataFrame):
        raise ShipmentValidationError(
            "Triage requires a shipment DataFrame."
        )

    if "p_failure" not in shipments_df.columns:
        raise ShipmentValidationError(
            "Triage requires a p_failure column."
        )

    queue = clean_raw_shipment_data(shipments_df).copy()
    probability = pd.to_numeric(
        queue["p_failure"], errors="coerce"
    )

    invalid = (
        probability.isna()
        | ~np.isfinite(probability)
        | probability.lt(0)
        | probability.gt(1)
    )
    if invalid.any():
        raise ShipmentValidationError(
            "p_failure must be finite and between 0 and 1."
        )

    queue["p_failure"] = probability.astype(float)
    urgency_weight = queue["Hospital_Urgency_Tier"].map(
        URGENCY_WEIGHTS
    )

    queue["expected_loss_usd"] = (
        probability * queue["Cargo_Value_USD"]
    )
    queue["urgency_score"] = (
        queue["expected_loss_usd"]
        * urgency_weight
        / queue["Buffer_of_Package"].clip(lower=0.5)
    )

    if not np.isfinite(
        queue[
            ["expected_loss_usd", "urgency_score"]
        ].to_numpy()
    ).all():
        raise ShipmentValidationError(
            "Cargo value is too large "
            "to compute finite triage scores."
        )

    return queue.sort_values(
        "urgency_score",
        ascending=False,
        kind="mergesort",
    )