"""Priority score using total packaging capacity."""
import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import (
    URGENCY_WEIGHTS,
)


def calculate_triage_queue(
    shipments_df: pd.DataFrame,
):
    if (
        not isinstance(
            shipments_df, pd.DataFrame
        )
        or "p_failure" not in shipments_df
    ):
        raise ShipmentValidationError(
            "Triage requires a DataFrame with p_failure."
        )

    queue = clean_raw_shipment_data(
        shipments_df
    ).copy()

    unsupported = queue[
        "p_failure"
    ].map(
        lambda value: isinstance(
            value,
            (
                bool,
                np.bool_,
                complex,
                np.complexfloating,
            ),
        )
    )

    probability = pd.to_numeric(
        queue["p_failure"].where(
            ~unsupported,
            np.nan,
        ),
        errors="coerce",
    )

    invalid = (
        probability.isna()
        | ~np.isfinite(probability)
        | ~probability.between(0, 1)
    )

    if invalid.any():
        raise ShipmentValidationError(
            "p_failure must be finite and between 0 and 1."
        )

    queue["p_failure"] = (
        probability.astype(float)
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        queue["expected_loss_usd"] = (
            probability
            * queue["Cargo_Value_USD"]
        )

        queue["urgency_score"] = (
            queue["expected_loss_usd"]
            * queue["Hospital_Urgency_Tier"]
            .map(URGENCY_WEIGHTS)
            / queue["Buffer_of_Package"]
            .clip(lower=0.5)
        )

    if not np.isfinite(
        queue[
            [
                "expected_loss_usd",
                "urgency_score",
            ]
        ].to_numpy()
    ).all():
        raise ShipmentValidationError(
            "Shipment values exceed finite priority arithmetic."
        )

    queue["priority_basis"] = (
        "Total packaging hold capacity; "
        "elapsed time unavailable"
    )

    return queue.sort_values(
        "urgency_score",
        ascending=False,
        kind="mergesort",
    )