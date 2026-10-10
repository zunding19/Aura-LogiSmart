"""Notebook 05 pharmaceutical-alert queue; financial and urgency inputs are simulated."""
import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    prepare_prediction_inputs,
)
from src.recommendations import (
    ALERT_LINE,
    LOSS_SHARE,
    _finite_number,
)

URGENCY_WEIGHT = {
    "Routine Restock": 1.0,
    "Elevated Procedure": 1.5,
    "Critical Emergency": 2.5,
}


def _probabilities(values):
    unsupported = values.map(
        lambda value: isinstance(
            value, (bool, np.bool_, complex, np.complexfloating)
        )
    )

    result = pd.to_numeric(
        values.where(~unsupported, np.nan),
        errors="coerce",
    )

    if (
        result.isna()
        | ~np.isfinite(result)
        | ~result.between(0, 1)
    ).any():
        raise ShipmentValidationError(
            "Disruption probabilities must be finite numbers from 0 to 1."
        )

    return result.astype(float)


def calculate_triage_queue(shipments_df):
    """Retain alerted pharmaceuticals and apply the exact notebook priority formula."""
    if not isinstance(shipments_df, pd.DataFrame):
        raise ShipmentValidationError(
            "Triage requires a DataFrame."
        )

    source = (
        "p_disruption"
        if "p_disruption" in shipments_df
        else "p_failure"
    )

    if source not in shipments_df:
        raise ShipmentValidationError(
            "Supply p_disruption (or legacy p_failure)."
        )

    queue = prepare_prediction_inputs(shipments_df).copy()
    probability = _probabilities(queue[source])

    if (
        "p_disruption" in queue
        and "p_failure" in queue
        and not np.allclose(
            probability,
            _probabilities(queue["p_failure"]),
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ShipmentValidationError(
            "p_failure alias disagrees with p_disruption."
        )

    required = [
        "Product_Category",
        "Cargo_Value_USD_Simulated",
        "Hospital_Urgency_Tier_Simulated",
        "Remaining_Buffer_Hours_Simulated",
    ]

    if any(column not in queue for column in required):
        raise ShipmentValidationError(
            "Notebook queue requires: " + ", ".join(required)
        )

    queue["p_disruption"], queue["p_failure"] = probability, probability
    queue["risk"], queue["alert"] = probability, probability >= ALERT_LINE

    queue = queue[
        queue["alert"]
        & queue["Product_Category"].eq("Pharmaceuticals")
    ].copy()

    for column in [
        "Cargo_Value_USD_Simulated",
        "Remaining_Buffer_Hours_Simulated",
    ]:
        queue[column] = queue[column].map(
            lambda value: _finite_number(value, column)
        ).astype(float)

    weights = queue["Hospital_Urgency_Tier_Simulated"].map(
        URGENCY_WEIGHT
    )

    if weights.isna().any():
        raise ShipmentValidationError(
            "Pharmaceutical queue requires "
            "a notebook simulated hospital urgency tier."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        queue["priority"] = (
            queue["risk"]
            * queue["Cargo_Value_USD_Simulated"]
            * weights
            / queue["Remaining_Buffer_Hours_Simulated"].clip(lower=.5)
        )

        queue["expected_loss_usd"] = (
            queue["risk"]
            * LOSS_SHARE
            * queue["Cargo_Value_USD_Simulated"]
        )

    if not np.isfinite(
        queue[["priority", "expected_loss_usd"]].to_numpy()
    ).all():
        raise ShipmentValidationError(
            "Simulated values exceed finite priority arithmetic."
        )

    queue["urgency_score"], queue["risk_score"] = (
        queue["priority"],
        queue["risk"],
    )

    queue["priority_basis"] = (
        "Notebook scenario: p × simulated cargo USD × simulated urgency "
        "/ max(simulated remaining buffer, 0.5). "
        "Expected loss assumes 10% cargo cost per disruption."
    )

    return queue.sort_values(
        "priority",
        ascending=False,
        kind="mergesort",
    )