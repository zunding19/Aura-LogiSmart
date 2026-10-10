"""Priority from supplied-data disruption probability, without invented USD loss."""
import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)


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
            "Disruption probabilities must be finite numbers "
            "between 0 and 1."
        )

    return result.astype(float)


def calculate_triage_queue(shipments_df: pd.DataFrame) -> pd.DataFrame:
    """Stable descending probability; urgency_score is a compatibility alias."""
    if not isinstance(shipments_df, pd.DataFrame):
        raise ShipmentValidationError(
            "Triage requires a shipment DataFrame."
        )

    probability_column = (
        "p_disruption"
        if "p_disruption" in shipments_df
        else "p_failure"
    )
    if probability_column not in shipments_df:
        raise ShipmentValidationError(
            "Supply p_disruption; legacy p_failure is also accepted."
        )

    queue = clean_raw_shipment_data(shipments_df).copy()
    probability = _probabilities(queue[probability_column])

    if "p_disruption" in queue and "p_failure" in queue:
        alias = _probabilities(queue["p_failure"])
        if not np.allclose(
            probability, alias, rtol=0, atol=1e-12
        ):
            raise ShipmentValidationError(
                "p_failure alias disagrees with p_disruption."
            )

    queue["p_disruption"] = probability
    queue["p_failure"] = probability
    queue["risk_score"] = probability
    queue["urgency_score"] = probability
    queue["priority_basis"] = (
        "Recorded-data disruption probability only; "
        "no observed loss cost or hospital urgency. "
        "urgency_score and p_failure are compatibility aliases, "
        "not clinical or spoilage measures."
    )

    return queue.sort_values(
        "risk_score",
        ascending=False,
        kind="mergesort",
    )