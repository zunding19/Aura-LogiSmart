"""Recorded-input what-if scenarios; no medical effects or financial guarantees."""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import TARGET_COLUMN

_DESCRIPTIONS = [
    (
        "monitor",
        "Continue monitoring",
        "Review supplied records; no inputs change.",
    ),
    (
        "carrier_review",
        "Review an alternative carrier",
        "Compare a hypothetical qualified higher-reliability carrier.",
    ),
    (
        "route_review",
        "Review alternative route exposure",
        "Compare a hypothetical feasible lower-exposure route.",
    ),
    (
        "weather_contingency",
        "Review a weather contingency",
        "Compare Clear weather as a hypothetical reference; "
        "weather cannot be controlled.",
    ),
]

ACTION_CATALOG = {
    key: dict(
        action_id=key,
        title=title,
        cost_usd=0.0 if key == "monitor" else None,
        rationale=reason,
    )
    for key, title, reason in _DESCRIPTIONS
}
ACTIONS = ACTION_CATALOG

RISK_CATEGORIES = {
    "Weather exposure",
    "Carrier reliability",
    "Geopolitical exposure",
    "Combined exposure",
    "Routine monitoring",
}


def get_action_catalog() -> list[dict]:
    return [dict(action) for action in ACTION_CATALOG.values()]


def _finite_number(value, name, minimum=0.0, maximum=None):
    if isinstance(
        value, (bool, np.bool_, complex, np.complexfloating)
    ):
        raise ShipmentValidationError(
            f"{name} must be a real number."
        )

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ShipmentValidationError(
            f"{name} must be numeric."
        ) from None

    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        raise ShipmentValidationError(
            f"{name} is nonfinite or outside its permitted range."
        )

    return number


def _clean_shipment(shipment_data) -> dict:
    if isinstance(shipment_data, pd.Series):
        shipment_data = shipment_data.to_dict()

    if not isinstance(shipment_data, Mapping):
        raise ShipmentValidationError(
            "Supply one shipment dictionary or Series."
        )

    return clean_raw_shipment_data(
        pd.DataFrame([dict(shipment_data)])
    ).iloc[0].to_dict()


def predict_failure_risk(features_df, model):
    from src.model import predict_failure_risk as predict

    return predict(features_df, model)


def _predict_one(shipment, model):
    if model is None:
        raise ShipmentValidationError(
            "A fitted disruption model is required "
            "for scenario comparison."
        )

    from src.features import extract_features

    prediction = predict_failure_risk(
        extract_features(pd.DataFrame([shipment])),
        model,
    )
    column = (
        "p_disruption"
        if "p_disruption" in prediction
        else "p_failure"
    )

    return _finite_number(
        prediction.iloc[0][column],
        "model probability",
        maximum=1,
    )


def get_recommended_action(
    risk_category,
    risk_prob,
    buffer_hours=None,
) -> dict:
    """buffer_hours remains for compatibility and is unused by this target."""
    probability = _finite_number(
        risk_prob, "risk_prob", maximum=1
    )

    if risk_category not in RISK_CATEGORIES:
        raise ShipmentValidationError(
            "Unknown recorded-data exposure category."
        )

    mapping = {
        "Weather exposure": "weather_contingency",
        "Carrier reliability": "carrier_review",
        "Geopolitical exposure": "route_review",
        "Combined exposure": "carrier_review",
    }
    action = (
        "monitor"
        if probability <= .3
        else mapping.get(risk_category, "monitor")
    )

    return dict(
        ACTION_CATALOG[action],
        target=TARGET_COLUMN,
        note=(
            "Heuristic review suggestion, not an optimal or proven "
            "intervention. Packaging buffer is unrelated to this "
            "disruption target; costs are unavailable."
        ),
    )


def simulate_intervention(
    shipment_data,
    action_id,
    model,
) -> dict:
    """Same-model comparisons with explicit assumptions and signed changes."""
    if action_id not in ACTION_CATALOG:
        raise ShipmentValidationError(
            "Unknown action_id; use get_action_catalog()."
        )

    original = _clean_shipment(shipment_data)
    scenario = dict(original)
    assumptions = ["Monitoring changes no model inputs."]

    if action_id == "carrier_review":
        scenario["Carrier_Reliability_Score"] = max(
            original["Carrier_Reliability_Score"], .95
        )
        assumptions = [
            "Assume an available feasible replacement carrier with "
            "reliability >=0.95; review alone does not improve reliability."
        ]
    elif action_id == "route_review":
        scenario["Geopolitical_Risk_Score"] = min(
            original["Geopolitical_Risk_Score"], 2
        )
        assumptions = [
            "Assume a feasible route with exposure score <=2. "
            "Other route fields stay fixed; this is not "
            "a fully specified reroute."
        ]
    elif action_id == "weather_contingency":
        scenario["Weather_Condition"] = "Clear"
        assumptions = [
            "Hypothetical Clear weather reference; contingency planning "
            "cannot control weather, establish a forecast, or prove "
            "when conditions improve."
        ]

    scenario = _clean_shipment(scenario)
    fields = [
        "Carrier_Reliability_Score",
        "Geopolitical_Risk_Score",
        "Weather_Condition",
    ]
    changed = {
        field: dict(
            before=original[field],
            after=scenario[field],
        )
        for field in fields
        if original[field] != scenario[field]
    }

    before = _predict_one(original, model)
    after = _predict_one(scenario, model)

    return dict(
        action_id=action_id,
        title=ACTION_CATALOG[action_id]["title"],
        target=TARGET_COLUMN,
        before_probability=before,
        after_probability=after,
        probability_change=before - after,
        before_p_disruption=before,
        after_p_disruption=after,
        action_cost_usd=ACTION_CATALOG[action_id]["cost_usd"],
        estimated_net_benefit_usd=None,
        estimated_loss_reduction_usd=None,
        changed_fields=changed,
        assumptions=assumptions,
        note=(
            "Hypothetical model sensitivity, not causal effect or "
            "guaranteed improvement. Recorded disruption is not cargo "
            "loss, spoilage or patient harm; disruption costs are unknown. "
            "No action dispatched."
        ),
    )


def check_sla_status(
    risk_prob,
    buffer_hours=None,
    corrective_proof=False,
) -> dict:
    """Disruption risk cannot trigger the old medical SLA rule."""
    _finite_number(risk_prob, "risk_prob", maximum=1)

    if not isinstance(corrective_proof, bool):
        raise ShipmentValidationError(
            "corrective_proof must be True or False."
        )

    return dict(
        available=False,
        target=TARGET_COLUMN,
        trigger_condition=False,
        breach_warning=False,
        simulated_escrow_locked=False,
        corrective_proof_recorded=corrective_proof,
        status="Unavailable for disruption target",
        note=(
            "Disruption_Occurred does not measure cold-chain failure. "
            "Neither simulated nor observed packaging buffer makes this "
            "probability compatible with the old medical SLA threshold. "
            "No notice, escrow or payment action."
        ),
    )