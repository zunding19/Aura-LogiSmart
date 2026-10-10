"""Local scenarios for ETA and total packaging capacity."""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)


_DESCRIPTIONS = [
    (
        "monitor",
        "Continue monitoring",
        0,
        "Review inputs; no shipment changes.",
    ),
    (
        "cold_storage_transfer",
        "Qualified cold-storage transfer",
        200,
        "Explore qualified storage compatible with the product.",
    ),
    (
        "direct_flight",
        "Available direct-flight scenario",
        450,
        "Explore a feasible air route and qualified carrier.",
    ),
    (
        "refrigerant_refresh",
        "Qualified coolant refresh",
        120,
        "Use only product-compatible validated coolant.",
    ),
    (
        "refrigerated_ground_transport",
        "Qualified refrigerated ground transport",
        500,
        "Explore an available qualified road carrier.",
    ),
    (
        "emergency_stock_diversion",
        "Hospital replacement-stock diversion",
        750,
        "Replacement stock does not repair this shipment.",
    ),
]

ACTION_CATALOG = {
    key: {
        "action_id": key,
        "title": title,
        "cost_usd": float(cost),
        "rationale": reason,
    }
    for key, title, cost, reason in _DESCRIPTIONS
}

ACTIONS = ACTION_CATALOG


def get_action_catalog():
    return [
        dict(action)
        for action in ACTION_CATALOG.values()
    ]


def _finite_number(
    value,
    name,
    minimum=0.0,
    maximum=None,
):
    if isinstance(
        value,
        (
            bool,
            np.bool_,
            complex,
            np.complexfloating,
        ),
    ):
        raise ShipmentValidationError(
            f"{name} must be a real number."
        )

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        raise ShipmentValidationError(
            f"{name} must be numeric."
        ) from None

    if (
        not math.isfinite(number)
        or number < minimum
        or (
            maximum is not None
            and number > maximum
        )
    ):
        raise ShipmentValidationError(
            f"{name} is nonfinite "
            "or outside its permitted range."
        )

    return number


def _clean_shipment(shipment_data):
    if isinstance(
        shipment_data, pd.Series
    ):
        shipment_data = (
            shipment_data.to_dict()
        )

    if not isinstance(
        shipment_data, Mapping
    ):
        raise ShipmentValidationError(
            "Supply one shipment dictionary or Series."
        )

    return clean_raw_shipment_data(
        pd.DataFrame([
            dict(shipment_data)
        ])
    ).iloc[0].to_dict()


def predict_failure_risk(features_df, model):
    from src.model import (
        predict_failure_risk as predict,
    )

    return predict(
        features_df, model
    )


def _predict_one(shipment, model):
    if model is None:
        raise ShipmentValidationError(
            "A fitted model is required "
            "for scenario comparison."
        )

    from src.features import (
        extract_features,
    )

    result = predict_failure_risk(
        extract_features(
            pd.DataFrame([shipment])
        ),
        model,
    )

    return _finite_number(
        result.iloc[0]["p_failure"],
        "model probability",
        maximum=1,
    )


def get_recommended_action(
    risk_category,
    risk_prob,
    buffer_hours,
):
    """Buffer here is total hold capacity."""
    probability = _finite_number(
        risk_prob,
        "risk_prob",
        maximum=1,
    )

    capacity = _finite_number(
        buffer_hours,
        "buffer_hours",
    )

    categories = {
        "Cold-chain excursion",
        "Transport delay",
        "Handover delay",
        "Combined risk",
        "Routine monitoring",
    }

    if risk_category not in categories:
        raise ShipmentValidationError(
            "Unknown risk category."
        )

    action = "monitor"

    if (
        probability > .3
        and risk_category != "Routine monitoring"
    ):
        if risk_category in {
            "Cold-chain excursion",
            "Combined risk",
        }:
            action = (
                "refrigerant_refresh"
                if capacity < 3
                else "cold_storage_transfer"
            )
        elif risk_category == "Transport delay":
            action = "direct_flight"
        else:
            action = (
                "refrigerated_ground_transport"
            )

    return dict(
        ACTION_CATALOG[action],
        note=(
            "Heuristic scenario; verify availability, "
            "compatibility and example costs. "
            "Total hold capacity is not a live countdown."
        ),
    )


def simulate_intervention(
    shipment_data,
    action_id,
    model,
):
    if action_id not in ACTION_CATALOG:
        raise ShipmentValidationError(
            "Unknown action_id; use get_action_catalog()."
        )

    original = _clean_shipment(
        shipment_data
    )
    scenario = dict(original)

    assumption = (
        "Monitoring changes no shipment inputs."
    )

    if action_id in {
        "cold_storage_transfer",
        "refrigerant_refresh",
    }:
        hours = (
            24
            if action_id == "cold_storage_transfer"
            else 48
        )

        scenario["Buffer_of_Package"] += hours

        assumption = (
            "Assume qualified product-compatible intervention "
            f"adds {hours} hours to total hold capacity, "
            "not a measured remaining countdown."
        )

    elif action_id == "direct_flight":
        scenario.update(
            Transport_Mode="Air",
            Carrier_Reliability_Score=max(
                original["Carrier_Reliability_Score"],
                .95,
            ),
            Handover_Count=min(
                original["Handover_Count"],
                2,
            ),
        )

        assumption = (
            "Assume feasible direct Air route, "
            "available carrier reliability >=0.95 "
            "and at most two planned handovers. "
            "Journey ETA and planned transit remain unchanged "
            "without route evidence."
        )

    elif action_id == "refrigerated_ground_transport":
        scenario.update(
            Transport_Mode="Land",
            Carrier_Reliability_Score=max(
                original["Carrier_Reliability_Score"],
                .9,
            ),
            Buffer_of_Package=(
                original["Buffer_of_Package"] + 12
            ),
        )

        assumption = (
            "Assume available qualified Land carrier "
            "reliability >=0.90 and 12 added total hold hours. "
            "No travel ETA improvement is assumed."
        )

    elif action_id == "emergency_stock_diversion":
        assumption = (
            "Replacement stock may address hospital disruption; "
            "every original-shipment failure input stays unchanged. "
            "No stockout benefit is modeled."
        )

    scenario = _clean_shipment(
        scenario
    )

    changed = {
        field: {
            "before": original[field],
            "after": scenario[field],
        }
        for field in (
            "Buffer_of_Package",
            "Transport_Mode",
            "Carrier_Reliability_Score",
            "Handover_Count",
        )
        if original[field] != scenario[field]
    }

    before = _predict_one(
        original, model
    )
    after = _predict_one(
        scenario, model
    )

    cost = ACTION_CATALOG[
        action_id
    ]["cost_usd"]

    reduction = (
        (before - after)
        * original["Cargo_Value_USD"]
    )

    if not math.isfinite(
        reduction - cost
    ):
        raise ShipmentValidationError(
            "Cargo value exceeds finite scenario arithmetic."
        )

    return {
        "action_id": action_id,
        "title": ACTION_CATALOG[action_id]["title"],
        "before_probability": before,
        "after_probability": after,
        "action_cost_usd": cost,
        "estimated_net_benefit_usd": reduction - cost,
        "estimated_loss_reduction_usd": reduction,
        "changed_fields": changed,
        "assumptions": [assumption],
        "note": (
            "Assumed model sensitivity, not causal effect "
            "or guaranteed savings; probability can increase. "
            "Assumes full cargo loss on failure; "
            "example USD fees; no action dispatched."
        ),
    }


def check_sla_status(
    risk_prob,
    buffer_hours,
    corrective_proof=False,
):
    """Pass an actual remaining observation, not CSV total capacity."""
    probability = _finite_number(
        risk_prob,
        "risk_prob",
        maximum=1,
    )

    remaining = _finite_number(
        buffer_hours,
        "actual remaining buffer hours",
    )

    if not isinstance(
        corrective_proof, bool
    ):
        raise ShipmentValidationError(
            "corrective_proof must be True or False."
        )

    triggered = (
        probability > .8
        and remaining < 3
    )
    locked = (
        triggered
        and not corrective_proof
    )

    return {
        "trigger_condition": triggered,
        "breach_warning": locked,
        "simulated_escrow_locked": locked,
        "corrective_proof_recorded": corrective_proof,
        "status": (
            "Simulated lock"
            if locked
            else "No simulated lock"
        ),
        "requires_actual_remaining_observation": True,
        "note": (
            "Manual prototype requires actual observed "
            "remaining hold hours. This dataset supplies total "
            "capacity and cannot establish a live countdown. "
            "No real escrow or notice; proof is user-attested."
        ),
    }