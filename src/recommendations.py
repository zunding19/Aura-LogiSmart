#suggests intervention, creates a hypothetical scenerio, reruns same model and estimates the financial benefit
import math
from collections.abc import Mapping

import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)


ACTION_CATALOG = {
    "monitor": {
        "action_id": "monitor",
        "title": "Continue monitoring",
        "cost_usd": 0.0,
        "rationale": (
            "Review telemetry and qualified packaging hold time; "
            "no scenario inputs change."
        ),
    },
    "cold_storage_transfer": {
        "action_id": "cold_storage_transfer",
        "title": "Qualified cold-storage transfer",
        "cost_usd": 200.0,
        "rationale": (
            "Consider a storage-compatible qualified facility "
            "at the current hub."
        ),
    },
    "direct_flight": {
        "action_id": "direct_flight",
        "title": "Available direct-flight scenario",
        "cost_usd": 450.0,
        "rationale": (
            "Explore an available air route "
            "with a higher-reliability carrier."
        ),
    },
    "refrigerant_refresh": {
        "action_id": "refrigerant_refresh",
        "title": "Qualified coolant refresh",
        "cost_usd": 120.0,
        "rationale": (
            "Qualified staff replenish storage-compatible coolant; "
            "compatibility must be checked."
        ),
    },
    "refrigerated_ground_transport": {
        "action_id": "refrigerated_ground_transport",
        "title": "Refrigerated ground-transport scenario",
        "cost_usd": 500.0,
        "rationale": (
            "Consider an available qualified "
            "temperature-controlled road carrier."
        ),
    },
    "emergency_stock_diversion": {
        "action_id": "emergency_stock_diversion",
        "title": "Hospital replacement-stock diversion",
        "cost_usd": 750.0,
        "rationale": (
            "Assess replacement stock separately; "
            "diversion does not repair the original shipment."
        ),
    },
}

ACTIONS = ACTION_CATALOG


def get_action_catalog() -> list[dict]:
    return [
        dict(action)
        for action in ACTION_CATALOG.values()
    ]


def _finite_number(
    value,
    name: str,
    minimum: float = 0.0,
    maximum=None,
) -> float:
    if isinstance(value, bool):
        raise ShipmentValidationError(
            f"{name} must be a number, not a boolean."
        )

    try:
        number = float(value)
    except (TypeError, ValueError):
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
        upper = (
            f" and at most {maximum}"
            if maximum is not None
            else ""
        )
        raise ShipmentValidationError(
            f"{name} must be finite, "
            f"at least {minimum}{upper}."
        )

    return number


def _clean_shipment(shipment_data) -> dict:
    if isinstance(shipment_data, pd.Series):
        shipment_data = shipment_data.to_dict()

    if not isinstance(shipment_data, Mapping):
        raise ShipmentValidationError(
            "Supply one shipment as a dictionary or Series."
        )

    return clean_raw_shipment_data(
        pd.DataFrame([dict(shipment_data)])
    ).iloc[0].to_dict()


def predict_failure_risk(features_df, model):
    from src.model import predict_failure_risk as predict

    return predict(features_df, model)


def _predict_one(shipment: dict, model) -> float:
    if model is None:
        raise ShipmentValidationError(
            "A fitted model is required "
            "for a scenario comparison."
        )

    from src.features import extract_features

    predicted = predict_failure_risk(
        extract_features(pd.DataFrame([shipment])),
        model,
    )
    return _finite_number(
        predicted.iloc[0]["p_failure"],
        "model probability",
        maximum=1.0,
    )


def get_recommended_action(
    risk_category,
    risk_prob,
    buffer_hours,
) -> dict:
    probability = _finite_number(
        risk_prob, "risk_prob", maximum=1.0
    )
    buffer = _finite_number(
        buffer_hours, "buffer_hours"
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
            "Use one of the documented risk categories."
        )

    if (
        probability <= 0.3
        or risk_category == "Routine monitoring"
    ):
        action_id = "monitor"
    elif risk_category in {
        "Cold-chain excursion", "Combined risk"
    }:
        action_id = (
            "refrigerant_refresh"
            if buffer < 3.0
            else "cold_storage_transfer"
        )
    elif risk_category == "Transport delay":
        action_id = "direct_flight"
    else:
        action_id = (
            "cold_storage_transfer"
            if buffer < 6.0
            else "refrigerated_ground_transport"
        )

    recommendation = dict(ACTION_CATALOG[action_id])
    recommendation["note"] = (
        "Heuristic suggestion; verify feasibility, costs "
        "and packaging qualification with an operator."
    )
    return recommendation


def simulate_intervention(
    shipment_data,
    action_id,
    model,
) -> dict:
    if action_id not in ACTION_CATALOG:
        raise ShipmentValidationError(
            "Unknown action_id. Choose from: "
            + ", ".join(ACTION_CATALOG)
            + "."
        )

    original = _clean_shipment(shipment_data)
    scenario = dict(original)
    assumptions = []

    if action_id == "cold_storage_transfer":
        scenario["Buffer_of_Package"] += 24.0
        assumptions = [
            "Assume qualified, storage-compatible hub transfer "
            "adds 24 remaining hold hours."
        ]

    elif action_id == "refrigerant_refresh":
        scenario["Buffer_of_Package"] += 48.0
        assumptions = [
            "Assume qualified storage-compatible coolant refresh "
            "adds 48 hold hours; dry ice is not suitable "
            "for every product."
        ]

    elif action_id == "direct_flight":
        scenario["Transport_Mode"] = "Air"
        scenario["Carrier_Reliability_Score"] = max(
            original["Carrier_Reliability_Score"], 0.95
        )
        assumptions = [
            "Assume a feasible Air route and available carrier "
            "with reliability at least 0.95; elapsed transit "
            "and deadline remain unchanged."
        ]

    elif action_id == "refrigerated_ground_transport":
        scenario["Transport_Mode"] = "Land"
        scenario["Carrier_Reliability_Score"] = max(
            original["Carrier_Reliability_Score"], 0.9
        )
        scenario["Buffer_of_Package"] += 12.0
        assumptions = [
            "Assume an available qualified Land carrier "
            "with reliability at least 0.90 "
            "and 12 added hold hours."
        ]

    elif action_id == "emergency_stock_diversion":
        assumptions = [
            "Replacement inventory may reduce hospital disruption, "
            "but original-shipment failure inputs remain unchanged; "
            "no stockout benefit is modeled."
        ]

    else:
        assumptions = [
            "Monitoring changes no shipment inputs."
        ]

    scenario = _clean_shipment(scenario)

    changed = {
        key: {
            "before": original[key],
            "after": scenario[key],
        }
        for key in (
            "Buffer_of_Package",
            "Transport_Mode",
            "Carrier_Reliability_Score",
        )
        if original[key] != scenario[key]
    }

    before_probability = _predict_one(original, model)
    after_probability = _predict_one(scenario, model)

    cost = float(ACTION_CATALOG[action_id]["cost_usd"])
    expected_loss_reduction = (
        (before_probability - after_probability)
        * original["Cargo_Value_USD"]
    )
    net_benefit = expected_loss_reduction - cost

    if not math.isfinite(net_benefit):
        raise ShipmentValidationError(
            "Cargo value is too large "
            "to compute a finite scenario benefit."
        )

    return {
        "action_id": action_id,
        "title": ACTION_CATALOG[action_id]["title"],
        "before_probability": before_probability,
        "after_probability": after_probability,
        "action_cost_usd": cost,
        "estimated_net_benefit_usd": net_benefit,
        "estimated_loss_reduction_usd": expected_loss_reduction,
        "changed_fields": changed,
        "assumptions": assumptions,
        "note": (
            "Illustrative model sensitivity, "
            "not proven treatment effect or guaranteed savings. "
            "Assumes total cargo loss on failure. "
            "Costs are example USD amounts; "
            "no action is dispatched."
        ),
    }


def check_sla_status(
    risk_prob,
    buffer_hours,
    corrective_proof=False,
) -> dict:
    probability = _finite_number(
        risk_prob, "risk_prob", maximum=1.0
    )
    buffer = _finite_number(
        buffer_hours, "buffer_hours"
    )

    if not isinstance(corrective_proof, bool):
        raise ShipmentValidationError(
            "corrective_proof must be True or False."
        )

    triggered = probability > 0.8 and buffer < 3.0
    locked = triggered and not corrective_proof

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
        "note": (
            "Local demonstration only: "
            "no carrier notice, payment or escrow action. "
            "Corrective proof is user-attested, "
            "not independently verified."
        ),
    }