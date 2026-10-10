"""Notebook 05 action scenarios; fees, simulated cargo and 10% loss are assumptions."""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    prepare_prediction_inputs,
)
from src.schema import TARGET_COLUMN, WEATHER_SEVERITY

ALERT_LINE, LOSS_SHARE = .60, .10

SLOWDOWN = {
    "Clear": 1.0,
    "Rain": 1.2,
    "Fog": 1.5,
    "Storm": 3.0,
    "Hurricane": 8.0,
}

_DEFINITIONS = [
    ("monitor", "No action", {}, 0),
    (
        "refrigerant_refresh",
        "Replenish dry ice",
        {"buffer_hours": 24},
        1500,
    ),
    (
        "carrier_review",
        "Switch to a top-rated carrier",
        {"carrier_score": .95},
        2000,
    ),
    (
        "route_review",
        "Reroute off the high-risk corridor",
        {"risk_cap": 3.0},
        5000,
    ),
    (
        "carrier_and_route",
        "Carrier and reroute together",
        {"carrier_score": .95, "risk_cap": 3.0},
        7000,
    ),
    (
        "weather_contingency",
        "Hold or route around the weather",
        {"weather_at_worst": "Rain"},
        8000,
    ),
]

ACTION_CATALOG = {
    key: dict(
        action_id=key,
        title=title,
        changes=change,
        cost_usd=float(cost),
        rationale=(
            "Notebook 05 hypothetical scenario; "
            "availability and costs are not independently verified."
        ),
    )
    for key, title, change, cost in _DEFINITIONS
}

ACTIONS = ACTION_CATALOG


def get_action_catalog():
    return [
        dict(item, changes=dict(item["changes"]))
        for item in ACTION_CATALOG.values()
    ]


def _finite_number(value, name, minimum=0.0, maximum=None):
    if isinstance(
        value, (bool, np.bool_, complex, np.complexfloating)
    ):
        raise ShipmentValidationError(
            f"{name} must be a real number."
        )

    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
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


def _config(model):
    return getattr(model, "metadata", {}) or {}


def _missing(value):
    return (
        value is None
        or value is pd.NA
        or (
            isinstance(value, (float, np.floating))
            and math.isnan(value)
        )
    )


def _validate_context(frame):
    for column in [
        "Elapsed_Transit_Hours_Simulated",
        "Remaining_Buffer_Hours_Simulated",
        "Cargo_Value_USD_Simulated",
    ]:
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: (
                    np.nan
                    if _missing(value)
                    else _finite_number(value, column)
                )
            )

    return frame


def _clean_shipment(shipment_data, model=None):
    if isinstance(shipment_data, pd.Series):
        shipment_data = shipment_data.to_dict()

    if not isinstance(shipment_data, Mapping):
        raise ShipmentValidationError(
            "Supply one shipment dictionary or Series."
        )

    prepared = prepare_prediction_inputs(
        pd.DataFrame([dict(shipment_data)]),
        config=_config(model),
    )

    return _validate_context(prepared).iloc[0].to_dict()


def predict_failure_risk(features_df, model):
    from src.model import predict_failure_risk as predict
    return predict(features_df, model)


def _predict_one(shipment, model):
    if model is None:
        raise ShipmentValidationError(
            "Load the existing notebook 05 artifact "
            "before comparing scenarios."
        )

    from src.features import extract_features

    prediction = predict_failure_risk(
        extract_features(
            pd.DataFrame([shipment]),
            config=_config(model),
        ),
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


def apply_action(
    shipments,
    buffer_hours=0,
    carrier_score=None,
    risk_cap=None,
    weather_at_worst=None,
):
    """Exact notebook 05 transforms; altered weather/duration are hypothetical."""
    changed = _validate_context(
        prepare_prediction_inputs(shipments).copy()
    )

    if buffer_hours:
        if "Remaining_Buffer_Hours_Simulated" not in changed:
            raise ShipmentValidationError(
                "Dry-ice scenario requires "
                "Remaining_Buffer_Hours_Simulated."
            )

        changed["Remaining_Buffer_Hours_Simulated"] = (
            changed["Remaining_Buffer_Hours_Simulated"]
            + buffer_hours
        )

    if carrier_score is not None:
        changed["Carrier_Reliability_Score"] = (
            changed["Carrier_Reliability_Score"].clip(
                lower=carrier_score
            )
        )

    if risk_cap is not None:
        changed["Geopolitical_Risk_Score"] = (
            changed["Geopolitical_Risk_Score"].clip(
                upper=risk_cap
            )
        )

    if weather_at_worst is not None:
        severity = changed["Weather_Condition"].map(
            WEATHER_SEVERITY
        )
        worse = severity > WEATHER_SEVERITY[weather_at_worst]

        factor = (
            SLOWDOWN[weather_at_worst]
            / changed.loc[worse, "Weather_Condition"].map(SLOWDOWN)
        )

        changed.loc[worse, "Lead_Time_Days"] = (
            changed.loc[worse, "Lead_Time_Days"] * factor
        )
        changed.loc[worse, "Weather_Condition"] = weather_at_worst

    return changed


def simulate_intervention(shipment_data, action_id, model):
    if action_id not in ACTION_CATALOG:
        aliases = {
            item["title"]: key
            for key, item in ACTION_CATALOG.items()
        }
        action_id = aliases.get(action_id)

    if action_id is None or action_id not in ACTION_CATALOG:
        raise ShipmentValidationError(
            "Unknown action; use get_action_catalog()."
        )

    original = _clean_shipment(shipment_data, model)
    action = ACTION_CATALOG[action_id]

    scenario = apply_action(
        pd.DataFrame([original]),
        **action["changes"],
    ).iloc[0].to_dict()

    fields = [
        "Carrier_Reliability_Score",
        "Geopolitical_Risk_Score",
        "Weather_Condition",
        "Lead_Time_Days",
        "Remaining_Buffer_Hours_Simulated",
    ]

    changed = {
        field: dict(
            before=original[field],
            after=scenario[field],
        )
        for field in fields
        if field in original
        and not (
            pd.isna(original[field])
            and pd.isna(scenario[field])
        )
        and original[field] != scenario[field]
    }

    before = _predict_one(original, model)
    after = _predict_one(scenario, model)
    cargo = original.get("Cargo_Value_USD_Simulated")

    reduction = (
        None
        if cargo is None or pd.isna(cargo)
        else (
            (before - after)
            * LOSS_SHARE
            * _finite_number(cargo, "simulated cargo value")
        )
    )

    net = (
        None
        if reduction is None
        else reduction - action["cost_usd"]
    )

    if net is not None and not math.isfinite(net):
        raise ShipmentValidationError(
            "Simulated cargo value exceeds finite scenario arithmetic."
        )

    from src.features import add_features

    featured = add_features(
        pd.DataFrame([scenario]),
        config=_config(model),
    ).iloc[0]

    context = {
        name: None if pd.isna(featured[name]) else float(featured[name])
        for name in ["hours_to_go", "cold_gap_hours"]
    }

    return dict(
        action_id=action_id,
        title=action["title"],
        target=TARGET_COLUMN,
        before_probability=before,
        after_probability=after,
        probability_change=before - after,
        action_cost_usd=action["cost_usd"],
        estimated_net_benefit_usd=net,
        estimated_loss_reduction_usd=reduction,
        changed_fields=changed,
        simulated_context=context,
        scenario_timing_inconsistent=(
            context["hours_to_go"] is not None
            and context["hours_to_go"] < 0
        ),
        assumptions=[
            (
                "Notebook 05: costs are examples; disruption loss "
                "is assumed to be 10% of simulated cargo value."
            ),
            (
                "Carrier floor 0.95, route cap 3, and weather no worse "
                "than Rain are hypothetical feasible alternatives."
            ),
            (
                "Weather scenario also scales full Lead_Time_Days "
                "by notebook slowdown ratio; "
                "simulated elapsed time stays fixed."
            ),
            (
                "Dry ice adds 24 simulated remaining-buffer hours only; "
                "it cannot change this disruption model's probability. "
                "Verify product-compatible coolant."
            ),
        ],
        note=(
            "Signed notebook model scenario, not causal effectiveness "
            "or guaranteed savings. Simulated cold-gap hours "
            "are not measured temperature. Negative hours_to_go "
            "means the duration scenario conflicts with simulated "
            "elapsed history. No action dispatched."
        ),
    )


def get_recommended_action(
    risk_category,
    risk_prob,
    buffer_hours=None,
    *,
    shipment_data=None,
    model=None,
):
    """Pass shipment_data and model to reproduce notebook's maximum net-saving choice."""
    probability = _finite_number(
        risk_prob, "risk_prob", maximum=1
    )

    if (shipment_data is None) != (model is None):
        raise ShipmentValidationError(
            "Provide both shipment_data and the loaded notebook model."
        )

    if model is not None:
        outcomes = [
            simulate_intervention(shipment_data, key, model)
            for key in ACTION_CATALOG
        ]

        if any(
            item["estimated_net_benefit_usd"] is None
            for item in outcomes
        ):
            raise ShipmentValidationError(
                "Notebook best-action comparison requires "
                "Cargo_Value_USD_Simulated."
            )

        selected = (
            outcomes[0]
            if outcomes[0]["before_probability"] < ALERT_LINE
            else max(
                outcomes,
                key=lambda item: item["estimated_net_benefit_usd"],
            )
        )

        return dict(
            ACTION_CATALOG[selected["action_id"]],
            simulation=selected,
            selection_method=(
                "Notebook 05 maximum assumed net saving; "
                "no-action wins when alternatives do not pay."
            ),
        )

    mapping = {
        "Weather exposure": "weather_contingency",
        "Carrier reliability": "carrier_review",
        "Geopolitical exposure": "route_review",
        "Combined exposure": "carrier_and_route",
    }

    action = (
        "monitor"
        if probability < ALERT_LINE
        else mapping.get(risk_category, "monitor")
    )

    return dict(
        ACTION_CATALOG[action],
        selection_method=(
            "Candidate only; shipment and model are needed "
            "for notebook net-saving selection."
        ),
        note=(
            "Uses notebook 60% alert line. "
            "buffer_hours alone is not a risk-model input."
        ),
    )


def check_sla_status(
    risk_prob,
    buffer_hours=None,
    corrective_proof=False,
):
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
        status="Not implemented by notebook 05",
        note=(
            "Notebook disruption probability and simulated buffer "
            "do not establish a medical SLA breach. "
            "No notices, escrow or payment actions."
        ),
    )