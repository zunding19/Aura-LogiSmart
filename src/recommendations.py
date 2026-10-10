"""Action scenarios, kept in step with notebook 05; fees, effects and 10% loss are assumptions."""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.preprocessing import (
    ShipmentValidationError,
    prepare_prediction_inputs,
)
from src.protocols import find_corrective_proof
from src.schema import TARGET_COLUMN, WEATHER_SEVERITY

ALERT_LINE, LOSS_SHARE = .60, .10
SLA_RISK_LINE, SLA_BUFFER_HOURS = .80, 3.0

SLOWDOWN = {
    "Clear": 1.0,
    "Rain": 1.2,
    "Fog": 1.5,
    "Storm": 3.0,
    "Hurricane": 8.0,
}

# kind says what an action changes: the risk score, the thermal runway, both or neither.
_DEFINITIONS = [
    (
        "monitor",
        "Keep monitoring",
        {},
        0,
        "watch",
        "No spend. Score the shipment again when the weather, "
        "the hours elapsed or the buffer change.",
    ),
    (
        "cold_storage_transfer",
        "Prioritise immediate cold-storage transfer",
        {"buffer_hours": 24},
        200,
        "thermal",
        "Moves the cargo into the hub cold room, which adds about "
        "24 hours of thermal runway. The disruption risk stays the same.",
    ),
    (
        "direct_flight",
        "Move to next available direct flight",
        {"weather_at_worst": "Rain", "risk_cap": 3.0},
        450,
        "risk",
        "Skips the stopover, so the shipment avoids the worst weather "
        "and the high-risk corridor, and the trip gets shorter.",
    ),
    (
        "refrigerant_refresh",
        "Hub refrigerant refresh (re-icing)",
        {"buffer_hours": 48},
        120,
        "thermal",
        "Hub crew replenishes the coolant, which adds about 48 hours "
        "of thermal runway. The disruption risk stays the same.",
    ),
    (
        "refrigerated_ground",
        "Assign refrigerated ground transport",
        {"carrier_score": .95, "buffer_hours": 72},
        500,
        "risk and thermal",
        "A dedicated temperature-controlled vehicle takes over: "
        "a dependable carrier and about 72 more hours of thermal runway.",
    ),
    (
        "stock_diversion",
        "Hospital emergency stock diversion",
        {},
        0,
        "backup",
        "Backup plan. A replacement batch is sent from a regional depot "
        "so the hospital is covered. It does not change this shipment, "
        "and its cost is not estimated.",
    ),
]

ACTION_CATALOG = {
    key: dict(
        action_id=key,
        title=title,
        changes=change,
        cost_usd=float(cost),
        kind=kind,
        effect=effect,
        rationale=effect,
    )
    for key, title, change, cost, kind, effect in _DEFINITIONS
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
        # Cargo with no thermal buffer (ambient) has nothing to extend.
        if "Remaining_Buffer_Hours_Simulated" not in changed:
            changed["Remaining_Buffer_Hours_Simulated"] = np.nan

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
        pd.DataFrame([original, scenario]),
        config=_config(model),
    )

    # A faster trip shortens the hours still to go in the same proportion;
    # the share of the trip already completed stays as it was.
    hours_to_go = featured["hours_to_go"].iloc[0] * (
        scenario["Lead_Time_Days"] / original["Lead_Time_Days"]
    )
    cold_gap = (
        featured["Remaining_Buffer_Hours_Simulated"].iloc[1] - hours_to_go
        if "Remaining_Buffer_Hours_Simulated" in featured
        else np.nan
    )

    context = {
        name: None if pd.isna(value) else round(float(value), 1)
        for name, value in [
            ("hours_to_go", hours_to_go),
            ("cold_gap_hours", cold_gap),
        ]
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
                "A disruption is assumed to cost 10% of the cargo value. "
                "Action costs are example figures."
            ),
            (
                "A direct flight leaves rain at worst and a route risk "
                "of 3 out of 10 at most. It also shortens the trip, so "
                "the hours still to go shrink in the same proportion."
            ),
            (
                "Refrigerated ground transport counts as a carrier "
                "with a reliability of 0.95."
            ),
            (
                "Cold storage adds 24 hours of thermal runway, re-icing "
                "48 and refrigerated ground transport 72. More cooling "
                "does not change the disruption risk, because it does "
                "not stop a storm or a delay."
            ),
        ],
        note=(
            "This is the model's estimate of a what-if, not a guaranteed "
            "saving. Cargo value and thermal runway are simulated. "
            "No action has been sent to anyone."
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
                "Highest net saving under the stated assumptions; "
                "keep monitoring wins when no action pays."
            ),
        )

    mapping = {
        "Weather exposure": "direct_flight",
        "Carrier reliability": "refrigerated_ground",
        "Geopolitical exposure": "direct_flight",
        "Combined exposure": "direct_flight",
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
    *,
    shipment_id=None,
    custody_records=None,
):
    """Simulated SLA rule: critical risk and a short buffer lock the carrier's escrow."""
    probability = _finite_number(risk_prob, "risk_prob", maximum=1)

    if not isinstance(corrective_proof, bool):
        raise ShipmentValidationError(
            "corrective_proof must be True or False."
        )

    buffer = (
        None
        if buffer_hours is None or _missing(buffer_hours)
        else _finite_number(buffer_hours, "buffer_hours")
    )

    proof = (
        None
        if shipment_id is None
        else find_corrective_proof(
            custody_records, shipment_id, SLA_BUFFER_HOURS
        )
    )
    proven = corrective_proof or proof is not None

    trigger = (
        probability > SLA_RISK_LINE
        and buffer is not None
        and buffer < SLA_BUFFER_HOURS
    )
    locked = trigger and not proven

    if locked:
        status = "SLA breach notice issued. Carrier payout escrow locked."
    elif trigger:
        status = "Corrective proof verified. Escrow released."
    else:
        status = "Within SLA."

    return dict(
        available=True,
        target=TARGET_COLUMN,
        trigger_condition=trigger,
        breach_warning=locked,
        simulated_escrow_locked=locked,
        corrective_proof_recorded=proven,
        corrective_proof=proof,
        status=status,
        breach_notice=(
            dict(
                shipment_id=shipment_id,
                text=(
                    f"Disruption risk is {probability:.0%} and only "
                    f"{buffer:.1f} hours of thermal buffer remain, with no "
                    "corrective action recorded. Carrier payout is held "
                    "until a cold-storage transfer, re-icing or "
                    "refrigerated transport handshake is logged."
                ),
            )
            if locked
            else None
        ),
        rule=(
            f"Risk above {SLA_RISK_LINE:.0%} and thermal buffer under "
            f"{SLA_BUFFER_HOURS:g} hours, with no corrective proof."
        ),
        note=(
            "Simulation for the demo. No real notice is sent "
            "and no payment is held."
        ),
    )
