"""Operational heuristics and signed model sensitivity."""
import math

from src.preprocessing import (
    ShipmentValidationError,
)
from src.recommendations import (
    _clean_shipment,
    _predict_one,
)


def _domain_indicators(shipment):
    eta = shipment["Transit_Duration"]
    hold = shipment["Buffer_of_Package"]
    plan = shipment["Planned_Transit_Hours"]
    deadline = (
        shipment["Lead-Time in days"] * 24
    )

    if not math.isfinite(deadline):
        raise ShipmentValidationError(
            "Lead time exceeds finite deadline arithmetic."
        )

    indicators = []

    def add(category, feature, severity, description):
        indicators.append({
            "category": category,
            "feature": feature,
            "severity": min(
                max(severity, 0), 1
            ),
            "description": description,
        })

    if hold < eta:
        add(
            "Cold-chain excursion",
            "Buffer_of_Package",
            (eta - hold) / max(eta, .5),
            (
                f"Total qualified hold {hold:g} h "
                f"versus estimated total journey {eta:g} h: "
                f"margin {hold - eta:g} h. "
                "No core temperature is measured."
            ),
        )

    reliability = (
        shipment["Carrier_Reliability_Score"]
    )
    corridor = (
        shipment["Geopolitical_Risk"]
    )
    scores = {}

    if eta > deadline or eta > plan:
        scores["Transit_Duration"] = max(
            (eta - deadline)
            / max(deadline, .5),
            (eta - plan)
            / max(plan, .5),
        )

    if reliability < .65:
        scores["Carrier_Reliability_Score"] = (
            1 - reliability
        )

    if corridor >= 6:
        scores["Geopolitical_Risk"] = (
            corridor / 10
        )

    if scores:
        field = max(
            scores, key=scores.get
        )

        add(
            "Transport delay",
            field,
            scores[field],
            (
                f"Estimated total journey {eta:g} h; "
                f"planned {plan:g} h; "
                f"deadline {deadline:g} h; "
                f"known ETA-plan difference {eta - plan:g} h. "
                f"Carrier {reliability:g}, "
                f"corridor {corridor:g}/10 "
                "are review indicators."
            ),
        )

    handovers = (
        shipment["Handover_Count"]
    )

    if handovers >= 3:
        add(
            "Handover delay",
            "Handover_Count",
            handovers / 6,
            (
                f"{handovers:g} planned handovers "
                "increase handling exposure; "
                "actual dwell time and delay cause are unknown."
            ),
        )

    return sorted(
        indicators,
        key=lambda item: item["severity"],
        reverse=True,
    )


def explain_prediction(
    shipment_data,
    model=None,
):
    shipment = _clean_shipment(
        shipment_data
    )
    indicators = _domain_indicators(
        shipment
    )

    category = (
        "Combined risk"
        if len(indicators) > 1
        else indicators[0]["category"]
        if indicators
        else "Routine monitoring"
    )

    result = {
        "category": category,
        "domain_indicators": indicators,
        "category_method": (
            "Heuristic operational indicators; "
            "not verified causal diagnosis"
        ),
        "note": (
            "ETA is estimated total journey time; "
            "buffer is total qualified packaging capacity. "
            "No elapsed time, live countdown or measured "
            "temperature is inferred."
        ),
    }

    if model is None:
        return dict(
            result,
            method=(
                "Heuristic domain indicators "
                "(no model supplied)"
            ),
            contribution_unit=(
                "heuristic severity score"
            ),
            drivers=[
                {
                    "feature": item["feature"],
                    "description": item["description"],
                    "contribution": item["severity"],
                }
                for item in indicators[:3]
            ],
        )

    candidates = [
        (
            "Buffer_of_Package",
            max(
                shipment["Buffer_of_Package"],
                shipment["Transit_Duration"] + 6,
            ),
        ),
        (
            "Carrier_Reliability_Score",
            max(
                shipment["Carrier_Reliability_Score"],
                .95,
            ),
        ),
        (
            "Geopolitical_Risk",
            min(
                shipment["Geopolitical_Risk"],
                2,
            ),
        ),
        (
            "Transit_Duration",
            shipment["Planned_Transit_Hours"],
        ),
        (
            "Handover_Count",
            min(
                shipment["Handover_Count"],
                2,
            ),
        ),
    ]

    baseline = _predict_one(
        shipment, model
    )
    drivers = []

    for field, reference in candidates:
        if shipment[field] == reference:
            continue

        comparison = dict(shipment)
        comparison[field] = reference

        probability = _predict_one(
            comparison, model
        )

        drivers.append({
            "feature": field,
            "original_value": shipment[field],
            "reference_value": reference,
            "counterfactual_probability": probability,
            "contribution": baseline - probability,
            "description": (
                f"Assumed one-field comparison: {field} "
                f"{shipment[field]} → {reference}; "
                f"modeled probability {baseline:.1%} "
                f"→ {probability:.1%}. "
                "Not a proven cause or intervention effect."
            ),
        })

    drivers.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True,
    )

    return dict(
        result,
        method=(
            "One-at-a-time model sensitivity "
            "(explicit references; not SHAP)"
        ),
        contribution_unit=(
            "probability difference"
        ),
        baseline_probability=baseline,
        drivers=drivers[:3],
    )