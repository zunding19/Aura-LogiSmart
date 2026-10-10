#produce operational risks categories and the top 3 model sensitives
import math

from src.preprocessing import ShipmentValidationError
from src.recommendations import (
    _clean_shipment,
    _predict_one,
)


def _domain_indicators(shipment: dict) -> list[dict]:
    buffer = shipment["Buffer_of_Package"]
    deadline = shipment["Lead-Time in days"] * 24.0

    if not math.isfinite(deadline):
        raise ShipmentValidationError(
            "Lead time is too large "
            "to compute a finite deadline scenario."
        )

    elapsed = shipment["Transit_Duration"]
    remaining = max(deadline - elapsed, 0.0)
    reliability = shipment["Carrier_Reliability_Score"]
    geopolitical = shipment["Geopolitical_Risk"]
    indicators = []

    if buffer < 3.0 or buffer < remaining:
        severity = max(
            max(0.0, (3.0 - buffer) / 3.0),
            max(
                0.0,
                (remaining - buffer) / max(remaining, 0.5),
            ),
        )
        indicators.append({
            "category": "Cold-chain excursion",
            "feature": "Buffer_of_Package",
            "severity": severity,
            "description": (
                f"{buffer:.1f} h qualified hold buffer "
                f"versus {remaining:.1f} h remaining "
                "until the planned deadline; "
                "packaging compatibility and actual ETA "
                "need operator confirmation."
            ),
        })

    elapsed_ratio = elapsed / max(deadline, 0.5)
    transport_scores = {}

    if elapsed_ratio >= 0.8:
        transport_scores["Transit_Duration"] = min(
            elapsed_ratio, 1.0
        )
    if reliability < 0.65:
        transport_scores["Carrier_Reliability_Score"] = (
            1.0 - reliability
        )
    if geopolitical >= 6.0:
        transport_scores["Geopolitical_Risk"] = (
            geopolitical / 10.0
        )

    if transport_scores:
        strongest_field = max(
            transport_scores,
            key=transport_scores.get,
        )
        indicators.append({
            "category": "Transport delay",
            "feature": strongest_field,
            "severity": transport_scores[strongest_field],
            "description": (
                f"Elapsed/planned-time ratio {elapsed_ratio:.2f}, "
                f"carrier reliability {reliability:.2f}, "
                f"corridor risk {geopolitical:.1f}/10 "
                "indicate an operational review; "
                "no actual delay cause is verified."
            ),
        })

    if (
        shipment["Transport_Mode"] == "Multi-Modal Air-to-Land"
        and (reliability < 0.75 or buffer < 6.0)
    ):
        indicators.append({
            "category": "Handover delay",
            "feature": "Transport_Mode",
            "severity": max(
                1.0 - reliability,
                max(0.0, (6.0 - buffer) / 6.0),
            ),
            "description": (
                "Multi-modal routing with reliability below 0.75 "
                "or packaging buffer below 6 h "
                "calls for handover review; "
                "actual dwell time is unavailable."
            ),
        })

    return sorted(
        indicators,
        key=lambda indicator: indicator["severity"],
        reverse=True,
    )


def explain_prediction(
    shipment_data,
    model=None,
) -> dict:
    shipment = _clean_shipment(shipment_data)
    indicators = _domain_indicators(shipment)

    category = (
        "Combined risk"
        if len(indicators) > 1
        else indicators[0]["category"]
        if indicators
        else "Routine monitoring"
    )

    result = {
        "category": category,
        "category_method": (
            "Heuristic operational indicators, "
            "not verified causal diagnosis"
        ),
        "domain_indicators": indicators,
        "note": (
            "Packaging hold time and planned deadlines "
            "are scenario proxies. No temperature measurement, "
            "prediction horizon or failure cause is inferred."
        ),
    }

    if model is None:
        result.update({
            "method": (
                "Heuristic domain indicators "
                "(no model supplied)"
            ),
            "contribution_unit": "heuristic severity score",
            "drivers": [
                {
                    "feature": item["feature"],
                    "description": item["description"],
                    "contribution": item["severity"],
                }
                for item in indicators[:3]
            ],
        })
        return result

    deadline = shipment["Lead-Time in days"] * 24.0
    remaining = max(
        deadline - shipment["Transit_Duration"], 0.0
    )

    candidates = [
        (
            "Buffer_of_Package",
            max(
                shipment["Buffer_of_Package"],
                remaining + 6.0,
                6.0,
            ),
        ),
        (
            "Carrier_Reliability_Score",
            max(
                shipment["Carrier_Reliability_Score"],
                0.95,
            ),
        ),
        (
            "Geopolitical_Risk",
            min(shipment["Geopolitical_Risk"], 2.0),
        ),
        (
            "Transit_Duration",
            min(
                shipment["Transit_Duration"],
                deadline * 0.5,
            ),
        ),
        (
            "Fuel_Price_Index",
            min(shipment["Fuel_Price_Index"], 1.0),
        ),
        ("Transport_Mode", "Air"),
    ]

    baseline_probability = _predict_one(shipment, model)
    drivers = []

    for field, reference_value in candidates:
        if shipment[field] == reference_value:
            continue

        counterfactual = dict(shipment)
        counterfactual[field] = reference_value
        counterfactual_probability = _predict_one(
            counterfactual, model
        )
        contribution = (
            baseline_probability - counterfactual_probability
        )

        drivers.append({
            "feature": field,
            "description": (
                f"One-field sensitivity: {field} "
                f"{shipment[field]!s} → {reference_value!s}; "
                "modeled failure probability "
                f"{baseline_probability:.1%} → "
                f"{counterfactual_probability:.1%}. "
                "An assumed comparison, not a proven cause "
                "or real-world intervention."
            ),
            "contribution": contribution,
            "original_value": shipment[field],
            "reference_value": reference_value,
            "counterfactual_probability": (
                counterfactual_probability
            ),
        })

    drivers.sort(
        key=lambda driver: abs(driver["contribution"]),
        reverse=True,
    )

    result.update({
        "method": (
            "One-at-a-time model sensitivity "
            "(explicit scenario references; not SHAP)"
        ),
        "contribution_unit": "probability difference",
        "baseline_probability": baseline_probability,
        "drivers": drivers[:3],
    })

    return result