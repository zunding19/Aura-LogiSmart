"""Exposure heuristics and signed model sensitivities, never failure diagnoses."""
from src.recommendations import _clean_shipment, _predict_one
from src.schema import TARGET_COLUMN

WEATHER_SEVERITY = {
    "Clear": 0.0,
    "Rain": .25,
    "Fog": .4,
    "Storm": .75,
    "Hurricane": 1.0,
}


def _domain_indicators(shipment):
    indicators = []

    def add(category, feature, severity, description):
        indicators.append(dict(
            category=category,
            feature=feature,
            severity=severity,
            description=description,
        ))

    weather = shipment["Weather_Condition"]
    if weather != "Clear":
        add(
            "Weather exposure",
            "Weather_Condition",
            WEATHER_SEVERITY[weather],
            f"Recorded weather is {weather}; its severity is an explicit "
            "prototype rule, not a learned probability.",
        )

    reliability = shipment["Carrier_Reliability_Score"]
    if reliability < .65:
        add(
            "Carrier reliability",
            "Carrier_Reliability_Score",
            1 - reliability,
            f"Recorded carrier reliability {reliability:g} is below "
            "the heuristic 0.65 review threshold.",
        )

    exposure = shipment["Geopolitical_Risk_Score"]
    if exposure >= 6:
        add(
            "Geopolitical exposure",
            "Geopolitical_Risk_Score",
            exposure / 10,
            f"Recorded corridor exposure {exposure:g}/10 meets "
            "the heuristic 6/10 review threshold.",
        )

    return sorted(
        indicators,
        key=lambda item: item["severity"],
        reverse=True,
    )


def explain_prediction(shipment_data, model=None) -> dict:
    """Contributions are p(original)-p(one-field reference).

    Only three explicit references are examined. This is not complete
    attribution, SHAP, causal diagnosis or a proven intervention effect.
    """
    shipment = _clean_shipment(shipment_data)
    indicators = _domain_indicators(shipment)

    category = (
        "Combined exposure"
        if len(indicators) > 1
        else indicators[0]["category"]
        if indicators
        else "Routine monitoring"
    )

    result = dict(
        category=category,
        target=TARGET_COLUMN,
        domain_indicators=indicators,
        category_method=(
            "Recorded-input exposure heuristics; "
            "not verified causal diagnosis"
        ),
        note=(
            "The label records disruption, not spoilage or clinical harm. "
            "No thermal state, ETA or hospital urgency is inferred. "
            "Routine monitoring means these heuristic thresholds are "
            "absent, not that model risk must be low."
        ),
    )

    if model is None:
        return dict(
            result,
            method="Heuristic exposure indicators (no model supplied)",
            contribution_unit="heuristic severity score",
            drivers=[
                dict(
                    feature=item["feature"],
                    description=item["description"],
                    contribution=item["severity"],
                )
                for item in indicators[:3]
            ],
        )

    candidates = [
        ("Weather_Condition", "Clear"),
        (
            "Carrier_Reliability_Score",
            max(shipment["Carrier_Reliability_Score"], .95),
        ),
        (
            "Geopolitical_Risk_Score",
            min(shipment["Geopolitical_Risk_Score"], 2),
        ),
    ]

    baseline = _predict_one(shipment, model)
    drivers = []

    for field, reference in candidates:
        if shipment[field] == reference:
            continue

        comparison = dict(shipment)
        comparison[field] = reference
        probability = _predict_one(comparison, model)

        drivers.append(dict(
            feature=field,
            original_value=shipment[field],
            reference_value=reference,
            counterfactual_probability=probability,
            contribution=baseline - probability,
            description=(
                f"Hypothetical one-field reference: {field} "
                f"{shipment[field]} → {reference}; disruption probability "
                f"{baseline:.1%} → {probability:.1%}. "
                "Signed sensitivity, not a verified cause; "
                "weather is not controllable."
            ),
        ))

    drivers.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )

    return dict(
        result,
        method="One-at-a-time model sensitivity to three explicit references",
        contribution_unit="disruption probability difference",
        baseline_probability=baseline,
        drivers=drivers[:3],
        coverage=(
            "Weather, carrier reliability and geopolitical "
            "exposure references only."
        ),
    )