"""Plain-language reasons for a risk score: warning signs and one-change what-ifs."""
import math

import pandas as pd

from src.features import add_features
from src.schema import (
    TARGET_COLUMN,
    WEATHER_SEVERITY,
    TYPICAL_PACE,
    HIGH_RISK_SCORE,
    LOW_RELIABILITY,
)
from src.recommendations import (
    _clean_shipment,
    _predict_one,
    _config,
    SLOWDOWN,
)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


PLAIN_NAME = {
    "Weather_Condition": "Weather on the route",
    "Carrier_Reliability_Score": "Carrier reliability",
    "Geopolitical_Risk_Score": "Route risk",
}


def _what_if(field, shipment, changes, baseline, probability):
    """One sentence a coordinator can read without knowing the model."""
    if field == "Weather_Condition":
        fact = f"{shipment[field]} on the route."
        alternative = "In clear weather"
    elif field == "Carrier_Reliability_Score":
        fact = f"Carrier reliability is {shipment[field]:.2f} out of 1."
        alternative = f"With a top-rated carrier ({changes[field]:.2f})"
    else:
        fact = f"Route risk score is {shipment[field]:g} out of 10."
        alternative = f"On a calm route (score {changes[field]:g})"

    if abs(baseline - probability) < .01:
        return f"{fact} {alternative} the risk would barely change."

    return (
        f"{fact} {alternative} the risk would be "
        f"{probability:.0%} instead of {baseline:.0%}."
    )


def explain_prediction(shipment_data, model=None):
    shipment = _clean_shipment(shipment_data, model)
    config = _config(model)

    featured = add_features(
        pd.DataFrame([shipment]), config=config
    ).iloc[0]

    weather_map = config.get("weather_severity", WEATHER_SEVERITY)
    weather = shipment["Weather_Condition"]
    severity = weather_map.get(weather)
    indicators = []

    def add(category, feature, score, description):
        indicators.append(
            dict(
                category=category,
                feature=feature,
                severity=score,
                description=description,
            )
        )

    if severity is not None and severity >= 3:
        add(
            "Weather exposure",
            "Weather_Condition",
            severity / 4,
            f"{weather} on the route. Storms and hurricanes "
            "are the strongest warning sign.",
        )

    carrier = shipment["Carrier_Reliability_Score"]
    corridor = shipment["Geopolitical_Risk_Score"]
    low_reliability = config.get("low_reliability", LOW_RELIABILITY)
    high_risk = config.get("high_risk_score", HIGH_RISK_SCORE)

    if _finite(carrier) and carrier < low_reliability:
        add(
            "Carrier reliability",
            "Carrier_Reliability_Score",
            1 - carrier,
            f"Carrier reliability is {carrier:.2f}, below "
            f"the warning level of {low_reliability:.2f}.",
        )

    if _finite(corridor) and corridor >= high_risk:
        add(
            "Geopolitical exposure",
            "Geopolitical_Risk_Score",
            corridor / 10,
            f"Route risk score is {corridor:g} out of 10, at or above "
            f"the warning level of {high_risk:g}.",
        )

    indicators.sort(
        key=lambda item: item["severity"],
        reverse=True,
    )

    category = (
        "Combined exposure"
        if len(indicators) > 1
        else indicators[0]["category"]
        if indicators
        else "Routine monitoring"
    )

    context = {
        name: None if not _finite(featured[name]) else float(featured[name])
        for name in [
            "hours_to_go",
            "cold_gap_hours",
            "pace_vs_typical",
            "red_flag_count",
        ]
    }

    result = dict(
        category=category,
        target=TARGET_COLUMN,
        domain_indicators=indicators,
        category_method=(
            "Based on which warning signs are present "
            "(severe weather, risky route, weak carrier)."
        ),
        notebook_indicators=context,
        note=(
            "Thermal runway is simulated and is not part of the risk "
            "score. It shows how exposed the cargo is, not whether it "
            "will spoil. A shipment with no warning signs can still "
            "be at risk."
        ),
    )

    if model is None:
        return dict(
            result,
            summary=(
                indicators[0]["description"]
                if indicators
                else "No warning signs on this shipment."
            ),
            method="Warning signs only (no model loaded)",
            contribution_unit="warning strength, 0 to 1",
            drivers=[
                dict(
                    feature=item["feature"],
                    label=PLAIN_NAME[item["feature"]],
                    contribution=item["severity"],
                    description=item["description"],
                )
                for item in indicators[:3]
            ],
        )

    baseline = _predict_one(shipment, model)
    candidates = []

    # Clear weather also means the normal trip time for this mode and distance.
    typical = config.get(
        "typical_pace", TYPICAL_PACE
    ).get(shipment["Transport_Mode"])

    distance = shipment["Distance_km"]

    if (
        severity is not None
        and weather != "Clear"
        and typical is not None
        and _finite(distance)
    ):
        candidates.append(
            (
                "Weather_Condition",
                {
                    "Weather_Condition": "Clear",
                    "Lead_Time_Days": (
                        typical
                        * SLOWDOWN["Clear"]
                        * distance
                        / 1000
                        / 24
                    ),
                },
            )
        )

    if _finite(carrier) and carrier < .95:
        candidates.append(
            (
                "Carrier_Reliability_Score",
                {"Carrier_Reliability_Score": .95},
            )
        )

    if _finite(corridor) and corridor > 1:
        candidates.append(
            (
                "Geopolitical_Risk_Score",
                {"Geopolitical_Risk_Score": 1.0},
            )
        )

    drivers = []

    for field, changes in candidates:
        scenario = dict(shipment)
        scenario.update(changes)
        probability = _predict_one(scenario, model)

        drivers.append(
            dict(
                feature=field,
                label=PLAIN_NAME[field],
                contribution=baseline - probability,
                original_value=shipment[field],
                reference_value=changes[field],
                changed_fields={
                    key: dict(before=shipment[key], after=value)
                    for key, value in changes.items()
                },
                counterfactual_probability=probability,
                description=_what_if(
                    field, shipment, changes, baseline, probability
                ),
            )
        )

    drivers.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )

    main = next(
        (item for item in drivers if item["contribution"] >= .05),
        None,
    )

    return dict(
        result,
        baseline_probability=baseline,
        summary=(
            f"{baseline:.0%} disruption risk. "
            + (
                f"Main reason: {main['description']}"
                if main
                else "No single input stands out; "
                "the risk comes from the overall mix."
            )
        ),
        drivers=drivers[:3],
        method=(
            "What-if comparison: change one input to a safe value "
            "and score the shipment again."
        ),
        contribution_unit="points of disruption risk removed",
        coverage="Weather, carrier reliability and route risk.",
    )