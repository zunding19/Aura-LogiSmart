"""Notebook 05 warning indicators and signed scenario sensitivities, not causes."""
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
            f"Notebook severe-weather warning: {weather}, "
            f"ordinal severity {severity}/4.",
        )

    carrier = shipment["Carrier_Reliability_Score"]
    corridor = shipment["Geopolitical_Risk_Score"]

    if (
        _finite(carrier)
        and carrier < config.get("low_reliability", LOW_RELIABILITY)
    ):
        add(
            "Carrier reliability",
            "Carrier_Reliability_Score",
            1 - carrier,
            f"Carrier score {carrier:g} is below "
            "notebook warning threshold 0.70.",
        )

    if (
        _finite(corridor)
        and corridor >= config.get("high_risk_score", HIGH_RISK_SCORE)
    ):
        add(
            "Geopolitical exposure",
            "Geopolitical_Risk_Score",
            corridor / 10,
            f"Corridor score {corridor:g}/10 "
            "meets notebook warning threshold 6.",
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
            "Notebook red-flag rules, not verified disruption causes"
        ),
        notebook_indicators=context,
        note=(
            "Thermal gap uses simulated elapsed/buffer fields "
            "and stays outside both saved models. "
            "It is not a temperature or spoilage prediction. "
            "No warning flags does not guarantee low model risk."
        ),
    )

    if model is None:
        return dict(
            result,
            method="Notebook heuristic warning indicators (no model supplied)",
            contribution_unit="heuristic severity score",
            drivers=[
                dict(
                    feature=item["feature"],
                    contribution=item["severity"],
                    description=item["description"],
                )
                for item in indicators[:3]
            ],
        )

    baseline = _predict_one(shipment, model)
    candidates = []

    # Notebook cell 24 couples a weather reference to its mode/distance duration.
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
                contribution=baseline - probability,
                original_value=shipment[field],
                reference_value=changes[field],
                changed_fields={
                    key: dict(before=shipment[key], after=value)
                    for key, value in changes.items()
                },
                counterfactual_probability=probability,
                description=(
                    f"Notebook-style reference {changes}: "
                    f"probability {baseline:.1%} → {probability:.1%}. "
                    "Weather comparison also assumes "
                    "the notebook mode/distance duration; "
                    "this is not a causal attribution."
                ),
            )
        )

    drivers.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )

    return dict(
        result,
        baseline_probability=baseline,
        drivers=drivers[:3],
        method=(
            "Signed notebook-style model scenario sensitivities "
            "(not SHAP or permutation importance)"
        ),
        contribution_unit="disruption probability difference",
        coverage=(
            "Weather with assumed duration, carrier reliability "
            "and corridor references only."
        ),
    )