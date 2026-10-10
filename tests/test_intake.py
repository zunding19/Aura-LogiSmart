"""The order form's field names go in; a complete, JSON-safe assessment comes out."""
import json

import pytest

from src.intake import (
    INSUFFICIENT,
    assess_order,
    estimate_distance_km,
    rank_orders,
    thermal_runway_curve,
)
from src.model import load_model

FORM_ORDER = {
    "Shipment_ID": "BIO-8841-KL",
    "Place_of_Origin": "Frankfurt CargoCity",
    "Place_of_Destination": "Kuala Lumpur General Hospital",
    "Transport_Mode": "Multi-Modal Air-to-Land",
    "Lead-Time in days": 3,
    "Transit_Duration": 60.0,
    "Stock_Volume": 12,
    "Buffer_of_Package": 2.8,
    "Storage_standard": "2°C to 8°C",
    "Cargo_Value_USD": 85000,
    "Hospital_Urgency_Tier": "Critical Emergency",
}


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_form_order_is_assessed_end_to_end(model):
    result = assess_order(FORM_ORDER, model)

    assert result["status"] == "ok"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert 0 <= result["risk"]["p_failure"] <= 1
    assert result["risk"]["risk_level"] in {"Low", "Medium", "High", "Critical"}

    # No weather was entered, so the backup model scores it.
    assert result["risk"]["model_variant"] == "weather_missing"
    assert {"Distance_km", "Geopolitical_Risk_Score", "Weather_Condition"} <= set(
        result["inferred"]
    )
    assert 9000 < result["shipment"]["Distance_km"] < 12000
    assert result["shipment"]["Transport_Mode"] == "Air"

    # 72 planned hours, 60 elapsed, 2.8 hours of buffer: 9.2 hours short.
    assert result["thermal"]["hours_to_go"] == pytest.approx(12.0)
    assert result["thermal"]["runway_hours"] == pytest.approx(-9.2)
    assert result["thermal"]["curve"]["buffer_ends_at_hour"] == pytest.approx(2.8)
    assert "excursion" in result["failure_mode"]["mode"].lower()
    assert result["explanation"]["top_reasons"][0]["kind"] == "thermal runway"

    p = result["risk"]["p_failure"]
    assert result["priority"]["urgency_score"] == pytest.approx(
        p * 85000 * 2.5 / 2.8
    )

    options = {item["action_id"]: item for item in result["actions"]["options"]}
    assert options["refrigerant_refresh"]["covers_thermal_gap"]
    assert options["refrigerant_refresh"]["cargo_protected_usd"] == 85000
    assert options["refrigerant_refresh"]["risk_after"] == pytest.approx(p)
    assert result["actions"]["thermal_action_id"] == "refrigerant_refresh"
    assert options["refrigerant_refresh"]["action_cost_usd"] == 120
    assert options["direct_flight"]["kind"] == "risk"
    assert set(options) == {
        "monitor", "cold_storage_transfer", "direct_flight",
        "refrigerant_refresh", "refrigerated_ground", "stock_diversion",
    }
    assert result["scorecard"]["precision"] == model.metrics["precision"]


def test_entered_weather_uses_the_main_model(model):
    calm = assess_order(dict(FORM_ORDER, Weather="Clear"), model)
    storm = assess_order(dict(FORM_ORDER, Weather="Hurricane"), model)

    assert calm["risk"]["model_variant"] == "primary"
    assert storm["risk"]["p_failure"] > calm["risk"]["p_failure"]


def test_entered_values_are_not_overwritten(model):
    result = assess_order(
        dict(FORM_ORDER, Distance=5000, Geopolitical_Risk=8.5), model
    )

    assert result["shipment"]["Distance_km"] == 5000
    assert result["shipment"]["Geopolitical_Risk_Score"] == 8.5
    assert "Distance_km" not in result["inferred"]


def test_live_risk_source_is_used_and_failures_fall_back(model):
    live = assess_order(FORM_ORDER, model, risk_provider=lambda a, b: 7.0)
    assert live["shipment"]["Geopolitical_Risk_Score"] == 7.0

    def broken(origin, destination):
        raise RuntimeError("service down")

    fallback = assess_order(FORM_ORDER, model, risk_provider=broken)
    assert fallback["status"] == "ok"
    assert "World Bank" in fallback["inferred"]["Geopolitical_Risk_Score"]
    # Frankfurt to Kuala Lumpur: Germany (0.208) is the less stable end.
    assert fallback["shipment"]["Geopolitical_Risk_Score"] == 4.6


@pytest.mark.parametrize(
    "order",
    [
        {},
        {"Place_of_Origin": "Nowhere", "Place_of_Destination": "Elsewhere"},
        dict(FORM_ORDER, **{"Lead-Time in days": "soon"}),
        dict(FORM_ORDER, Buffer_of_Package=-4),
        dict(FORM_ORDER, Transport_Mode="Teleport"),
        "not an order",
    ],
)
def test_bad_or_missing_input_never_crashes(model, order):
    result = assess_order(order, model)

    assert result["status"] == INSUFFICIENT
    assert result["reason"]


def test_orders_are_ranked_most_urgent_first(model):
    routine = dict(
        FORM_ORDER,
        Shipment_ID="ROUTINE",
        Hospital_Urgency_Tier="Routine Restock",
        Buffer_of_Package=40,
    )
    ranked = rank_orders([routine, {}, FORM_ORDER], model)

    assert ranked[0]["shipment"]["Shipment_ID"] == "BIO-8841-KL"
    assert ranked[1]["shipment"]["Shipment_ID"] == "ROUTINE"
    assert ranked[2]["status"] == INSUFFICIENT


def test_powered_container_buffer_is_not_a_countdown(model):
    import pandas as pd

    data = pd.read_csv("data/processed/aura_shipments.csv")
    reefer = data[
        data.Transport_Mode.eq("Road")
        & data.Product_Category.eq("Pharmaceuticals")
    ].iloc[0]
    result = assess_order(reefer, model)

    assert result["status"] == "ok"
    assert result["thermal"]["buffer_counts_down"] is False
    assert result["thermal"]["runway_hours"] is None
    assert "excursion" not in result["failure_mode"]["mode"].lower()
    assert result["actions"]["thermal_action_id"] is None

    ambient = assess_order(
        data[
            data.Product_Category.eq("Textiles") & data.Transport_Mode.eq("Road")
        ].iloc[0],
        model,
    )
    assert ambient["status"] == "ok"
    assert len(ambient["actions"]["options"]) == 6

    # A road shipment cannot be moved to the next flight.
    flight = next(
        item for item in result["actions"]["options"]
        if item["action_id"] == "direct_flight"
    )
    assert flight["available"] is False
    assert flight["risk_after"] == flight["risk_before"]
    assert result["actions"]["recommended_action_id"] != "direct_flight"


def test_sea_and_rail_are_out_of_scope(model):
    import pandas as pd

    from src.intake import OUT_OF_SCOPE, triage_queue

    data = pd.read_csv("data/processed/aura_shipments.csv")
    for mode in ["Sea", "Rail"]:
        result = assess_order(data[data.Transport_Mode.eq(mode)].iloc[0], model)
        assert result["status"] == OUT_OF_SCOPE
        assert mode.lower() in result["reason"]

    queue = triage_queue(data, model)
    assert set(queue.Transport_Mode) <= {"Air", "Road"}
    assert queue.Product_Category.eq("Pharmaceuticals").all()
    assert queue.priority.is_monotonic_decreasing
    assert len(triage_queue(data, model, top=10)) == 10
    assert queue.iloc[0].Shipment_ID == "SC-12132"


def test_route_risk_from_world_bank_index():
    from src.intake import (
        PLACE_COUNTRY,
        PLACES,
        POLITICAL_STABILITY,
        estimate_route_risk,
        stability_to_risk,
    )

    # Every known place has a country, and every country has a value.
    assert set(PLACE_COUNTRY) == set(PLACES)
    assert set(PLACE_COUNTRY.values()) <= set(POLITICAL_STABILITY)

    assert stability_to_risk(2.5) == 0
    assert stability_to_risk(0) == 5
    assert stability_to_risk(-2.5) == 10
    assert stability_to_risk(-9) == 10

    score, source = estimate_route_risk("Singapore", "Jakarta depot")
    assert score == stability_to_risk(POLITICAL_STABILITY["Indonesia"]) == 6.5
    assert "Indonesia" in source
    assert estimate_route_risk("Mars", "Venus") == (None, None)


def test_helpers():
    assert estimate_distance_km("Mars", "Singapore", "Air") is None
    assert estimate_distance_km("Singapore", "Singapore depot", "Road") == 50

    curve = thermal_runway_curve(40, 10)
    assert curve["buffer_ends_at_hour"] is None
    assert curve["points"][0]["buffer_hours_left"] == 40
    assert thermal_runway_curve(None, 10) is None
