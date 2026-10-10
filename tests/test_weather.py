"""Weather auto-fill and the adjustable alert line. No test here uses the network."""
import pandas as pd
import pytest

from src.intake import (
    alert_line_options,
    assess_order,
    lookup_weather,
    triage_queue,
)
from src.model import load_model
from src.weather import classify_weather, current_weather

ORDER = {
    "Place_of_Origin": "Frankfurt CargoCity",
    "Place_of_Destination": "Kuala Lumpur General Hospital",
    "Transport_Mode": "Air",
    "Lead-Time in days": 3,
    "Transit_Duration": 60.0,
    "Buffer_of_Package": 2.8,
    "Cargo_Value_USD": 85000,
    "Hospital_Urgency_Tier": "Critical Emergency",
}


@pytest.fixture(scope="module")
def model():
    return load_model()


def reading(code, wind):
    return {"current": {"weather_code": code, "wind_speed_10m": wind}}


@pytest.mark.parametrize(
    "code,wind,expected",
    [
        (0, 5, "Clear"),
        (3, 30, "Clear"),        # overcast is still clear for a shipment
        (45, 5, "Fog"),
        (61, 10, "Rain"),
        (73, 10, "Rain"),        # snow slows a shipment the way rain does
        (95, 10, "Storm"),       # thunderstorm
        (2, 62, "Storm"),        # gale-force wind under a clear sky
        (61, 118, "Hurricane"),
    ],
)
def test_weather_codes_map_to_the_models_five_conditions(code, wind, expected):
    assert classify_weather(code, wind) == expected


def test_route_takes_the_worse_end_and_survives_a_dead_service():
    points = [(50.03, 8.57), (3.14, 101.69)]

    condition, note = current_weather(
        points, fetch=lambda p: [reading(0, 5), reading(95, 20)]
    )
    assert condition == "Storm"
    assert "Open-Meteo" in note

    def down(points):
        raise OSError("no internet")

    assert current_weather(points, fetch=down) == (None, None)
    assert current_weather(points, fetch=lambda p: [{"oops": 1}, {}]) == (None, None)
    assert current_weather(points, fetch=lambda p: [reading(0, 5)]) == (None, None)

    assert lookup_weather("Mars", "Singapore") == (None, None)


def test_missing_weather_is_filled_and_entered_weather_is_kept(model):
    storm = lambda origin, destination: ("Storm", "Live reading (test).")

    filled = assess_order(ORDER, model, weather_provider=storm)
    assert filled["shipment"]["Weather_Condition"] == "Storm"
    assert filled["risk"]["model_variant"] == "primary"
    assert filled["inferred"]["Weather_Condition"] == "Live reading (test)."

    kept = assess_order(dict(ORDER, Weather="Clear"), model, weather_provider=storm)
    assert kept["shipment"]["Weather_Condition"] == "Clear"
    assert "Weather_Condition" not in kept["inferred"]

    # A failing or silly provider falls back to the backup model, never a crash.
    def broken(origin, destination):
        raise RuntimeError("service down")

    for provider in [broken, lambda o, d: ("Typhoon", "x"), lambda o, d: (None, None)]:
        result = assess_order(ORDER, model, weather_provider=provider)
        assert result["status"] == "ok"
        assert result["risk"]["model_variant"] == "weather_missing"

    # Without a provider nothing is looked up.
    assert assess_order(ORDER, model)["risk"]["model_variant"] == "weather_missing"


def test_live_weather_removes_the_slow_trip_false_alarm(model):
    # A normal 3-day air trip with no weather reads as High on the backup model.
    blank = assess_order(dict(ORDER, Transit_Duration=10, Buffer_of_Package=96), model)
    clear = assess_order(
        dict(ORDER, Transit_Duration=10, Buffer_of_Package=96), model,
        weather_provider=lambda o, d: ("Clear", "Live reading (test)."),
    )

    assert blank["risk"]["p_failure"] > .60
    assert clear["risk"]["p_failure"] < .40


def test_alert_line_options_and_adjustable_queue(model):
    options = {round(row["alert_line"], 2): row for row in alert_line_options(model)}

    # The 60% row is the published scorecard.
    assert options[.6]["precision"] == pytest.approx(model.metrics["precision"], abs=1e-4)
    assert options[.6]["recall"] == pytest.approx(model.metrics["recall"], abs=1e-4)
    assert options[.6]["false_alarms_per_1000"] == 77

    # Lowering the line catches more and raises more false alarms.
    lines = sorted(options)
    assert [options[l]["recall"] for l in lines] == sorted(
        (options[l]["recall"] for l in lines), reverse=True
    )
    assert options[.4]["false_alarms_per_1000"] > options[.6]["false_alarms_per_1000"]

    data = pd.read_csv("data/processed/aura_shipments.csv")
    default = triage_queue(data, model)
    wider = triage_queue(data, model, alert_line=.40)

    assert len(wider) > len(default)
    assert set(default.Shipment_ID) <= set(wider.Shipment_ID)
    assert wider.p_failure.min() >= .40
    assert default.p_failure.min() >= .60
