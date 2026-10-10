"""One entered order in, one full assessment out: risk, cause, urgency, reasons, actions."""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.build_dataset import PASSIVE_CLASSES_HOURS, REEFER_HOLD_HOURS
from src.explain import explain_prediction
from src.features import add_features
from src.model import load_model, predict_failure_risk
from src.preprocessing import ShipmentValidationError, _key
from src.data_loader import DEMO_PATH
from src.prioritize import URGENCY_WEIGHT, calculate_triage_queue
from src.recommendations import (
    ACTION_CATALOG,
    ALERT_LINE,
    LOSS_SHARE,
    _clean_shipment,
    action_applies,
    check_sla_status,
    get_recommended_action,
    simulate_intervention,
)
from src.schema import TYPICAL_PACE, WEATHER_SEVERITY

INSUFFICIENT = "Insufficient data for reliable prediction"
OUT_OF_SCOPE = "Outside Aura's scope"

# The product covers the air leg and the road leg to the hospital.
# The model still learns from every mode, because mode does not change
# how often a shipment is disrupted.
IN_SCOPE_MODES = ("Air", "Road")

# Form field names (left) and the names the rest of the backend uses (right).
ORDER_ALIASES = {
    "Shipment_ID": ["Order_ID"],
    "Origin_Port": ["Place_of_Origin", "Origin"],
    "Destination_Port": ["Place_of_Destination", "Destination"],
    "Transport_Mode": ["Mode"],
    "Product_Category": ["Product"],
    "Lead_Time_Days": ["Lead-Time in days", "Lead_Time"],
    "Weight_MT": ["Stock_Volume"],
    "Weather_Condition": ["Weather"],
    "Distance_km": ["Distance"],
    "Geopolitical_Risk_Score": ["Geopolitical_Risk"],
    "Carrier_Reliability_Score": ["Carrier_Reliability"],
    "Elapsed_Transit_Hours_Simulated": ["Transit_Duration", "Elapsed_Transit_Hours"],
    "Remaining_Buffer_Hours_Simulated": ["Buffer_of_Package", "Remaining_Buffer_Hours"],
    "Storage_Standard_Simulated": ["Storage_Standard"],
    "Cargo_Value_USD_Simulated": ["Cargo_Value_USD", "Cargo_Value"],
    "Hospital_Urgency_Tier_Simulated": ["Hospital_Urgency_Tier", "Urgency_Tier"],
}
ORDER_LOOKUP = {
    _key(name): column
    for column, aliases in ORDER_ALIASES.items()
    for name in [column, *aliases]
}

MODE_ALIASES = {
    "multimodalairtoland": "Air",
    "airtoland": "Air",
    "land": "Road",
    "truck": "Road",
    "ocean": "Sea",
}

# Latitude and longitude, used only to estimate distance when none is entered.
PLACES = {
    "Singapore": (1.29, 103.85),
    "Los Angeles": (33.74, -118.27),
    "Shanghai": (31.23, 121.47),
    "Rotterdam": (51.92, 4.48),
    "Busan": (35.10, 129.04),
    "Hamburg": (53.55, 9.99),
    "Dubai": (25.27, 55.30),
    "Antwerp": (51.22, 4.40),
    "Marseille": (43.30, 5.37),
    "Frankfurt": (50.03, 8.57),
    "Kuala Lumpur": (3.14, 101.69),
    "Penang": (5.30, 100.28),
    "Johor Bahru": (1.49, 103.74),
    "Bangkok": (13.69, 100.75),
    "Jakarta": (-6.13, 106.66),
    "Hong Kong": (22.31, 113.91),
    "Tokyo": (35.55, 139.78),
    "Mumbai": (19.09, 72.87),
    "London": (51.47, -0.45),
    "New York": (40.64, -73.78),
    "Sydney": (-33.94, 151.18),
}
# Real routes are longer than a straight line; these multipliers are assumptions.
ROUTE_FACTOR = {"Air": 1.05, "Road": 1.3, "Rail": 1.3, "Sea": 1.5}

# Country of each known place, for the route risk lookup.
PLACE_COUNTRY = {
    "Singapore": "Singapore", "Los Angeles": "United States",
    "New York": "United States", "Shanghai": "China",
    "Hong Kong": "Hong Kong SAR, China", "Rotterdam": "Netherlands",
    "Busan": "Korea, Rep.", "Hamburg": "Germany", "Frankfurt": "Germany",
    "Dubai": "United Arab Emirates", "Antwerp": "Belgium",
    "Marseille": "France", "Kuala Lumpur": "Malaysia", "Penang": "Malaysia",
    "Johor Bahru": "Malaysia", "Bangkok": "Thailand", "Jakarta": "Indonesia",
    "Tokyo": "Japan", "Mumbai": "India", "London": "United Kingdom",
    "Sydney": "Australia",
}

# World Bank Worldwide Governance Indicators, "Political Stability and Absence
# of Violence/Terrorism" estimate (about -2.5 unstable to +2.5 stable), 2025.
# Fetched 2026-10-10 from
# api.worldbank.org/v2/country/all/indicator/GOV_WGI_PV.EST?source=3
POLITICAL_STABILITY_YEAR = 2025
POLITICAL_STABILITY = {
    "Singapore": 1.334, "Japan": 1.259, "Korea, Rep.": 0.917,
    "United Arab Emirates": 0.826, "Australia": 0.781,
    "Hong Kong SAR, China": 0.775, "Netherlands": 0.535, "Malaysia": 0.340,
    "United Kingdom": 0.290, "Germany": 0.208, "Belgium": 0.173,
    "China": 0.022, "France": -0.092, "United States": -0.302,
    "Thailand": -0.560, "India": -0.608, "Indonesia": -0.739,
}


def stability_to_risk(estimate):
    """Turn the -2.5..+2.5 stability estimate into a 0 to 10 risk score (10 is worst)."""
    return round(min(max((2.5 - estimate) * 2, 0), 10), 1)


STORAGE_BY_PRODUCT = {
    "Pharmaceuticals": "2°C to 8°C",
    "Perishables": "2°C to 8°C",
}

_MODEL = None


def _default_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def _blank(value):
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and math.isnan(value))
        or (isinstance(value, str) and not value.strip())
    )


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain(value):
    """Plain Python values only, so the result can be shown or saved as JSON."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _place(text):
    if not isinstance(text, str):
        return None
    key = _key(text)
    matches = [name for name in PLACES if _key(name) in key]
    return max(matches, key=len) if matches else None


def estimate_distance_km(origin, destination, mode):
    """Straight-line distance between two known places, stretched for the mode."""
    start, end = _place(origin), _place(destination)
    if start is None or end is None:
        return None

    lat1, lon1, lat2, lon2 = map(math.radians, [*PLACES[start], *PLACES[end]])
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    straight = 2 * 6371 * math.asin(math.sqrt(a))
    return round(min(max(straight * ROUTE_FACTOR.get(mode, 1.2), 50), 25000), 1)


def estimate_route_risk(origin, destination, risk_provider=None):
    """Return (score, source). risk_provider(origin, destination) can supply a live score."""
    if risk_provider is not None:
        try:
            score = _number(risk_provider(origin, destination))
        except Exception:
            score = None
        if score is not None and 0 <= score <= 10:
            return score, "Live route risk source."

    countries = [
        PLACE_COUNTRY[name]
        for name in (_place(origin), _place(destination))
        if name in PLACE_COUNTRY
    ]
    if not countries:
        return None, None

    # The route is as risky as its less stable end.
    worst = min(countries, key=POLITICAL_STABILITY.get)
    return stability_to_risk(POLITICAL_STABILITY[worst]), (
        f"World Bank political stability index {POLITICAL_STABILITY_YEAR} "
        f"for {worst}, the less stable end of the route, "
        "converted to a 0 to 10 risk score. A yearly figure, not a live feed."
    )


def prepare_order(order, risk_provider=None):
    """Rename the form fields and fill in what can be inferred. Says what it inferred."""
    if not isinstance(order, Mapping):
        raise ShipmentValidationError("Supply one order as a dictionary.")

    shipment, ignored, inferred = {}, [], {}

    for name, value in order.items():
        column = ORDER_LOOKUP.get(_key(name))
        if column is None:
            ignored.append(str(name))
        elif not _blank(value):
            if column in shipment:
                raise ShipmentValidationError(
                    f"Two entered fields both mean {column}."
                )
            shipment[column] = value.strip() if isinstance(value, str) else value

    if "Product_Category" not in shipment:
        shipment["Product_Category"] = "Pharmaceuticals"
        inferred["Product_Category"] = "Assumed pharmaceuticals."

    mode = shipment.get("Transport_Mode")
    if isinstance(mode, str) and _key(mode) in MODE_ALIASES:
        shipment["Transport_Mode"] = MODE_ALIASES[_key(mode)]
        inferred["Transport_Mode"] = (
            f"'{mode}' is scored as {shipment['Transport_Mode']}, "
            "the main leg of the trip."
        )

    origin = shipment.get("Origin_Port")
    destination = shipment.get("Destination_Port")

    if "Distance_km" not in shipment:
        distance = estimate_distance_km(
            origin, destination, shipment.get("Transport_Mode")
        )
        if distance is not None:
            shipment["Distance_km"] = distance
            inferred["Distance_km"] = (
                "Estimated from the two places (straight line, "
                "stretched for the transport mode)."
            )

    if "Geopolitical_Risk_Score" not in shipment:
        score, source = estimate_route_risk(origin, destination, risk_provider)
        if score is not None:
            shipment["Geopolitical_Risk_Score"] = score
            inferred["Geopolitical_Risk_Score"] = source
        else:
            inferred["Geopolitical_Risk_Score"] = (
                "Not known for this route. A typical value is assumed."
            )

    if "Carrier_Reliability_Score" not in shipment:
        inferred["Carrier_Reliability_Score"] = (
            "Not entered. A typical carrier is assumed."
        )

    if "Weather_Condition" not in shipment:
        inferred["Weather_Condition"] = (
            "Not entered. The backup model reads the pace of the trip instead; "
            "it is a little less accurate."
        )

    if "Storage_Standard_Simulated" not in shipment:
        shipment["Storage_Standard_Simulated"] = STORAGE_BY_PRODUCT.get(
            shipment["Product_Category"], "Ambient"
        )
        inferred["Storage_Standard_Simulated"] = (
            f"Usual storage for {shipment['Product_Category']}."
        )

    storage = shipment["Storage_Standard_Simulated"]
    lead_hours = _number(shipment.get("Lead_Time_Days"))
    lead_hours = None if lead_hours is None else lead_hours * 24

    if (
        "Remaining_Buffer_Hours_Simulated" not in shipment
        and storage in REEFER_HOLD_HOURS
    ):
        if shipment.get("Transport_Mode") == "Air" and lead_hours is not None:
            index = min(
                int(np.searchsorted(PASSIVE_CLASSES_HOURS, lead_hours * 1.2)),
                len(PASSIVE_CLASSES_HOURS) - 1,
            )
            elapsed = _number(shipment.get("Elapsed_Transit_Hours_Simulated")) or 0
            shipment["Remaining_Buffer_Hours_Simulated"] = round(
                max(float(PASSIVE_CLASSES_HOURS[index]) - elapsed, 0), 1
            )
            inferred["Remaining_Buffer_Hours_Simulated"] = (
                f"Assumed a {int(PASSIVE_CLASSES_HOURS[index])}-hour insulated "
                "shipper, minus the hours already travelled."
            )
        elif shipment.get("Transport_Mode") in {"Sea", "Road", "Rail"}:
            shipment["Remaining_Buffer_Hours_Simulated"] = float(
                np.mean(REEFER_HOLD_HOURS[storage])
            )
            inferred["Remaining_Buffer_Hours_Simulated"] = (
                "Assumed a powered refrigerated container; this is how long "
                "it holds temperature if the power fails."
            )

    if (
        "Hospital_Urgency_Tier_Simulated" not in shipment
        and shipment["Product_Category"] == "Pharmaceuticals"
    ):
        shipment["Hospital_Urgency_Tier_Simulated"] = "Routine Restock"
        inferred["Hospital_Urgency_Tier_Simulated"] = "Assumed Routine Restock."

    return shipment, inferred, ignored


def thermal_runway_curve(buffer_hours, hours_to_go, points=48):
    """Hours of thermal buffer left at each hour until arrival, and when it hits zero."""
    buffer_hours, hours_to_go = _number(buffer_hours), _number(hours_to_go)
    if buffer_hours is None or hours_to_go is None:
        return None

    hours_to_go = max(hours_to_go, 0)
    span = max(hours_to_go, min(buffer_hours, hours_to_go * 1.5), 1.0)
    step = span / points

    return {
        "points": [
            {
                "hour": round(i * step, 2),
                "buffer_hours_left": round(max(buffer_hours - i * step, 0), 2),
            }
            for i in range(points + 1)
        ],
        "arrival_at_hour": round(hours_to_go, 1),
        "buffer_ends_at_hour": (
            round(buffer_hours, 1) if buffer_hours < hours_to_go else None
        ),
        "note": (
            "Shows buffer hours, not temperature. "
            "The data has no temperature readings."
        ),
    }


def _failure_mode(thermal, featured, weather):
    """What kind of trouble is most likely. Rules on the entered facts, not a second model."""
    spoilage = thermal["runway_hours"] is not None and thermal["runway_hours"] < 0
    pace = _number(featured["pace_vs_typical"])
    severity = WEATHER_SEVERITY.get(weather)
    delay = (
        thermal["overdue_hours"] > 0
        or (severity is not None and severity >= 3)
        or (pace is not None and pace >= 1.5)
    )

    if spoilage and delay:
        return dict(
            mode="Delay leading to cold-chain excursion",
            reason=(
                "The trip is running slow and the thermal buffer "
                "ends before arrival."
            ),
        )
    if spoilage:
        return dict(
            mode="Cold-chain excursion",
            reason="The thermal buffer ends before arrival.",
        )
    if delay:
        return dict(
            mode="Transport delay",
            reason=(
                "The trip is running slow or severe weather is on the route."
            ),
        )
    return dict(
        mode="No dominant cause",
        reason="No delay sign and the thermal buffer covers the trip.",
    )


def _top_reasons(thermal, featured, clean, explanation):
    reasons = []

    if thermal["runway_hours"] is not None and thermal["runway_hours"] < 0:
        reasons.append(dict(
            kind="thermal runway",
            text=(
                f"Thermal buffer runs out {-thermal['runway_hours']:.1f} hours "
                f"before arrival ({thermal['buffer_hours']:.1f} hours of buffer "
                f"left, {thermal['hours_to_go']:.1f} hours still to go)."
            ),
        ))

    if thermal["overdue_hours"] > 0:
        reasons.append(dict(
            kind="transit time",
            text=(
                f"Already {thermal['overdue_hours']:.1f} hours past the planned "
                f"delivery time ({thermal['elapsed_hours']:.1f} hours elapsed "
                f"against {thermal['planned_hours']:.1f} planned)."
            ),
        ))

    pace = _number(featured["pace_vs_typical"])
    if _blank(clean.get("Weather_Condition")) and pace is not None and pace >= 1.5:
        reasons.append(dict(
            kind="risk driver",
            text=(
                f"The trip is planned at {pace:.1f} times the normal time "
                f"for {clean['Transport_Mode'].lower()} over this distance."
            ),
        ))

    for driver in explanation["drivers"]:
        if driver["contribution"] >= .01:
            reasons.append(dict(kind="risk driver", text=driver["description"]))

    return reasons[:3]


def _actions(clean, model, category, probability, cargo, countdown, alert):
    outcomes = []

    for action_id, action in ACTION_CATALOG.items():
        if not action_applies(action_id, clean):
            # Kept in the list so the screen can show it as unavailable.
            outcomes.append(dict(
                action_id=action_id,
                title=action["title"],
                available=False,
                unavailable_reason="Only available for air shipments.",
                risk_before=probability,
                risk_after=probability,
                action_cost_usd=action["cost_usd"],
                net_saving_usd=None,
                thermal_runway_hours_after=None,
                kind=action["kind"],
                effect=action["effect"],
            ))
            continue

        try:
            result = simulate_intervention(clean, action_id, model)
        except ValueError:
            continue

        context = result["simulated_context"]
        to_go, gap = context["hours_to_go"], context["cold_gap_hours"]
        runway = (
            None if gap is None or not countdown
            else round(gap + min(to_go or 0, 0), 1)
        )

        outcomes.append(dict(
            action_id=action_id,
            title=result["title"],
            available=True,
            unavailable_reason=None,
            risk_before=result["before_probability"],
            risk_after=result["after_probability"],
            action_cost_usd=result["action_cost_usd"],
            net_saving_usd=result["estimated_net_benefit_usd"],
            thermal_runway_hours_after=runway,
            kind=ACTION_CATALOG[action_id]["kind"],
            effect=ACTION_CATALOG[action_id]["effect"],
        ))

    baseline = next(
        (item for item in outcomes if item["action_id"] == "monitor"), None
    )
    exposed = (
        baseline is not None
        and baseline["thermal_runway_hours_after"] is not None
        and baseline["thermal_runway_hours_after"] < 0
    )

    for item in outcomes:
        covers = (
            exposed
            and item["thermal_runway_hours_after"] is not None
            and item["thermal_runway_hours_after"] >= 0
        )
        item["covers_thermal_gap"] = covers
        # Cargo that stops being exposed to a buffer run-out. Not a predicted saving.
        item["cargo_protected_usd"] = cargo if covers and cargo is not None else 0.0

    if cargo is not None:
        chosen = get_recommended_action(
            category, probability, shipment_data=clean, model=model
        )
    else:
        chosen = get_recommended_action(category, probability)

    covering = [item for item in outcomes if item["covers_thermal_gap"]]
    thermal_action = (
        min(covering, key=lambda item: item["action_cost_usd"])["action_id"]
        if covering
        else None
    )

    # When no action brings an alert under the line, cover the hospital instead.
    lowest = min(
        (item["risk_after"] for item in outcomes), default=probability
    )
    backup = (
        "stock_diversion"
        if alert and lowest >= ALERT_LINE and "stock_diversion" in ACTION_CATALOG
        else None
    )

    return dict(
        options=outcomes,
        recommended_action_id=chosen["action_id"],
        recommended_title=chosen["title"],
        thermal_action_id=thermal_action,
        backup_action_id=backup,
        note=(
            "The recommended action is the one with the highest net saving "
            "on disruption risk. The thermal action is the cheapest one that "
            "makes the buffer cover the trip. The backup action covers the "
            "hospital when no action brings the risk under the alert line."
        ),
    )


def assess_order(order, model=None, risk_provider=None, custody_records=None):
    """Stages 1 to 5 for one entered order. Bad input returns a status, never a crash."""
    model = _default_model() if model is None else model
    inferred, ignored = {}, []
    if isinstance(order, pd.Series):
        order = order.to_dict()

    try:
        shipment, inferred, ignored = prepare_order(order, risk_provider)
        clean = _clean_shipment(shipment, model)

        if clean.get("Transport_Mode") in {"Sea", "Rail"}:
            return _plain(dict(
                status=OUT_OF_SCOPE,
                reason=(
                    "Aura covers air and road shipments. "
                    f"This one travels by {clean['Transport_Mode'].lower()}."
                ),
                inferred=inferred,
                ignored_fields=ignored,
            ))

        frame = pd.DataFrame([clean])
        risk = predict_failure_risk(frame, model).iloc[0]
        featured = add_features(frame, config=model.metadata).iloc[0]
        explanation = explain_prediction(clean, model)
    except (ValueError, TypeError) as exc:
        return _plain(dict(
            status=INSUFFICIENT,
            reason=str(exc),
            inferred=inferred,
            ignored_fields=ignored,
        ))

    probability = float(risk["p_failure"])
    planned = clean["Lead_Time_Days"] * 24
    elapsed = _number(clean.get("Elapsed_Transit_Hours_Simulated"))
    buffer = _number(clean.get("Remaining_Buffer_Hours_Simulated"))
    cargo = _number(clean.get("Cargo_Value_USD_Simulated"))
    planned = _number(planned)

    hours_to_go = (
        None if planned is None or elapsed is None
        else round(max(planned - elapsed, 0), 1)
    )
    # In a powered refrigerated container the assumed buffer is how long the
    # cargo holds if the power fails. It only counts down in passive packaging
    # (air freight) or when the user entered the buffer themselves.
    assumed_buffer = (
        "Remaining_Buffer_Hours_Simulated" in inferred
        or (isinstance(order, Mapping) and "Context_Source" in order)
    )
    countdown = not (
        clean.get("Transport_Mode") == "Road" and assumed_buffer
    )
    thermal = dict(
        storage_standard=clean.get("Storage_Standard_Simulated"),
        buffer_hours=buffer,
        buffer_counts_down=countdown,
        planned_hours=planned,
        elapsed_hours=elapsed,
        hours_to_go=hours_to_go,
        overdue_hours=(
            0.0 if planned is None or elapsed is None
            else round(max(elapsed - planned, 0), 1)
        ),
        runway_hours=(
            None if buffer is None or hours_to_go is None or not countdown
            else round(buffer - hours_to_go, 1)
        ),
        note=(
            None if countdown
            else "Powered refrigerated container. The buffer is how long "
            "the cargo holds temperature if the power fails."
        ),
    )
    thermal["curve"] = (
        thermal_runway_curve(buffer, hours_to_go) if countdown else None
    )

    weight = URGENCY_WEIGHT.get(clean.get("Hospital_Urgency_Tier_Simulated"))
    urgency = (
        None if cargo is None or weight is None or buffer is None
        else probability * cargo * weight / max(buffer, .5)
    )

    return _plain(dict(
        status="ok",
        shipment=clean,
        inferred=inferred,
        ignored_fields=ignored,
        risk=dict(
            p_failure=probability,
            risk_level=risk["risk_level"],
            risk_tier=risk["risk_tier"],
            is_alert=risk["is_alert"],
            model_variant=risk["model_variant"],
        ),
        failure_mode=_failure_mode(
            thermal, featured, clean.get("Weather_Condition")
        ),
        priority=dict(
            urgency_score=urgency,
            urgency_weight=weight,
            expected_loss_usd=(
                None if cargo is None else probability * LOSS_SHARE * cargo
            ),
            formula=(
                "risk x cargo value x urgency weight "
                "/ max(buffer hours, 0.5)"
            ),
        ),
        explanation=dict(
            summary=explanation["summary"],
            category=explanation["category"],
            top_reasons=_top_reasons(thermal, featured, clean, explanation),
            drivers=explanation["drivers"],
            warning_signs=explanation["domain_indicators"],
        ),
        thermal=thermal,
        actions=_actions(
            clean, model, explanation["category"], probability, cargo,
            countdown, bool(risk["is_alert"]),
        ),
        # A powered container's assumed buffer is not a countdown, so it cannot breach.
        sla=check_sla_status(
            probability,
            buffer if countdown else None,
            shipment_id=clean.get("Shipment_ID"),
            custody_records=custody_records,
        ),
        scorecard=dict(
            precision=model.metrics["precision"],
            recall=model.metrics["recall"],
            f1=model.metrics["f1"],
            roc_auc=model.metrics["roc_auc"],
            tested_on_shipments=model.metadata["holdout_rows"],
            note="Measured on held-out simulated shipments at the 60% alert line.",
        ),
    ))


def triage_queue(shipments=None, model=None, top=None):
    """Score shipments and return the in-scope pharmaceutical alerts, most urgent first."""
    model = _default_model() if model is None else model
    if shipments is None:
        shipments = pd.read_csv(DEMO_PATH)

    in_scope = shipments[
        shipments["Transport_Mode"].isin(IN_SCOPE_MODES)
    ]
    scored = in_scope.join(predict_failure_risk(in_scope, model))
    queue = calculate_triage_queue(scored)
    return queue if top is None else queue.head(top)


def rank_orders(orders, model=None, risk_provider=None, custody_records=None):
    """Assess several entered orders and put the most urgent first."""
    assessed = [
        assess_order(order, model, risk_provider, custody_records)
        for order in orders
    ]

    def position(item):
        score = (item.get("priority") or {}).get("urgency_score")
        return (score is None, -(score or 0))

    return sorted(assessed, key=position)


__all__ = [
    "INSUFFICIENT",
    "OUT_OF_SCOPE",
    "IN_SCOPE_MODES",
    "ORDER_ALIASES",
    "assess_order",
    "triage_queue",
    "rank_orders",
    "prepare_order",
    "estimate_distance_km",
    "estimate_route_risk",
    "thermal_runway_curve",
]
