"""LogiSmart web server: a JSON API over the Python backend, plus the built React app.

Run with:  python server.py     (then open http://localhost:8000)

The React app lives in web/. Build it once with `npm run build` inside web/.
State (custody handshakes, logged actions) is kept in memory for the demo.
"""
import math
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src import intake
from src.data_loader import DEMO_PATH
from src.model import load_model, predict_failure_risk
from src.protocols import record_custody_handshake
from src.recommendations import ACTION_CATALOG, LOSS_SHARE, _clean_shipment, check_sla_status

WEB = Path(__file__).with_name("web") / "dist"
WEATHER = ["Clear", "Rain", "Fog", "Storm", "Hurricane"]
TIERS = ["Routine Restock", "Elevated Procedure", "Critical Emergency"]
ALERT_LINES = [.30, .40, .50, .60, .70, .80]
CORRECTIVE = {"cold_storage_transfer", "refrigerant_refresh", "refrigerated_ground"}
# Best action across the 508 test alerts at the 60% line, from notebook 05.
ACTION_MIX = [
    dict(name="Keep monitoring", count=271),
    dict(name="Move to next direct flight", count=128),
    dict(name="Refrigerated ground transport", count=109),
]

MODEL = load_model()
SHIPMENTS = pd.read_csv(DEMO_PATH)
CUSTODY = []     # digital handshake records
COMMITTED = {}   # shipment id -> actions logged


def clean(value):
    """Plain JSON values: NaN and pandas blanks become null."""
    value = intake._plain(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reply(data, status=200):
    return JSONResponse(clean(data), status_code=status)


def alert_line(request):
    try:
        line = float(request.query_params.get("line", .6))
    except ValueError:
        line = .6
    if not math.isfinite(line):
        line = .6
    return min(ALERT_LINES, key=lambda option: abs(option - line))


@lru_cache(maxsize=8)
def queue(line):
    q = intake.triage_queue(SHIPMENTS, MODEL, alert_line=line).copy()
    # A road shipment that would have to cross an ocean fails the data check and stays out.
    if "Flag_Land_Route_Across_Ocean" in q:
        q = q[~q["Flag_Land_Route_Across_Ocean"].fillna(0).astype(bool)]
    to_go = (q["Lead_Time_Days"] * 24 - q["Elapsed_Transit_Hours_Simulated"]).clip(lower=0)
    q["hours_to_go"] = to_go
    # Only passive packaging (air freight) counts down; a powered reefer does not.
    q["runway"] = (q["Remaining_Buffer_Hours_Simulated"] - to_go).where(
        q["Transport_Mode"] == "Air"
    )
    return q.reset_index(drop=True)


@lru_cache(maxsize=256)
def assess_shipment(shipment_id):
    match = SHIPMENTS[SHIPMENTS["Shipment_ID"] == shipment_id]
    if match.empty:
        return dict(status=intake.INSUFFICIENT, reason="Shipment not found.")
    return intake.assess_order(match.iloc[0].to_dict(), model=MODEL)


@lru_cache(maxsize=512)
def assess_scenario(items):
    return intake.assess_order(dict(items), model=MODEL)


def sla_for(shipment_id, probability, buffer_hours):
    try:
        return check_sla_status(
            probability, buffer_hours, shipment_id=shipment_id, custody_records=CUSTODY
        )
    except ValueError:
        return None


def money(value):
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def build_alerts(top):
    alerts = []
    for row in top.to_dict("records"):
        sid, p = row["Shipment_ID"], row["p_failure"]
        runway = None if pd.isna(row["runway"]) else row["runway"]
        sla = sla_for(sid, p, countdown_buffer(row))
        tier = row["Hospital_Urgency_Tier_Simulated"]
        body = (
            f"{p:.0%} disruption risk on {money(row['Cargo_Value_USD_Simulated'])} of "
            f"pharmaceuticals for {'an' if tier[:1] in 'AEIOU' else 'a'} {tier} order."
        )
        if sla and sla["breach_warning"]:
            kind, tone, head = "SLA breach", "red", "Carrier payment on hold"
            body = sla["breach_notice"]["text"]
        elif sla and sla["trigger_condition"]:
            kind, tone, head = "Resolved", "teal", "Fix logged. Carrier payment released."
            body = "A cooling fix that restores at least 3 hours of buffer has been logged."
        elif runway is not None and runway < 0:
            kind, tone = "Buffer short", "red"
            head = f"Thermal buffer ends {-runway:.1f} h before arrival"
        elif row["Weather_Condition"] in ("Storm", "Hurricane"):
            kind, tone, head = "Weather", "amber", f"{row['Weather_Condition']} on the route"
        else:
            kind, tone, head = "Risk", "amber", f"{row['risk_level']} disruption risk"
        alerts.append(dict(
            id=sid, kind=kind, tone=tone, head=head, body=body,
            route=f"{row['Origin_Port']} to {row['Destination_Port']} by {row['Transport_Mode'].lower()}",
            hours_to_go=row["hours_to_go"], logged=bool(COMMITTED.get(sid)),
        ))
    return alerts


def line_row(line):
    return next(o for o in intake.alert_line_options() if abs(o["alert_line"] - line) < .001)


def weather_sweep(shipment):
    """The same shipment under each of the five weather conditions."""
    frame = pd.DataFrame([dict(shipment, Weather_Condition=w) for w in WEATHER])
    try:
        risks = predict_failure_risk(frame, MODEL)["p_failure"].tolist()
    except (ValueError, TypeError):
        return None
    return [dict(weather=w, risk=p) for w, p in zip(WEATHER, risks)]


def with_after(result):
    """Add the buffer and hours to go after each action, for the runway chart."""
    thermal = result["thermal"]
    for option in result["actions"]["options"]:
        if not option["available"] or thermal["buffer_hours"] is None:
            option["after"] = None
            continue
        added = ACTION_CATALOG[option["action_id"]]["changes"].get("buffer_hours", 0)
        buffer_after = thermal["buffer_hours"] + added
        runway_after = option["thermal_runway_hours_after"]
        option["after"] = dict(
            buffer_hours=buffer_after,
            hours_to_go=(
                thermal["hours_to_go"] if runway_after is None
                else max(buffer_after - runway_after, 0)
            ),
        )
    return result


def decide(result, line):
    """Rank the actions for the screen: what pays first, then what protects the batch.

    The alert line comes from the slider, so the advice follows it.
    """
    actions, p = result["actions"], result["risk"]["p_failure"]
    options = actions["options"]
    by_id = {o["action_id"]: o for o in options}
    monitor = by_id.get("monitor")
    usable = [
        o for o in options
        if o["available"] and o["action_id"] not in ("monitor", "stock_diversion")
    ]
    flagged = p >= line
    paying = sorted(
        (o for o in usable if (o["net_saving_usd"] or 0) > 0),
        key=lambda o: -o["net_saving_usd"],
    )
    covering = sorted(
        (o for o in usable if o["covers_thermal_gap"]),
        key=lambda o: o["action_cost_usd"],
    )
    best = paying[0] if flagged and paying else None
    # The best action may already make the buffer cover the trip.
    cover = None if (best and best["covers_thermal_gap"]) or not covering else covering[0]

    ranked = []

    def add(option):
        if option is not None and option not in ranked:
            ranked.append(option)

    add(best)
    add(cover)
    if not ranked:
        add(monitor)
    for option in (paying if flagged else []):
        add(option)
    for option in covering:
        add(option)
    add(monitor)
    # Whatever is left costs money and wins nothing, so it sits below doing nothing.
    for option in sorted(usable, key=lambda o: -(o["net_saving_usd"] or 0)):
        add(option)

    for option in options:
        if not option["available"]:
            payoff = "unavailable"
        elif option["action_id"] == "stock_diversion":
            payoff = "backup"
        elif option["action_id"] == "monitor":
            payoff = "none"
        elif (option["net_saving_usd"] or 0) > 0:
            payoff = "saves"
        elif option["covers_thermal_gap"]:
            payoff = "protects"
        else:
            payoff = "costs"
        option["payoff"] = payoff
        option["rank"] = ranked.index(option) + 1 if option in ranked else None

    lowest = min((o["risk_after"] for o in options if o["available"]), default=p)
    # With nothing to save on risk, protecting the batch is still the thing to do.
    chosen = best or cover or monitor
    actions.update(
        options=ranked + [o for o in options if o not in ranked],
        flagged=flagged,
        recommended_action_id=chosen["action_id"] if chosen else None,
        recommended_title=chosen["title"] if chosen else None,
        thermal_action_id=cover["action_id"] if cover else None,
        # When no action brings an alert under the line, cover the hospital instead.
        backup_action_id=(
            "stock_diversion" if flagged and lowest >= line and "stock_diversion" in by_id else None
        ),
    )
    return result


def countdown_buffer(row):
    """Buffer hours for a queue row, or None in a powered container (no countdown)."""
    return row["Remaining_Buffer_Hours_Simulated"] if row["Transport_Mode"] == "Air" else None


def neighbours(q, position, span=2):
    """The shipments ranked just above and below, with the four things urgency weighs."""
    start = max(min(position - span, len(q) - (2 * span + 1)), 0)
    rows = q.iloc[start:start + 2 * span + 1].to_dict("records")
    return [dict(
        rank=start + i + 1, id=r["Shipment_ID"], p_failure=r["p_failure"],
        cargo=r["Cargo_Value_USD_Simulated"], tier=r["Hospital_Urgency_Tier_Simulated"],
        buffer_hours=r["Remaining_Buffer_Hours_Simulated"], counts_down=r["Transport_Mode"] == "Air",
    ) for i, r in enumerate(rows)]


def safe(handler):
    """A backend error becomes the standard 'insufficient data' answer, never a crash page."""
    def wrapped(request):
        try:
            return handler(request)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            return reply(dict(status=intake.INSUFFICIENT, reason=str(exc)))
    return wrapped


# ---------------------------------------------------------------- endpoints

def get_meta(request):
    return reply(dict(
        places=sorted(intake.PLACES), coordinates=intake.PLACES, weather=WEATHER,
        tiers=TIERS, alert_lines=ALERT_LINES, loss_share=LOSS_SHARE,
        corrective=sorted(CORRECTIVE), lines=intake.alert_line_options(),
    ))


def get_queue(request):
    line = alert_line(request)
    q = queue(line)
    top = q.head(10)
    columns = [
        "Shipment_ID", "Origin_Port", "Destination_Port", "Transport_Mode",
        "Weather_Condition", "p_failure", "risk_level", "Cargo_Value_USD_Simulated",
        "Hospital_Urgency_Tier_Simulated", "Remaining_Buffer_Hours_Simulated",
        "hours_to_go", "runway", "urgency_score", "expected_loss_usd",
    ]
    alerts = build_alerts(top)
    return reply(dict(
        alert_line=line, count=len(q), top=top[columns].to_dict("records"),
        alerts=alerts, line=line_row(line), loss_share=LOSS_SHARE,
        stats=dict(
            expected_loss_top=float(top["expected_loss_usd"].sum()) if len(top) else 0,
            buffer_short=int((top["runway"] < 0).sum()), shown=len(top),
            sla_breaches=sum(a["kind"] == "SLA breach" for a in alerts),
        ),
    ))


def shipment_payload(shipment_id, line):
    result = assess_shipment(shipment_id)
    if result.get("status") != "ok":
        return dict(result)
    result = decide(with_after(clean(result)), line)  # clean() copies, so the cache stays untouched
    q = queue(line)
    ids = list(q["Shipment_ID"])
    thermal, risk = result["thermal"], result["risk"]
    row = SHIPMENTS.loc[SHIPMENTS["Shipment_ID"] == shipment_id]
    crosses_ocean = (
        "Flag_Land_Route_Across_Ocean" in row
        and bool(row["Flag_Land_Route_Across_Ocean"].fillna(0).astype(bool).any())
    )
    rank = ids.index(shipment_id) + 1 if shipment_id in ids else None
    if rank:
        rank_note = None
    elif risk["p_failure"] < line:
        rank_note = "Below the alert line, so it is not in the queue."
    elif crosses_ocean:
        rank_note = "Left out of the queue: the data check found a land route that crosses an ocean."
    elif not (row["Product_Category"] == "Pharmaceuticals").any():
        rank_note = "Not a pharmaceutical shipment, so it is not in the queue."
    else:
        rank_note = "Not in the queue."
    result.update(
        sla=sla_for(
            shipment_id, risk["p_failure"],
            thermal["buffer_hours"] if thermal["buffer_counts_down"] else None,
        ),
        rank=rank, rank_note=rank_note,
        queue_size=len(q),
        flag_land_route=crosses_ocean,
        committed=COMMITTED.get(shipment_id, []),
        neighbours=neighbours(q, rank - 1) if rank else [],
        line=line_row(line), loss_share=LOSS_SHARE,
        sweep=weather_sweep(result["shipment"]),
    )
    return result


def get_shipment(request):
    return reply(shipment_payload(request.path_params["shipment_id"], alert_line(request)))


def scenario_order(body):
    return tuple(sorted((str(k), v) for k, v in body.items() if isinstance(v, (str, int, float))))


async def post_assess(request):
    """Full what-if for an entered order: risk, cause, runway, actions and money."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return reply(dict(status=intake.INSUFFICIENT, reason="Send one order as JSON."))
    result = assess_scenario(scenario_order(body))
    if result.get("status") != "ok":
        return reply(result)
    result = decide(with_after(clean(result)), alert_line(request))
    result["sweep"] = weather_sweep(result["shipment"])
    return reply(result)


async def post_quick(request):
    """Risk only, in a few milliseconds, so the gauge follows a slider as it moves."""
    try:
        body = await request.json()
        shipment, _, _ = intake.prepare_order(dict(scenario_order(body)))
        if shipment.get("Transport_Mode") in ("Sea", "Rail"):
            return reply(dict(status=intake.OUT_OF_SCOPE, reason="LogiSmart covers air and road shipments."))
        frame = pd.DataFrame([_clean_shipment(shipment, MODEL)])
        risk = predict_failure_risk(frame, MODEL).iloc[0]
        return reply(dict(status="ok", p_failure=float(risk["p_failure"]),
                          risk_level=risk["risk_level"]))
    except (ValueError, TypeError, AttributeError) as exc:
        return reply(dict(status=intake.INSUFFICIENT, reason=str(exc)))


def get_weather(request):
    origin = request.query_params.get("origin", "")
    destination = request.query_params.get("destination", "")
    try:
        condition, source = intake.lookup_weather(origin, destination)
    except Exception:
        condition, source = None, None
    if condition not in WEATHER:
        return reply(dict(condition=None, source=None))
    return reply(dict(condition=condition, source=source))


def get_evidence(request):
    line = alert_line(request)
    q = queue(line)
    return reply(dict(
        alert_line=line, total=len(SHIPMENTS),
        pharma=int((SHIPMENTS["Product_Category"] == "Pharmaceuticals").sum()),
        queue_size=len(q), top_id=q.iloc[0]["Shipment_ID"] if len(q) else None,
        line=line_row(line), roc_auc=MODEL.metrics["roc_auc"],
        action_mix=ACTION_MIX, actions=len(ACTION_CATALOG),
    ))


async def post_commit(request):
    """Log an action. Corrective ones record a custody handshake, which can release the escrow."""
    try:
        body = await request.json()
        shipment_id, action_id = str(body["shipment_id"]), str(body["action_id"])
        title = ACTION_CATALOG[action_id]["title"]
        if not (SHIPMENTS["Shipment_ID"] == shipment_id).any():
            raise KeyError(shipment_id)
    except (ValueError, KeyError, TypeError):
        return reply(dict(ok=False, message="Unknown shipment or action."), 400)

    if action_id in CORRECTIVE:
        try:
            record_custody_handshake(
                shipment_id, "Hub cold-chain crew", "Carrier duty officer",
                body.get("buffer_after", 0), f"{body.get('hub', 'Origin')} hub", True,
                records=CUSTODY, corrective_action=action_id,
            )
        except ValueError as exc:
            return reply(dict(ok=False, message=f"Handshake not logged: {exc}"), 400)
        message = f"Handshake logged for '{title}'."
    else:
        message = f"'{title}' logged."
    COMMITTED.setdefault(shipment_id, []).append(action_id)
    return reply(dict(ok=True, message=message, committed=COMMITTED[shipment_id]))


def spa(request):
    index = WEB / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        dict(message="The web app is not built yet. Run `npm install` then `npm run build` in web/."),
        status_code=503,
    )


routes = [
    Route("/api/meta", get_meta),
    Route("/api/queue", safe(get_queue)),
    Route("/api/shipment/{shipment_id}", safe(get_shipment)),
    Route("/api/assess", post_assess, methods=["POST"]),
    Route("/api/quick", post_quick, methods=["POST"]),
    Route("/api/weather", get_weather),
    Route("/api/evidence", safe(get_evidence)),
    Route("/api/commit", post_commit, methods=["POST"]),
]
if (WEB / "assets").exists():
    routes.append(Mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets"))
routes.append(Route("/{path:path}", spa))

app = Starlette(routes=routes)

def warm():
    """Score the top of the queue ahead of the first click, so a shipment opens at once."""
    for shipment_id in queue(.6).head(10)["Shipment_ID"]:
        assess_shipment(shipment_id)


if __name__ == "__main__":
    threading.Thread(target=warm, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
