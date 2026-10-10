"""Aura web server: a JSON API over the Python backend, plus the built React app.

Run with:  python server.py     (then open http://localhost:8000)

The React app lives in web/. Build it once with `npm run build` inside web/.
State (custody handshakes, logged actions) is kept in memory for the demo.
"""
import math
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
    return min(ALERT_LINES, key=lambda option: abs(option - line))


@lru_cache(maxsize=8)
def queue(line):
    q = intake.triage_queue(SHIPMENTS, MODEL, alert_line=line).copy()
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
        buffer = row["Remaining_Buffer_Hours_Simulated"] if row["Transport_Mode"] == "Air" else None
        sla = sla_for(sid, p, buffer)
        body = (
            f"{p:.0%} disruption risk on {money(row['Cargo_Value_USD_Simulated'])} of "
            f"pharmaceuticals for a {row['Hospital_Urgency_Tier_Simulated']} order."
        )
        if sla and sla["breach_warning"]:
            kind, tone, head = "SLA breach", "red", "SLA breach notice issued"
            body = sla["breach_notice"]["text"]
        elif sla and sla["trigger_condition"]:
            kind, tone, head = "Resolved", "teal", "Corrective proof verified. Escrow released."
            body = "A corrective handshake with at least 3 hours of buffer is on record."
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
    alerts = build_alerts(q.head(12))
    return reply(dict(
        alert_line=line, count=len(q), top=top[columns].to_dict("records"),
        alerts=alerts, line=line_row(line),
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
    result = with_after(clean(result))  # clean() copies, so the cache stays untouched
    q = queue(line)
    ids = list(q["Shipment_ID"])
    thermal, risk = result["thermal"], result["risk"]
    raw = SHIPMENTS.loc[SHIPMENTS["Shipment_ID"] == shipment_id, "Flag_Land_Route_Across_Ocean"]
    top = q.head(5)
    peak = float(top["urgency_score"].max() or 1) if len(top) else 1
    result.update(
        sla=sla_for(
            shipment_id, risk["p_failure"],
            thermal["buffer_hours"] if thermal["buffer_counts_down"] else None,
        ),
        rank=ids.index(shipment_id) + 1 if shipment_id in ids else None,
        queue_size=len(q),
        flag_land_route=bool(raw.fillna(False).any()),
        committed=COMMITTED.get(shipment_id, []),
        top5=[dict(id=r["Shipment_ID"], share=r["urgency_score"] / peak)
              for r in top.to_dict("records")],
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
    result = with_after(clean(result))
    result["sweep"] = weather_sweep(result["shipment"])
    return reply(result)


async def post_quick(request):
    """Risk only, in a few milliseconds, so the gauge follows a slider as it moves."""
    try:
        body = await request.json()
        shipment, _, _ = intake.prepare_order(dict(scenario_order(body)))
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
    Route("/api/queue", get_queue),
    Route("/api/shipment/{shipment_id}", get_shipment),
    Route("/api/assess", post_assess, methods=["POST"]),
    Route("/api/quick", post_quick, methods=["POST"]),
    Route("/api/weather", get_weather),
    Route("/api/evidence", get_evidence),
    Route("/api/commit", post_commit, methods=["POST"]),
]
if (WEB / "assets").exists():
    routes.append(Mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets"))
routes.append(Route("/{path:path}", spa))

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
