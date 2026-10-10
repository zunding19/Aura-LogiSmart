"""Aura command deck. Run with: streamlit run aura_app.py

Five screens, one decision loop:
  Command map     map-first view of the most urgent pharmaceutical shipments
  Alerts inbox    notification-driven list, each alert opens a response
  Guided response Detect, Categorize, Prioritize, Explain, Decide for one shipment
  What-if         change the order and watch risk, runway and money move
  Evidence        infographic of how Aura decides and how often it is right
"""
import pandas as pd
import streamlit as st

from deck import charts
from deck import theme as t
from deck.theme import blocks, chip, esc, flat, hours, usd
from src import intake
from src.data_loader import DEMO_PATH
from src.model import load_model, predict_failure_risk
from src.protocols import record_custody_handshake
from src.recommendations import ACTION_CATALOG, LOSS_SHARE, check_sla_status

st.set_page_config(page_title="Aura Predictive", layout="wide")
st.markdown(f"<style>{t.CSS}</style>", unsafe_allow_html=True)

VIEWS = ["Command map", "Alerts inbox", "Guided response", "What-if simulator", "Evidence"]
STEPS = ["Detect", "Categorize", "Prioritize", "Explain", "Decide"]
WEATHER = ["Clear", "Rain", "Fog", "Storm", "Hurricane"]
TIERS = ["Routine Restock", "Elevated Procedure", "Critical Emergency"]
CORRECTIVE = {"cold_storage_transfer", "refrigerant_refresh", "refrigerated_ground"}
ALERT_LINES = [.30, .40, .50, .60, .70, .80]
# Best action across the 508 test alerts at the 60% line, from notebook 05.
ACTION_MIX = [("Keep monitoring", 271), ("Move to next direct flight", 128),
              ("Refrigerated ground transport", 109)]


def md(markup):
    st.markdown(flat(markup), unsafe_allow_html=True)


# ---------------------------------------------------------------- data

@st.cache_resource
def model():
    return load_model()


@st.cache_data
def shipments():
    return pd.read_csv(DEMO_PATH)


@st.cache_data
def queue(alert_line):
    q = intake.triage_queue(shipments(), model(), alert_line=alert_line).copy()
    to_go = (q["Lead_Time_Days"] * 24 - q["Elapsed_Transit_Hours_Simulated"]).clip(lower=0)
    q["hours_to_go"] = to_go
    # Only passive packaging (air freight) counts down; a powered reefer does not.
    q["runway"] = (q["Remaining_Buffer_Hours_Simulated"] - to_go).where(
        q["Transport_Mode"] == "Air"
    )
    return q.reset_index(drop=True)


@st.cache_data(show_spinner="Scoring shipment")
def assess_shipment(shipment_id):
    rows = shipments()
    match = rows[rows["Shipment_ID"] == shipment_id]
    if match.empty:
        return dict(status=intake.INSUFFICIENT, reason="Shipment not found.")
    return intake.assess_order(match.iloc[0].to_dict(), model=model())


@st.cache_data(show_spinner="Scoring scenario")
def assess_scenario(items):
    return intake.assess_order(dict(items), model=model())


@st.cache_data
def line_options():
    return intake.alert_line_options()


@st.cache_data
def weather_sweep(items):
    """The same shipment under each of the five weather conditions."""
    frame = pd.DataFrame([dict(items, Weather_Condition=w) for w in WEATHER])
    try:
        return predict_failure_risk(frame, model())["p_failure"].tolist()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- state

state = st.session_state
# A link can open a screen directly, e.g. ?view=Guided response&shipment=SC-12132&step=Decide
link = st.query_params
if "view" not in state:
    if link.get("view") in VIEWS:
        state.view = link["view"]
    if link.get("step") in STEPS:
        state.step = link["step"]
    if link.get("shipment"):
        state.selected = link["shipment"]
        if link.get("try") in ACTION_CATALOG:
            state.sim = {link["shipment"]: link["try"]}
state.setdefault("view", VIEWS[0])
state.setdefault("step", STEPS[0])
state.setdefault("alert_line", .60)
state.setdefault("sim", {})         # shipment id -> action being tried
state.setdefault("committed", {})   # shipment id -> actions logged
state.setdefault("custody", [])     # digital handshake records
state.setdefault("flash", None)

Q = queue(state.alert_line)
if "selected" not in state:
    state.selected = Q.iloc[0]["Shipment_ID"] if len(Q) else None


def go(view, shipment_id=None, step=None):
    state.view = view
    if shipment_id is not None:
        state.selected = shipment_id
    if step is not None:
        state.step = step


def try_action(shipment_id, action_id):
    state.sim[shipment_id] = action_id


def commit(shipment_id, action_id, buffer_after, hub):
    title = ACTION_CATALOG[action_id]["title"]
    if action_id in CORRECTIVE:
        try:
            record_custody_handshake(
                shipment_id, "Hub cold-chain crew", "Carrier duty officer",
                buffer_after, f"{hub} hub", True,
                records=state.custody, corrective_action=action_id,
            )
        except ValueError as exc:
            state.flash = f"Handshake not logged: {exc}"
            return
        state.flash = f"{shipment_id}: handshake logged for '{title}'."
    else:
        state.flash = f"{shipment_id}: '{title}' logged."
    state.committed.setdefault(shipment_id, []).append(action_id)


def sla_for(shipment_id, probability, buffer_hours):
    try:
        return check_sla_status(
            probability, buffer_hours, shipment_id=shipment_id,
            custody_records=state.custody,
        )
    except ValueError:
        return None


def build_alerts(top):
    alerts = []
    for row in top.to_dict("records"):
        sid, p = row["Shipment_ID"], row["p_failure"]
        runway = None if pd.isna(row["runway"]) else row["runway"]
        buffer = row["Remaining_Buffer_Hours_Simulated"] if row["Transport_Mode"] == "Air" else None
        sla = sla_for(sid, p, buffer)
        body = (
            f"{p:.0%} disruption risk on {usd(row['Cargo_Value_USD_Simulated'])} of "
            f"pharmaceuticals for a {row['Hospital_Urgency_Tier_Simulated']} order."
        )
        if sla and sla["breach_warning"]:
            kind, tone, head = "SLA breach", "red", "SLA breach notice issued"
            body = sla["breach_notice"]["text"]
        elif sla and sla["trigger_condition"]:
            kind, tone, head = "Resolved", "teal", "Corrective proof verified. Escrow released."
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
            when=f"{row['hours_to_go']:.1f} h to arrival",
        ))
    return alerts


ALERTS = build_alerts(Q.head(12))
OPEN_ALERTS = [a for a in ALERTS if a["kind"] != "Resolved"]


# ---------------------------------------------------------------- shared pieces

def crumbs(view):
    line = next(o for o in line_options() if abs(o["alert_line"] - state.alert_line) < .001)
    md(f"""
    <div class="crumbs">
      <b>Aura</b>{chip("Simulated shipments", "teal")}
      <span class="sep">/</span><b>{esc(view)}</b>
      <span class="push num">{len(Q)} pharma alerts at the {state.alert_line:.0%} line</span>
      <span class="sep">/</span>
      <span class="num">{line['false_alarms_per_1000']:.0f} false alarms per 1,000 tested</span>
    </div>""")


def meter(probability, before=None):
    before = probability if before is None else before
    color = t.risk_color(probability)
    ghost = (
        f'<div class="ghost" style="transform:scaleX({before:.4f});background:{t.risk_color(before)}"></div>'
        if abs(before - probability) > .001 else ""
    )
    return f"""
    <div class="meter">{ghost}
      <div class="fill" style="--v:{probability:.4f};--from:{before:.4f};background:{color}"></div>
      <div class="cuts"></div>
      <div class="line" style="left:{state.alert_line:.0%}"></div>
    </div>
    <div class="scale"><span>0%</span><span>Alert line {state.alert_line:.0%}</span><span>100%</span></div>"""


def problem(result):
    """Shown instead of a crash when the backend cannot score the input."""
    md(f"""<div class="banner red"><b>{esc(result.get('status', intake.INSUFFICIENT))}</b>
    <p>{esc(result.get('reason', ''))}</p></div>""")


def option_after(result, action_id):
    """Buffer and hours to go after a tried action, for the runway chart."""
    option = next((o for o in result["actions"]["options"] if o["action_id"] == action_id), None)
    thermal = result["thermal"]
    if option is None or not option["available"] or thermal["buffer_hours"] is None:
        return option, None
    buffer_after = thermal["buffer_hours"] + ACTION_CATALOG[action_id]["changes"].get("buffer_hours", 0)
    runway_after = option["thermal_runway_hours_after"]
    to_go_after = thermal["hours_to_go"] if runway_after is None else max(buffer_after - runway_after, 0)
    return option, dict(buffer_hours=buffer_after, hours_to_go=to_go_after, title=option["title"])


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    md('<div class="brand"><span class="brand-mark"></span><span class="brand-name">aura</span></div>')
    md('<div class="eyebrow navlabel">Operations</div>')
    labels = {
        "Command map": "Command map",
        "Alerts inbox": f"Alerts inbox  ·  {len(OPEN_ALERTS)}",
        "Guided response": "Guided response",
        "What-if simulator": "What-if simulator",
    }
    for name, label in labels.items():
        st.button(label, key=f"nav_{name}", width="stretch", on_click=go, args=(name,),
                  type="primary" if state.view == name else "secondary")
    md('<div class="eyebrow navlabel">Proof</div>')
    st.button("Evidence", key="nav_Evidence", width="stretch", on_click=go, args=("Evidence",),
              type="primary" if state.view == "Evidence" else "secondary")

    with st.container(key="panel_line"):
        md('<div class="eyebrow">Alert line</div>')
        st.select_slider(
            "Alert line", ALERT_LINES, key="alert_line",
            format_func=lambda v: f"{v:.0%}", label_visibility="collapsed",
        )
        row = next(o for o in line_options() if abs(o["alert_line"] - state.alert_line) < .001)
        md(f"""<p class="note"><span class="num" style="color:{t.TEXT}">{row['precision']:.0%}</span> of alerts are real,
        <span class="num" style="color:{t.TEXT}">{row['recall']:.0%}</span> of disruptions caught on 1,000 test shipments.</p>
        <p class="fine">Lower line: more caught, more false alarms.</p>""")
    md('<p class="fine">Shipments, cargo values and thermal buffers are simulated.</p>')

if state.flash:
    st.toast(state.flash)
    state.flash = None
elif not state.get("greeted") and OPEN_ALERTS:
    st.toast(f"{OPEN_ALERTS[0]['id']}: {OPEN_ALERTS[0]['head']}")
state.greeted = True


# ---------------------------------------------------------------- 1. command map

def queue_row(rank, row, selected):
    p = row["p_failure"]
    color = t.risk_color(p)
    runway = None if pd.isna(row["runway"]) else row["runway"]
    runway_text = (
        "powered reefer" if runway is None
        else f'<span style="color:{t.RED if runway < 0 else t.TEXT_2}">runway {hours(runway)}</span>'
    )
    return f"""
    <div class="qrow" style="--i:{min(rank, 6)}">
      <div class="rank">{rank:02d}</div>
      <div>
        <div class="id" style="color:{t.TEXT if selected else t.TEXT_2}">{esc(row['Shipment_ID'])}</div>
        <div class="route">{esc(row['Origin_Port'])} to {esc(row['Destination_Port'])} · {esc(row['Transport_Mode'])} · {esc(row['Weather_Condition'])}</div>
        <div class="meta">{usd(row['Cargo_Value_USD_Simulated'])} · {esc(row['Hospital_Urgency_Tier_Simulated'])} · {runway_text}</div>
      </div>
      <div class="risk" style="color:{color}">{p:.0%}<small>{blocks(p, 10, color)}</small></div>
    </div>"""


def brief(sid):
    """The selected shipment in one glance, under the map."""
    result = assess_shipment(sid) if sid else {}
    if result.get("status") != "ok":
        return
    ship, risk, thermal = result["shipment"], result["risk"], result["thermal"]
    p = risk["p_failure"]
    best = result["actions"]["recommended_title"]
    with st.container(key="panel_brief"):
        info, action = st.columns([4, 1], vertical_alignment="top")
        with info:
            md(f"""
            <div class="eyebrow">Selected on the map</div>
            <div style="display:flex;align-items:baseline;gap:12px;margin-top:8px;flex-wrap:wrap">
              <span class="mid num">{esc(sid)}</span>
              <span class="mid" style="color:{t.risk_color(p)}">{p:.0%}</span>
              <span class="note">{esc(ship['Origin_Port'])} to {esc(ship['Destination_Port'])} by {esc(ship['Transport_Mode']).lower()}</span>
            </div>""")
        action.button("Respond", key="brief_open", type="primary", width="stretch",
                      on_click=go, args=("Guided response", sid, STEPS[0]))
        md(f"""
        {meter(p)}
        <p class="note" style="margin-top:12px">{esc(result['explanation']['summary'])}</p>
        <div class="kv" style="margin-top:16px">
          <div><div class="eyebrow">Failure mode</div><div class="v" style="font-size:15px;line-height:1.3">{esc(result['failure_mode']['mode'])}</div></div>
          <div><div class="eyebrow">Thermal runway</div><div class="v" style="font-size:15px;line-height:1.3;color:{t.RED if (thermal['runway_hours'] or 0) < 0 else t.TEXT}">{hours(thermal['runway_hours'])}</div></div>
          <div><div class="eyebrow">Best action</div><div class="v" style="font-size:15px;line-height:1.3">{esc(best)}</div></div>
        </div>""")


def view_map():
    crumbs("Command map")
    md('<h1 class="title">Which shipment needs you first</h1>'
       '<p class="subtitle">The ten most urgent pharmaceutical shipments, ranked by risk, cargo value, '
       'hospital urgency and hours of thermal buffer left. Pick one on the map or in the queue.</p>')
    if Q.empty:
        md('<div class="banner"><b>No alerts at this line</b><p>Lower the alert line in the sidebar to see more shipments.</p></div>')
        return

    top = Q.head(10)
    left, right = st.columns([1.3, 1], gap="medium")

    with left, st.container(key="panel_map"):
        md('<div class="h2">Live routes</div><p class="note">Squares mark shipments. '
           'Red is critical (80% and above), amber is high. Click a square to select it.</p>')
        rows = [
            dict(r, label=(
                f"<b>{r['Shipment_ID']}</b><br>{r['Origin_Port']} to {r['Destination_Port']} "
                f"by {r['Transport_Mode'].lower()}<br>{r['p_failure']:.0%} risk · "
                f"{usd(r['Cargo_Value_USD_Simulated'])}"
            ))
            for r in top.to_dict("records")
        ]
        event = st.plotly_chart(
            charts.route_map(rows, state.selected), key="map", on_select="rerun",
            selection_mode="points", config=charts.CONFIG, theme=None, width="stretch",
        )
        points = (event or {}).get("selection", {}).get("points", [])
        picked = points[0].get("customdata") if points else None
        if picked and picked != state.get("map_pick"):
            state.map_pick = picked
            state.selected = picked
            st.rerun()

        short = int((top["runway"] < 0).sum())
        breaches = sum(a["kind"] == "SLA breach" for a in ALERTS)
        md(f"""
        <div class="kv" style="margin-top:8px">
          <div><div class="eyebrow">In the queue</div><div class="v">{len(Q)}</div><div class="s">pharma alerts, air and road</div></div>
          <div><div class="eyebrow">Expected loss, top 10</div><div class="v">{usd(top['expected_loss_usd'].sum())}</div><div class="s">risk x 10% of cargo value</div></div>
          <div><div class="eyebrow">Buffer ends early</div><div class="v" style="color:{t.RED if short else t.TEXT}">{short} of {len(top)}</div><div class="s">before the shipment arrives</div></div>
          <div><div class="eyebrow">SLA notices</div><div class="v" style="color:{t.RED if breaches else t.TEXT}">{breaches}</div><div class="s">escrow locked (simulated)</div></div>
        </div>""")

    with left:
        brief(state.selected)

    with right, st.container(key="panel_queue"):
        md('<div class="h2">Triage queue</div><p class="note" style="margin-bottom:12px">Most urgent first.</p>')
        for rank, row in enumerate(top.to_dict("records"), start=1):
            sid = row["Shipment_ID"]
            with st.container(key=f"row_q_{sid}"):
                info, action = st.columns([4, 1.2], vertical_alignment="center")
                with info:
                    md(queue_row(rank, row, sid == state.selected))
                action.button(
                    "Open", key=f"open_{sid}", width="stretch",
                    type="primary" if sid == state.selected else "secondary",
                    on_click=go, args=("Guided response", sid, STEPS[0]),
                )


# ---------------------------------------------------------------- 2. alerts inbox

def view_inbox():
    crumbs("Alerts inbox")
    md('<h1 class="title">Alerts that need a decision</h1>'
       '<p class="subtitle">Each alert says what went wrong and how long you have. '
       'Start a response to see the cause and the action that pays.</p>')
    kinds = ["All"] + sorted({a["kind"] for a in ALERTS})
    pick = st.segmented_control("Filter", kinds, default="All", key="inbox_filter",
                                required=True, label_visibility="collapsed")
    shown = [a for a in ALERTS if pick in (None, "All", a["kind"])]
    if not shown:
        md('<div class="banner"><b>Nothing here</b><p>No alerts of this kind right now.</p></div>')
        return

    with st.container(key="panel_inbox"):
        for index, alert in enumerate(shown):
            with st.container(key=f"row_a_{alert['id']}" if index else f"first_a_{alert['id']}"):
                info, action = st.columns([5, 1.2], vertical_alignment="center")
                done = state.committed.get(alert["id"])
                with info:
                    md(f"""
                    <div class="alert">
                      <div>
                        <div class="chips">{chip(alert['kind'], alert['tone'])}
                          <span class="num" style="font-size:13px;color:{t.TEXT_2}">{esc(alert['id'])}</span>
                          <span class="note">{esc(alert['route'])}</span>
                          {chip('Action logged', 'teal') if done else ''}</div>
                        <div class="head" style="margin-top:8px">{esc(alert['head'])}</div>
                        <div class="body">{esc(alert['body'])}</div>
                      </div>
                      <div class="when">{esc(alert['when'])}</div>
                    </div>""")
                action.button(
                    "Review" if done or alert["kind"] == "Resolved" else "Start response",
                    key=f"alert_{alert['id']}", width="stretch",
                    type="primary" if index == 0 and not done else "secondary",
                    on_click=go, args=("Guided response", alert["id"], STEPS[0]),
                )
    md('<p class="fine">SLA notices and escrow are a simulation. No notice is sent and no payment is held.</p>')


# ---------------------------------------------------------------- 3. guided response

def rail(result, sid):
    """Right-hand column: the live numbers that react to a tried action."""
    risk, thermal, ship = result["risk"], result["thermal"], result["shipment"]
    before = risk["p_failure"]
    tried = state.sim.get(sid) if state.step == "Decide" else None
    option, after = option_after(result, tried) if tried else (None, None)
    now = option["risk_after"] if option else before
    runway_now = option["thermal_runway_hours_after"] if option else thermal["runway_hours"]
    level = "Critical" if now >= .8 else "High" if now >= .6 else "Medium" if now >= .3 else "Low"

    with st.container(key="panel_rail"):
        was = f'<span class="note"> was {before:.0%}</span>' if option and abs(now - before) > .001 else ""
        md(f"""
        <div class="eyebrow">Disruption risk{' after action' if option else ''}</div>
        <div style="display:flex;align-items:baseline;gap:12px;margin-top:8px">
          <span class="big" style="color:{t.risk_color(now)}">{now:.0%}</span>
          {chip(level, 'red' if level == 'Critical' else 'amber' if level == 'High' else 'teal' if level == 'Low' else '')}{was}
        </div>
        {meter(now, before)}
        <div class="kv" style="margin-top:16px">
          <div><div class="eyebrow">Cargo</div><div class="v">{usd(ship.get('Cargo_Value_USD_Simulated'))}</div></div>
          <div><div class="eyebrow">Thermal runway</div>
            <div class="v" style="color:{t.RED if (runway_now or 0) < 0 else t.TEAL if option and option.get('covers_thermal_gap') else t.TEXT}">{hours(runway_now)}</div></div>
        </div>""")

        if option:
            fee = option["action_cost_usd"]
            if option["covers_thermal_gap"] and abs(now - before) < .001:
                text = (
                    f"Thermal runway goes from {hours(thermal['runway_hours'])} to {hours(runway_now)}. "
                    f"The {usd(option['cargo_protected_usd'])} batch is protected for a {usd(fee)} fee. "
                    "The risk score does not move: more cooling does not stop a storm."
                )
                tone, head = "teal", "Batch protected"
            elif (option["net_saving_usd"] or 0) > 0:
                text = (
                    f"Risk falls from {before:.0%} to {now:.0%}. Expected loss avoided, "
                    f"less the {usd(fee)} fee."
                )
                tone, head = "teal", f"Net saving {usd(option['net_saving_usd'])}"
            elif tried == "stock_diversion":
                tone, head, text = "", "Hospital covered", option["effect"]
            elif tried == "monitor":
                tone, head, text = "", "No spend", option["effect"]
            else:
                tone, head = "", f"Costs {usd(fee)}, no risk saving"
                text = option["effect"]
            md(f'<div class="banner {tone}" style="margin-top:16px"><b>{esc(head)}</b><p>{esc(text)}</p></div>')

        if thermal["curve"] and thermal["buffer_hours"] is not None:
            md('<div class="eyebrow" style="margin-top:8px">Thermal runway</div>')
            st.plotly_chart(
                charts.thermal_chart(thermal["buffer_hours"], thermal["hours_to_go"], after),
                config=charts.CONFIG, theme=None, width="stretch", key=f"thermal_{sid}_{tried}",
            )
            md('<p class="fine">Buffer hours, not temperature. The data has no temperature readings.</p>')
        elif thermal.get("note"):
            md(f'<p class="fine">{esc(thermal["note"])}</p>')

        if option:
            logged = tried in state.committed.get(sid, [])
            st.button(
                "Logged" if logged else "Commit and log handshake" if tried in CORRECTIVE else "Commit action",
                key=f"commit_{sid}_{tried}", type="primary", width="stretch", disabled=logged,
                on_click=commit,
                args=(sid, tried, (after or {}).get("buffer_hours", 0), ship.get("Origin_Port", "Origin")),
            )


def step_detect(result):
    ship, risk, score = result["shipment"], result["risk"], result["scorecard"]
    flagged = risk["p_failure"] >= state.alert_line
    md(f"""
    <div class="h2">Will it be disrupted?</div>
    <p class="note">Aura scores this shipment at <b style="color:{t.TEXT}">{risk['p_failure']:.0%}</b>.
    The alert line is {state.alert_line:.0%}, so it is {'flagged' if flagged else 'not flagged'}.
    {'The backup model was used because weather is missing.' if risk['model_variant'] != 'primary' else ''}</p>
    <div class="eyebrow" style="margin-top:24px">What the model read</div>
    <div class="kv" style="margin-top:8px;row-gap:16px">
      <div><div class="eyebrow">Weather</div><div class="v">{esc(ship.get('Weather_Condition') or 'Not entered')}</div></div>
      <div><div class="eyebrow">Route risk</div><div class="v">{ship['Geopolitical_Risk_Score']:.1f} / 10</div></div>
      <div><div class="eyebrow">Carrier</div><div class="v">{ship['Carrier_Reliability_Score']:.2f} / 1</div></div>
      <div><div class="eyebrow">Lead time</div><div class="v">{ship['Lead_Time_Days']:.1f} d</div></div>
      <div><div class="eyebrow">Distance</div><div class="v">{ship['Distance_km']:,.0f} km</div></div>
    </div>
    <div class="eyebrow" style="margin-top:24px">Can you trust the alert?</div>
    <p class="note" style="margin-top:8px">On {score['tested_on_shipments']:,} held-out simulated shipments,
    {score['precision']:.0%} of alerts were real disruptions and Aura caught about 7 in 10 of them.</p>""")


def step_categorize(result):
    mode, explain = result["failure_mode"], result["explanation"]
    signs = "".join(f"""
        <div class="driver"><div class="top"><span>{esc(s['category'])}</span>
        <span>{blocks(s['severity'], 10, t.RED if s['severity'] >= .8 else t.AMBER)}</span></div>
        <p>{esc(s['description'])}</p></div>""" for s in explain["warning_signs"])
    md(f"""
    <div class="h2">What kind of trouble?</div>
    <div class="mid" style="margin-top:12px">{esc(mode['mode'])}</div>
    <p class="note" style="margin-top:8px">{esc(mode['reason'])}</p>
    <div class="chips" style="margin-top:16px">{chip('Risk category: ' + explain['category'])}</div>
    <div class="eyebrow" style="margin:24px 0 12px">Warning signs</div>
    {signs or '<p class="note">No warning sign is above its threshold.</p>'}""")


def step_prioritize(result, sid):
    priority, ship, risk = result["priority"], result["shipment"], result["risk"]
    ids = list(Q["Shipment_ID"])
    rank = ids.index(sid) + 1 if sid in ids else None
    buffer = ship.get("Remaining_Buffer_Hours_Simulated")
    md(f"""
    <div class="h2">Which first?</div>
    <div class="mid" style="margin-top:12px">{f'Rank {rank} of {len(Q)}' if rank else 'Below the alert line'}</div>
    <p class="note" style="margin-top:8px">Urgency weighs risk, money, the hospital's need and how little buffer is left.</p>
    <div class="formula" style="margin-top:16px">
      urgency = risk x cargo value x urgency weight / max(buffer hours, 0.5)<br>
      <b>{risk['p_failure']:.0%}</b> x <b>{usd(ship.get('Cargo_Value_USD_Simulated'))}</b> x
      <b>{priority['urgency_weight'] or 0:g}</b> ({esc(ship.get('Hospital_Urgency_Tier_Simulated'))}) /
      max(<b>{buffer if buffer is not None else 0:g} h</b>, 0.5)
    </div>
    <p class="note" style="margin-top:16px">Expected loss if nothing is done:
    <b style="color:{t.TEXT}">{usd(priority['expected_loss_usd'])}</b> (risk x 10% of cargo value).</p>
    <div class="eyebrow" style="margin:24px 0 12px">Top of the queue</div>""")
    top = Q.head(5)
    peak = top["urgency_score"].max() or 1
    md("".join(f"""
        <div class="driver"><div class="top">
          <span class="num" style="color:{t.TEXT if r['Shipment_ID'] == sid else t.TEXT_2}">{i:02d} {esc(r['Shipment_ID'])}</span>
          <span>{blocks(r['urgency_score'] / peak, 20, t.TEAL if r['Shipment_ID'] == sid else t.QUIET)}</span>
        </div></div>""" for i, r in enumerate(top.to_dict("records"), start=1)))


def step_explain(result):
    explain = result["explanation"]
    drivers = "".join(f"""
        <div class="driver"><div class="top"><span>{esc(d['label'])}</span>
        <span class="num" style="color:{t.TEXT_2}">{d['contribution'] * 100:+.0f} pts {blocks(abs(d['contribution']), 10, t.RED if d['contribution'] >= .2 else t.NEUTRAL)}</span></div>
        <p>{esc(d['description'])}</p></div>""" for d in explain["drivers"][:3])
    md(f"""
    <div class="h2">Why?</div>
    <p class="note" style="margin-top:8px">{esc(explain['summary'])}</p>
    <div class="eyebrow" style="margin:24px 0 12px">Top 3 drivers</div>{drivers}""")


def step_decide(result, sid):
    actions = result["actions"]
    tried = state.sim.get(sid)
    md('<div class="h2">What to do?</div><p class="note" style="margin-bottom:16px">'
       'Try an action. The risk, the thermal runway and the money on the right update at once. '
       'Nothing is sent to anyone until you commit.</p>')
    widths = [2.6, .7, .8, 1.5, 1.3, 1]
    head = st.columns(widths)
    for col, text in zip(head, ["Action", "Fee", "Risk after", "Runway after", "Net saving", ""]):
        col.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)

    for o in actions["options"]:
        aid = o["action_id"]
        tags = ""
        if aid == actions["recommended_action_id"]:
            tags += chip("Best net saving" if aid != "monitor" else "Nothing pays more", "teal")
        if aid == actions["thermal_action_id"]:
            tags += chip("Cheapest way to cover the buffer", "teal")
        if aid == actions["backup_action_id"]:
            tags += chip("Backup plan", "amber")
        with st.container(key=f"row_act_{aid}"):
            c = st.columns(widths, vertical_alignment="center")
            c[0].markdown(flat(f"""
                <div style="font-size:14px;font-weight:500;color:{t.TEXT}">{esc(o['title'])}</div>
                <div class="chips" style="margin-top:4px">{tags}</div>"""), unsafe_allow_html=True)
            fee = "n/a" if aid == "stock_diversion" else usd(o["action_cost_usd"])
            c[1].markdown(f'<span class="num">{fee}</span>', unsafe_allow_html=True)
            if not o["available"]:
                c[2].markdown(f'<span class="fine">{esc(o["unavailable_reason"])}</span>', unsafe_allow_html=True)
                c[5].button("Air only", key=f"try_{sid}_{aid}", disabled=True, width="stretch")
                continue
            moved = o["risk_after"] < o["risk_before"] - .001
            c[2].markdown(
                f'<span class="num" style="color:{t.risk_color(o["risk_after"]) if moved else t.TEXT_2}">'
                f'{o["risk_after"]:.0%}</span>',
                unsafe_allow_html=True)
            runway = o["thermal_runway_hours_after"]
            c[3].markdown(
                f'<span class="num" style="color:{t.TEAL if o["covers_thermal_gap"] else t.RED if (runway or 0) < 0 else t.TEXT_2}">'
                f'{hours(runway)}</span>', unsafe_allow_html=True)
            net = o["net_saving_usd"]
            c[4].markdown(
                f'<span class="num" style="color:{t.TEAL if (net or 0) > 0 else t.TEXT_2}">'
                f'{"n/a" if aid == "stock_diversion" else usd(net)}</span>', unsafe_allow_html=True)
            c[5].button(
                "Shown" if tried == aid else "Try", key=f"try_{sid}_{aid}", width="stretch",
                type="primary" if tried == aid or (tried is None and aid == actions["recommended_action_id"]) else "secondary",
                on_click=try_action, args=(sid, aid),
            )
    md('<p class="fine">A disruption is assumed to cost 10% of cargo value. Fees are example figures. '
       'Savings are the model\'s estimate, not a guarantee.</p>')


def view_guided():
    crumbs("Guided response")
    sid = state.selected
    if sid is None:
        md('<div class="banner"><b>No shipment selected</b><p>Pick one from the command map.</p></div>')
        return
    result = assess_shipment(sid)
    if result.get("status") != "ok":
        md(f'<h1 class="title">{esc(sid)}</h1>')
        problem(result)
        return

    ship, risk, thermal = result["shipment"], result["risk"], result["thermal"]
    raw = shipments()
    flag = bool(raw.loc[raw["Shipment_ID"] == sid, "Flag_Land_Route_Across_Ocean"].fillna(False).any())
    md(f"""
    <h1 class="title"><span class="num">{esc(sid)}</span></h1>
    <p class="subtitle" style="margin-bottom:12px">{esc(ship['Origin_Port'])} to {esc(ship['Destination_Port'])}
    by {esc(ship['Transport_Mode']).lower()}, {thermal['hours_to_go'] or 0:.1f} hours still to go.</p>
    <div class="chips" style="margin-bottom:24px">
      {chip(ship.get('Weather_Condition') or 'Weather not entered', 'red' if ship.get('Weather_Condition') in ('Storm', 'Hurricane') else '')}
      {chip(ship.get('Hospital_Urgency_Tier_Simulated') or 'No urgency tier', 'red' if ship.get('Hospital_Urgency_Tier_Simulated') == 'Critical Emergency' else '')}
      {chip('Storage ' + str(ship.get('Storage_Standard_Simulated')))}
      {chip('Data check: land route crosses an ocean', 'amber') if flag else ''}
    </div>""")

    sla = sla_for(sid, risk["p_failure"], thermal["buffer_hours"] if thermal["buffer_counts_down"] else None)
    if sla and sla["trigger_condition"]:
        tone = "red" if sla["breach_warning"] else "teal"
        md(f"""<div class="banner {tone}"><b>{esc(sla['status'])}</b>
        <p>{esc(sla['breach_notice']['text'] if sla['breach_notice'] else 'A corrective handshake with at least 3 hours of buffer is on record.')}
        Simulated: no real notice is sent.</p></div>""")

    st.segmented_control(
        "Step", STEPS, key="step", required=True, label_visibility="collapsed",
        format_func=lambda s: f"{STEPS.index(s) + 1}  {s}",
    )
    step = state.step or STEPS[0]
    left, right = st.columns([1.7, 1], gap="medium")
    with left:
        with st.container(key="panel_step"):
            if step == "Detect":
                step_detect(result)
            elif step == "Categorize":
                step_categorize(result)
            elif step == "Prioritize":
                step_prioritize(result, sid)
            elif step == "Explain":
                step_explain(result)
            else:
                step_decide(result, sid)
        index = STEPS.index(step)
        back, _, forward = st.columns([1, 2.4, 1.4])
        if index:
            back.button("Back", key="step_back", width="stretch",
                        on_click=go, args=("Guided response", None, STEPS[index - 1]))
        if index < len(STEPS) - 1:
            forward.button(f"Continue to {STEPS[index + 1]}", key="step_next", type="primary",
                           width="stretch", on_click=go, args=("Guided response", None, STEPS[index + 1]))
        else:
            forward.button("Back to alerts", key="step_done", width="stretch",
                           on_click=go, args=("Alerts inbox",))
    with right:
        rail(result, sid)


# ---------------------------------------------------------------- 4. what-if simulator

DEFAULTS = dict(
    wi_origin="Frankfurt", wi_dest="Kuala Lumpur", wi_mode="Air", wi_weather="Storm",
    wi_auto_risk=True, wi_risk=5.0, wi_carrier=.75, wi_lead=3.0, wi_pct=60,
    wi_buffer=12.0, wi_value=850_000, wi_tier="Critical Emergency", wi_weather_src=None,
)
for key, value in DEFAULTS.items():
    state.setdefault(key, value)


def load_selected():
    result = assess_shipment(state.selected) if state.selected else {}
    if result.get("status") != "ok":
        state.flash = "That shipment could not be loaded."
        return
    ship = result["shipment"]
    planned = max(ship["Lead_Time_Days"] * 24, 1)
    state.update(
        wi_origin=ship["Origin_Port"], wi_dest=ship["Destination_Port"],
        wi_mode=ship["Transport_Mode"], wi_weather=ship.get("Weather_Condition") or "Clear",
        wi_auto_risk=False, wi_risk=float(min(max(ship["Geopolitical_Risk_Score"], 0), 10)),
        wi_carrier=float(min(max(ship["Carrier_Reliability_Score"], .3), 1)),
        wi_lead=float(min(max(round(ship["Lead_Time_Days"] * 2) / 2, .5), 15)),
        wi_pct=int(min(100 * (ship.get("Elapsed_Transit_Hours_Simulated") or 0) / planned, 100)),
        wi_buffer=float(min(ship.get("Remaining_Buffer_Hours_Simulated") or 0, 120)),
        wi_value=int(min(max(ship.get("Cargo_Value_USD_Simulated") or 50_000, 50_000), 6_000_000)),
        wi_tier=ship.get("Hospital_Urgency_Tier_Simulated") if ship.get("Hospital_Urgency_Tier_Simulated") in TIERS else TIERS[0],
        wi_weather_src=None,
    )


def live_weather():
    try:
        condition, source = intake.lookup_weather(state.wi_origin, state.wi_dest)
    except Exception:
        condition, source = None, None
    if condition in WEATHER:
        state.wi_weather, state.wi_weather_src = condition, source
    else:
        state.flash = "Live weather is unavailable. Pick the weather by hand."


def view_whatif():
    crumbs("What-if simulator")
    md('<h1 class="title">What if the order changes?</h1>'
       '<p class="subtitle">Move a control and Aura re-scores the order: risk, thermal runway, '
       'the action that pays and what each choice leaves at stake.</p>')
    places = sorted(intake.PLACES)
    left, right = st.columns([1, 1.5], gap="medium")

    with left, st.container(key="panel_controls"):
        top = st.columns([1, 1])
        top[0].markdown('<div class="h2">Order</div>', unsafe_allow_html=True)
        if state.selected:
            top[1].button(f"Load {state.selected}", key="wi_load", width="stretch", on_click=load_selected)
        a, b = st.columns(2)
        a.selectbox("Origin", places, key="wi_origin")
        b.selectbox("Destination", places, key="wi_dest")
        st.segmented_control("Mode", ["Air", "Road"], key="wi_mode", required=True)
        a, b = st.columns([1.6, 1], vertical_alignment="bottom")
        a.select_slider("Weather on the route", WEATHER, key="wi_weather")
        b.button("Use live weather", key="wi_live", width="stretch", on_click=live_weather)
        if state.wi_weather_src:
            md(f'<p class="fine" style="margin-top:0">{esc(state.wi_weather_src)}</p>')
        st.slider("Carrier reliability (0 to 1)", .30, 1.0, step=.05, key="wi_carrier")
        st.toggle("Route risk from the World Bank stability index", key="wi_auto_risk")
        if not state.wi_auto_risk:
            st.slider("Route risk (0 to 10)", 0.0, 10.0, step=.1, key="wi_risk")
        st.slider("Planned lead time (days)", .5, 15.0, step=.5, key="wi_lead")
        st.slider("Trip completed (%)", 0, 100, step=5, key="wi_pct")
        st.slider("Thermal buffer left (hours)", 0.0, 120.0, step=1.0, key="wi_buffer")
        st.slider("Cargo value (USD)", 50_000, 6_000_000, step=50_000, key="wi_value", format="$%d")
        st.selectbox("Hospital urgency", TIERS, key="wi_tier")

    order = {
        "Shipment_ID": "WHAT-IF",
        "Place_of_Origin": state.wi_origin, "Place_of_Destination": state.wi_dest,
        "Transport_Mode": state.wi_mode or "Air", "Weather": state.wi_weather,
        "Carrier_Reliability_Score": state.wi_carrier,
        "Lead-Time in days": state.wi_lead,
        "Transit_Duration": round(state.wi_lead * 24 * state.wi_pct / 100, 1),
        "Buffer_of_Package": state.wi_buffer, "Cargo_Value_USD": state.wi_value,
        "Hospital_Urgency_Tier": state.wi_tier,
    }
    if not state.wi_auto_risk:
        order["Geopolitical_Risk_Score"] = state.wi_risk

    with right:
        if state.wi_origin == state.wi_dest:
            problem(dict(status=intake.INSUFFICIENT, reason="Origin and destination are the same place."))
            return
        result = assess_scenario(tuple(sorted(order.items())))
        if result.get("status") != "ok":
            problem(result)
            return
        risk, thermal, actions = result["risk"], result["thermal"], result["actions"]
        p = risk["p_failure"]
        best = next(o for o in actions["options"] if o["action_id"] == actions["recommended_action_id"])
        with st.container(key="panel_result"):
            md(f"""
            <div class="eyebrow">Disruption risk</div>
            <div style="display:flex;align-items:baseline;gap:12px;margin-top:8px;flex-wrap:wrap">
              <span class="big" style="color:{t.risk_color(p)}">{p:.0%}</span>
              {chip(risk['risk_level'], 'red' if p >= .8 else 'amber' if p >= .6 else 'teal' if p < .3 else '')}
              <span class="note">{esc(result['failure_mode']['mode'])}</span>
            </div>
            {meter(p)}
            <div class="kv" style="margin-top:16px">
              <div><div class="eyebrow">Thermal runway</div>
                <div class="v" style="color:{t.RED if (thermal['runway_hours'] or 0) < 0 else t.TEXT}">{hours(thermal['runway_hours'])}</div>
                <div class="s">{thermal['hours_to_go'] or 0:.1f} h still to go</div></div>
              <div><div class="eyebrow">Expected loss</div>
                <div class="v">{usd(result['priority']['expected_loss_usd'])}</div><div class="s">if nothing is done</div></div>
              <div><div class="eyebrow">Best action</div>
                <div class="v" style="font-size:16px;line-height:1.3">{esc(best['title'])}</div>
                <div class="s" style="color:{t.TEAL if (best['net_saving_usd'] or 0) > 0 else t.MUTED}">net saving {usd(best['net_saving_usd'])}</div></div>
            </div>""")
            if actions["thermal_action_id"]:
                cover = next(o for o in actions["options"] if o["action_id"] == actions["thermal_action_id"])
                md(f"""<div class="banner teal" style="margin-top:16px"><b>Buffer gap: {esc(cover['title'])}</b>
                <p>Cheapest action that makes the buffer cover the trip: {usd(cover['action_cost_usd'])} fee,
                runway {hours(cover['thermal_runway_hours_after'])}.</p></div>""")

        with st.container(key="panel_cost"):
            md('<div class="h2">What each choice leaves at stake</div>'
               '<p class="note">Grey is expected loss after the action, the dark tip is its fee. Teal is the cheapest total.</p>')
            st.plotly_chart(
                charts.cost_chart(actions["options"], p, state.wi_value, LOSS_SHARE),
                config=charts.CONFIG, theme=None, width="stretch", key="wi_cost",
            )
        sweep = weather_sweep(tuple(sorted(result["shipment"].items(), key=lambda kv: kv[0])))
        if sweep:
            with st.container(key="panel_sweep"):
                cells = "".join(f"""
                    <div><div class="eyebrow" style="color:{t.TEXT if w == state.wi_weather else t.MUTED}">{w}</div>
                    <div class="v" style="color:{t.risk_color(v)}">{v:.0%}</div>
                    <div class="s">{blocks(v, 10, t.risk_color(v))}</div></div>""" for w, v in zip(WEATHER, sweep))
                md(f'<div class="h2" style="margin-bottom:12px">Same order, five skies</div><div class="kv">{cells}</div>')
        if result["inferred"]:
            notes = "".join(f"<li>{esc(text)}</li>" for text in result["inferred"].values() if text)
            md(f'<p class="fine">What Aura assumed:</p><ul class="fine" style="padding-left:16px">{notes}</ul>')


# ---------------------------------------------------------------- 5. evidence

def view_evidence():
    crumbs("Evidence")
    md('<h1 class="title">How Aura decides, and how often it is right</h1>'
       '<p class="subtitle">Every figure here comes from the trained model and 1,000 held-out '
       'simulated shipments. None of it is real shipment data.</p>')
    raw = shipments()
    pharma = int((raw["Product_Category"] == "Pharmaceuticals").sum())
    top_id = Q.iloc[0]["Shipment_ID"] if len(Q) else "none"
    md(f"""
    <div class="stage">
      <div><div class="n">01</div><div class="verb">Detect</div><div class="fig">{len(raw):,}</div>
        <p>simulated shipments scored from 0 to 100% disruption risk.</p></div>
      <div><div class="n">02</div><div class="verb">Categorize</div><div class="fig">{pharma:,}</div>
        <p>pharmaceutical shipments, each tagged as delay, cold-chain excursion or both.</p></div>
      <div><div class="n">03</div><div class="verb">Prioritize</div><div class="fig">{len(Q)}</div>
        <p>air and road alerts at the {state.alert_line:.0%} line, ranked. {esc(top_id)} is first.</p></div>
      <div><div class="n">04</div><div class="verb">Explain</div><div class="fig">3</div>
        <p>top drivers per shipment, in plain words with a what-if for each.</p></div>
      <div><div class="n">05</div><div class="verb">Decide</div><div class="fig">6</div>
        <p>costed actions. The one with the highest net saving is recommended.</p></div>
    </div>""")

    row = next(o for o in line_options() if abs(o["alert_line"] - state.alert_line) < .001)
    alerts, false = round(row["alerts_per_1000"]), round(row["false_alarms_per_1000"])
    missed = round(row["missed_per_1000"])
    caught, quiet = alerts - false, 1000 - alerts - missed
    flag_all = 1000 - caught - missed  # false alarms if every shipment were flagged
    groups = [("Disruptions caught", caught, t.TEAL), ("False alarms", false, t.RED),
              ("Disruptions missed", missed, t.AMBER), ("Correctly left alone", quiet, "#CFDCDB")]

    md('<div style="height:24px"></div>')
    left, right = st.columns([1.2, 1], gap="medium")
    with left, st.container(key="panel_waffle"):
        cells = "".join(f'<span style="background:{color}"></span>' * count for _, count, color in groups)
        legend = "".join(f'<i style="background:{color}"></i><span>{name}</span><b>{count}</b>'
                         for name, count, color in groups)
        md(f"""
        <div class="h2">1,000 test shipments at the {state.alert_line:.0%} alert line</div>
        <p class="note" style="margin-bottom:16px">One square is one shipment the model never saw in training.
        Move the alert line in the sidebar to see the trade.</p>
        <div class="waffle">{cells}</div>
        <div class="legend" style="margin-top:16px;max-width:520px">{legend}</div>""")

    with right:
        with st.container(key="panel_score"):
            md(f"""
            <div class="h2">Fewer false alarms</div>
            <p class="note" style="margin-bottom:16px">Against flagging every shipment, which is what a cautious team does today.</p>
            <div class="driver"><div class="top"><span>Aura at {state.alert_line:.0%}</span><span class="num">{false}</span></div>
              <div style="margin-top:8px">{blocks(false / flag_all, 40, t.RED)}</div></div>
            <div class="driver"><div class="top"><span>Flag everything</span><span class="num">{flag_all}</span></div>
              <div style="margin-top:8px">{blocks(1, 40, t.QUIET)}</div></div>
            <div class="kv" style="margin-top:16px">
              <div><div class="eyebrow">Precision</div><div class="v">{row['precision']:.0%}</div><div class="s">of alerts are real</div></div>
              <div><div class="eyebrow">Recall</div><div class="v">{row['recall']:.0%}</div><div class="s">of disruptions caught</div></div>
              <div><div class="eyebrow">ROC AUC</div><div class="v">{model().metrics['roc_auc']:.2f}</div><div class="s">ranking quality</div></div>
            </div>
            <p class="fine">At the default 60% line recall is about 7 in 10: it averages 69% over 30 random splits.</p>""")
        with st.container(key="panel_mix"):
            total = sum(count for _, count in ACTION_MIX)
            rows = "".join(f"""
                <div class="driver"><div class="top"><span>{name}</span><span class="num">{count}</span></div>
                <div style="margin-top:8px">{blocks(count / total, 40, t.TEAL if name != 'Keep monitoring' else t.QUIET)}</div></div>"""
                for name, count in ACTION_MIX)
            md(f"""<div class="h2">A decision for every alert</div>
            <p class="note" style="margin-bottom:16px">Best action across the {total} test alerts at the 60% line.
            Aura says keep monitoring when no action pays.</p>{rows}""")
    md('<p class="fine">Assumptions: a disruption costs 10% of cargo value; cargo values, thermal buffers and '
       'urgency tiers are simulated; action fees are example figures.</p>')


{
    "Command map": view_map,
    "Alerts inbox": view_inbox,
    "Guided response": view_guided,
    "What-if simulator": view_whatif,
    "Evidence": view_evidence,
}[state.view]()
