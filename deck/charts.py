"""Plotly figures for the command deck: route map, thermal runway and cost comparison."""
import math

import plotly.graph_objects as go

from deck import theme as t
from src.intake import PLACES

AXIS = dict(
    gridcolor=t.LINE, zerolinecolor=t.LINE_STRONG, linecolor=t.LINE_STRONG,
    tickfont=dict(family=t.MONO, size=11, color=t.MUTED),
    title_font=dict(family=t.FONT, size=12, color=t.MUTED),
)
HOVER = dict(
    bgcolor=t.RAISED, bordercolor=t.LINE_STRONG,
    font=dict(family=t.FONT, size=13, color=t.TEXT),
)
CONFIG = dict(displayModeBar=False, scrollZoom=False)
# Ports that sit close together get their labels on different sides.
LABEL_SIDE = {"Rotterdam": "middle left", "Antwerp": "bottom center", "Hamburg": "top right",
              "Busan": "top right", "Shanghai": "bottom right"}


def _base(height):
    return dict(
        height=height, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=t.FONT, color=t.TEXT_2, size=12),
        showlegend=False, hoverlabel=HOVER,
    )


def _midpoint(start, end):
    """Halfway along the great circle, so the marker sits on the drawn route."""
    lat1, lon1, lat2, lon2 = map(math.radians, [*start, *end])
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    lat = math.atan2(
        math.sin(lat1) + math.sin(lat2),
        math.sqrt((math.cos(lat1) + bx) ** 2 + by ** 2),
    )
    lon = lon1 + math.atan2(by, math.cos(lat1) + bx)
    return math.degrees(lat), (math.degrees(lon) + 540) % 360 - 180


def route_map(rows, selected_id, height=300):
    """rows: dicts with Shipment_ID, Origin_Port, Destination_Port, p_failure, label."""
    fig = go.Figure()
    ports, mids = {}, []

    # The selected route is drawn last so it sits on top.
    for row in sorted(rows, key=lambda r: r["Shipment_ID"] == selected_id):
        start, end = PLACES.get(row["Origin_Port"]), PLACES.get(row["Destination_Port"])
        if start is None or end is None:
            continue
        chosen = row["Shipment_ID"] == selected_id
        color = t.risk_color(row["p_failure"])
        fig.add_trace(go.Scattergeo(
            lat=[start[0], end[0]], lon=[start[1], end[1]], mode="lines",
            line=dict(width=3 if chosen else 1.2, color=color),
            opacity=1 if chosen else .5, hoverinfo="skip",
        ))
        ports[row["Origin_Port"]] = start
        ports[row["Destination_Port"]] = end
        mids.append((row, _midpoint(start, end), color, chosen))

    fig.add_trace(go.Scattergeo(
        lat=[p[0] for p in ports.values()], lon=[p[1] for p in ports.values()],
        text=list(ports), mode="markers+text",
        textposition=[LABEL_SIDE.get(name, "top center") for name in ports],
        textfont=dict(family=t.FONT, size=11, color=t.TEXT_2),
        marker=dict(size=6, color=t.SURFACE, line=dict(width=1.5, color=t.TEXT_2)),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[m[1][0] for m in mids], lon=[m[1][1] for m in mids],
        customdata=[m[0]["Shipment_ID"] for m in mids],
        text=[m[0]["label"] for m in mids],
        mode="markers",
        marker=dict(
            size=[18 if m[3] else 12 for m in mids],
            color=[m[2] for m in mids], symbol="square",
            line=dict(width=[2 if m[3] else 1 for m in mids], color=t.TEXT),
        ),
        hovertemplate="%{text}<extra></extra>",
    ))

    # The view covers Europe to East Asia, where most routes run. It widens
    # only when the selected route has an end outside that window.
    lon_range, lat_range = [-15, 148], [-12, 63]
    for row, _, _, chosen in mids:
        ends = [PLACES[row["Origin_Port"]][1], PLACES[row["Destination_Port"]][1]]
        if chosen and not all(lon_range[0] <= lon <= lon_range[1] for lon in ends):
            lon_range, lat_range = [-130, 148], [-45, 75]

    fig.update_layout(**_base(height), dragmode=False)
    fig.update_geos(
        projection_type="equirectangular", bgcolor="rgba(0,0,0,0)",
        showland=True, landcolor="#D3E2E1", showocean=True, oceancolor="#F3F8F8",
        showcountries=True, countrycolor="#FFFFFF", coastlinecolor="#B4C6C5",
        showframe=True, framecolor=t.LINE_STRONG, showlakes=False,
        lonaxis_range=lon_range, lataxis_range=lat_range,
    )
    return fig


def thermal_chart(buffer_hours, hours_to_go, after=None, height=250):
    """Buffer hours left against hours still to go. No temperature: the data has none.

    after: optional dict(buffer_hours, hours_to_go, title) for a tried action.
    """
    span = max(hours_to_go, (after or {}).get("hours_to_go") or 0, 1.0) * 1.35
    top = max(buffer_hours, (after or {}).get("buffer_hours") or 0, hours_to_go, 1.0) * 1.15
    short = buffer_hours < hours_to_go
    fig = go.Figure()

    def depletion(buffer, color, name, dash=None, width=2.5):
        xs = [0, min(buffer, span), span] if buffer < span else [0, span]
        ys = [buffer, max(buffer - span, 0), max(buffer - span, 0)] if buffer < span else [buffer, buffer - span]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=name + "<br>%{y:.1f} h of buffer left at hour %{x:.1f}<extra></extra>",
        ))

    def gap(start, end, label, opacity):
        fig.add_vrect(
            x0=start, x1=end, fillcolor=t.RED, opacity=opacity, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(family=t.MONO, size=11, color=t.RED),
        )

    if short and not after:
        gap(buffer_hours, hours_to_go, f"Unprotected {hours_to_go - buffer_hours:.1f} h", .16)
    elif short:
        gap(buffer_hours, hours_to_go, "", .06)
    if after and after["buffer_hours"] < after["hours_to_go"]:
        left = after["hours_to_go"] - after["buffer_hours"]
        gap(after["buffer_hours"], after["hours_to_go"], f"Still unprotected {left:.1f} h", .16)
    depletion(buffer_hours, t.RED if short else t.NEUTRAL, "Now",
              dash="dot" if after else None, width=2 if after else 2.5)
    fig.add_vline(
        x=hours_to_go, line=dict(color=t.TEXT_2, width=1, dash="dash"),
        annotation_text="Arrival", annotation_position="top right",
        annotation_font=dict(family=t.MONO, size=11, color=t.TEXT_2),
    )

    if after:
        depletion(after["buffer_hours"], t.TEAL, after["title"])
        if abs(after["hours_to_go"] - hours_to_go) > .05:
            fig.add_vline(
                x=after["hours_to_go"], line=dict(color=t.TEAL, width=1, dash="dash"),
                annotation_text="New arrival", annotation_position="bottom right",
                annotation_font=dict(family=t.MONO, size=11, color=t.TEAL),
            )

    fig.update_layout(**_base(height))
    fig.update_layout(margin=dict(l=8, r=8, t=24, b=8))
    fig.update_xaxes(title_text="Hours from now", range=[0, span], automargin=True, **AXIS)
    fig.update_yaxes(title_text="Buffer left (h)", range=[0, top], automargin=True, **AXIS)
    return fig


def cost_chart(options, probability, cargo, loss_share, height=280):
    """Side by side: what each action leaves at stake (expected loss) plus its fee."""
    rows = [o for o in options if o["available"] and o["action_id"] != "stock_diversion"]
    rows.sort(key=lambda o: o["risk_after"] * loss_share * cargo + o["action_cost_usd"], reverse=True)
    names = [o["title"] for o in rows]
    loss = [o["risk_after"] * loss_share * cargo for o in rows]
    fees = [o["action_cost_usd"] for o in rows]
    best = min(range(len(rows)), key=lambda i: loss[i] + fees[i]) if rows else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=loss, orientation="h", name="Expected loss",
        marker=dict(color=[t.TEAL if i == best else t.QUIET for i in range(len(rows))]),
        hovertemplate="%{y}<br>Expected loss $%{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=fees, orientation="h", name="Fee",
        marker=dict(color=t.TEXT),
        text=[f"${l + f:,.0f}" for l, f in zip(loss, fees)], textposition="outside",
        textfont=dict(family=t.MONO, size=11, color=t.TEXT_2), cliponaxis=False,
        hovertemplate="%{y}<br>Fee $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(height), barmode="stack", bargap=.45)
    fig.update_layout(margin=dict(l=8, r=72, t=8, b=8))
    fig.update_xaxes(title_text="Expected loss plus fee (USD)", tickprefix="$", automargin=True, **AXIS)
    fig.update_yaxes(automargin=True, ticksuffix="  ",
                     **{**AXIS, "tickfont": dict(family=t.FONT, size=12, color=t.TEXT_2)})
    return fig
