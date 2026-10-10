import { useMemo, useState } from "react";
import { geoEquirectangular, geoInterpolate, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import clsx from "clsx";
import world from "world-atlas/countries-110m.json";
import { pct, riskColor, usd, type Data } from "./lib";

const WIDTH = 640;
const RAD = Math.PI / 180;
// Europe to East Asia, where most routes run. The view widens only when the
// selected route has an end outside this window.
const NEAR = { lon: [-15, 148], lat: [-12, 63] };
const WIDE = { lon: [-130, 148], lat: [-45, 75] };
// Ports that sit close together get their labels on different sides.
const LABEL: Record<string, [number, number, "start" | "middle" | "end"]> = {
  Rotterdam: [-8, 4, "end"],
  Antwerp: [0, 16, "middle"],
  Hamburg: [8, -6, "start"],
  Busan: [8, -6, "start"],
  Shanghai: [8, 14, "start"],
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const land = feature(world as any, (world as any).objects.countries);

export default function RouteMap({ rows, coordinates, selected, onSelect }: {
  rows: Data[];
  coordinates: Record<string, [number, number]>; // name -> [lat, lon]
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);

  const drawn = useMemo(() => {
    const point = (name: string): [number, number] | null =>
      coordinates[name] ? [coordinates[name][1], coordinates[name][0]] : null;
    const routes = rows
      .map((row) => ({ row, from: point(row.Origin_Port), to: point(row.Destination_Port) }))
      .filter((r) => r.from && r.to) as { row: Data; from: [number, number]; to: [number, number] }[];

    const chosen = routes.find((r) => r.row.Shipment_ID === selected);
    const inside = (lon: number) => lon >= NEAR.lon[0] && lon <= NEAR.lon[1];
    const view = chosen && !(inside(chosen.from[0]) && inside(chosen.to[0])) ? WIDE : NEAR;

    const scale = WIDTH / ((view.lon[1] - view.lon[0]) * RAD);
    const height = Math.round(scale * (view.lat[1] - view.lat[0]) * RAD);
    const projection = geoEquirectangular()
      .scale(scale)
      .center([(view.lon[0] + view.lon[1]) / 2, (view.lat[0] + view.lat[1]) / 2])
      .translate([WIDTH / 2, height / 2]);
    const path = geoPath(projection);

    const ports = new Map<string, [number, number]>();
    for (const r of routes) {
      ports.set(r.row.Origin_Port, r.from);
      ports.set(r.row.Destination_Port, r.to);
    }

    return {
      height,
      land: path(land) ?? "",
      // The selected route is drawn last so it sits on top.
      routes: [...routes]
        .sort((a, b) => Number(a.row.Shipment_ID === selected) - Number(b.row.Shipment_ID === selected))
        .map((r) => ({
          row: r.row,
          d: path({ type: "LineString", coordinates: [r.from, r.to] }) ?? "",
          mid: projection(geoInterpolate(r.from, r.to)(0.5)) ?? [0, 0],
        })),
      ports: [...ports].map(([name, lonLat]) => ({ name, xy: projection(lonLat) ?? [0, 0] })),
    };
  }, [rows, coordinates, selected]);

  const told = rows.find((r) => r.Shipment_ID === (hover ?? selected));

  return (
    <>
      <svg className="map" viewBox={`0 0 ${WIDTH} ${drawn.height}`} role="group" aria-label="Map of the ten most urgent shipment routes">
        <path className="land" d={drawn.land} />
        {drawn.routes.map(({ row, d }) => {
          const on = row.Shipment_ID === selected;
          return (
            <path key={row.Shipment_ID} className="route" d={d} stroke={riskColor(row.p_failure)}
              strokeWidth={on ? 3 : 1.2} opacity={on ? 1 : 0.5} />
          );
        })}
        {drawn.ports.map(({ name, xy }) => {
          const [dx, dy, anchor] = LABEL[name] ?? [0, -8, "middle"];
          return (
            <g key={name}>
              <circle className="port" cx={xy[0]} cy={xy[1]} r={3} />
              <text className="port-label" x={xy[0] + dx} y={xy[1] + dy} textAnchor={anchor}>{name}</text>
            </g>
          );
        })}
        {drawn.routes.map(({ row, mid }) => {
          const id = row.Shipment_ID;
          return (
            <g key={id} className={clsx("mark", id === selected && "on")} role="button" tabIndex={0}
              aria-label={`${id}, ${row.Origin_Port} to ${row.Destination_Port}, ${pct(row.p_failure)} risk`}
              aria-pressed={id === selected}
              onClick={() => onSelect(id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(id);
                }
              }}
              onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(id)} onBlur={() => setHover(null)}>
              {/* A larger invisible square makes the small mark easy to hit. */}
              <rect className="hit" x={mid[0] - 12} y={mid[1] - 12} width={24} height={24} />
              <rect className="dot" x={mid[0] - 6} y={mid[1] - 6} width={12} height={12} fill={riskColor(row.p_failure)} />
            </g>
          );
        })}
      </svg>
      <p className="map-tip">
        {told && (
          <>
            <span className="num ink">{told.Shipment_ID}</span> · {told.Origin_Port} to {told.Destination_Port} by{" "}
            {String(told.Transport_Mode).toLowerCase()} · {pct(told.p_failure)} risk · {usd(told.Cargo_Value_USD_Simulated)}
          </>
        )}
      </p>
    </>
  );
}
