import clsx from "clsx";
import RouteMap from "../RouteMap";
import { hours, pct, riskColor, riskFill, riskLevel, useApi, usd, verdict, type Data } from "../lib";
import { useStore } from "../store";
import { Banner, Blocks, Chip, KV, Meter } from "../ui";

function QueueRow({ row, rank, on, logged }: { row: Data; rank: number; on: boolean; logged: boolean }) {
  const go = useStore((s) => s.go);
  const color = riskColor(row.p_failure);
  return (
    <div className={clsx("qrow", on && "on")}>
      <div className="rank">{String(rank).padStart(2, "0")}</div>
      <div>
        <div className="id">{row.Shipment_ID}</div>
        <div className="route">
          {row.Origin_Port} to {row.Destination_Port} · {row.Transport_Mode} · {row.Weather_Condition}
        </div>
        <div className="meta">
          {usd(row.Cargo_Value_USD_Simulated)} · {row.Hospital_Urgency_Tier_Simulated} ·{" "}
          {row.runway == null ? "powered container" : <span className={row.runway < 0 ? "red" : undefined}>runway {hours(row.runway)}</span>}
        </div>
        {logged && <div className="chips"><Chip tone="teal">Action logged</Chip></div>}
      </div>
      <div className="risk" style={{ color }}>
        {pct(row.p_failure)}
        <div className="level">{riskLevel(row.p_failure).name}</div>
        <small><Blocks share={row.p_failure} color={riskFill(row.p_failure)} /></small>
      </div>
      <button type="button" className={clsx("btn", on && "primary")} aria-label={`Respond to ${row.Shipment_ID}`}
        onClick={() => go("respond", row.Shipment_ID, "Detect")}>
        Respond
      </button>
    </div>
  );
}

/** The selected shipment in one glance, under the map. */
function Brief({ id }: { id: string }) {
  const { line, rev, go } = useStore();
  const { data, loading } = useApi(`/api/shipment/${encodeURIComponent(id)}?line=${line}`, rev);
  if (!data) return loading ? <div className="skeleton" aria-busy="true" /> : null;
  if (data.status !== "ok") return null;
  const p = data.risk.p_failure;
  const runway = data.thermal.runway_hours;
  const advice = verdict(data.actions, data.thermal);
  return (
    <section className={clsx("panel", loading && "pending")} aria-label="Selected shipment">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "start" }}>
        <div>
          <div className="eyebrow">Selected on the map</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
            <span className="mid num">{id}</span>
            <span className="mid" style={{ color: riskColor(p) }}>{pct(p)}</span>
            <span className="note">
              {data.shipment.Origin_Port} to {data.shipment.Destination_Port} by {String(data.shipment.Transport_Mode).toLowerCase()}
            </span>
          </div>
        </div>
        <button type="button" className="btn primary" onClick={() => go("respond", id, "Detect")}>Respond</button>
      </div>
      <Meter value={p} />
      <p className="note" style={{ marginTop: 12 }}>{data.explanation.summary}</p>
      <KV style={{ marginTop: 16 }} items={[
        { label: "Kind of trouble", value: data.failure_mode.mode, small: true },
        { label: "Thermal runway", value: hours(runway), small: true, color: (runway ?? 0) < 0 ? "var(--red)" : undefined },
        { label: "Recommended", value: advice.head, small: true },
      ]} />
    </section>
  );
}

export default function MapView({ queue, coordinates }: { queue: Data; coordinates: Record<string, [number, number]> }) {
  const { selected, select } = useStore();
  const top: Data[] = queue.top;
  const stats = queue.stats;
  const logged = new Set((queue.alerts as Data[]).filter((a) => a.logged).map((a) => a.id));

  return (
    <>
      <h1 className="title">Which shipment needs you first</h1>
      <p className="subtitle">
        The ten most urgent pharmaceutical shipments, ranked by risk, cargo value, hospital urgency and hours of thermal
        buffer left. Pick one on the map or in the queue. Shipments are simulated.
      </p>
      {top.length === 0 ? (
        <Banner title="No alerts at this line">Lower the alert line in the sidebar to see more shipments.</Banner>
      ) : (
        <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)" }}>
          <div>
            <section className="panel">
              <h2 className="h2">Routes at risk</h2>
              <p className="note">Squares mark shipments. Red is critical (80% and above), amber is high. Click a square to select it.</p>
              <RouteMap rows={top} coordinates={coordinates} selected={selected} onSelect={select} />
              <KV style={{ marginTop: 8 }} items={[
                { label: `Expected loss, top ${stats.shown}`, value: usd(stats.expected_loss_top), sub: "if nothing is done" },
                { label: "Buffer short", value: `${stats.buffer_short} of ${stats.shown}`, sub: "ends before arrival", color: stats.buffer_short ? "var(--red)" : undefined },
                { label: "SLA breaches", value: `${stats.sla_breaches} of ${stats.shown}`, sub: "carrier payment on hold", color: stats.sla_breaches ? "var(--red)" : undefined },
              ]} />
              <p className="fine">A disruption is assumed to cost {pct(queue.loss_share)} of cargo value.</p>
            </section>
            {selected && <div style={{ marginTop: 16 }}><Brief id={selected} /></div>}
          </div>
          <section className="panel">
            <h2 className="h2">Triage queue</h2>
            <p className="note" style={{ marginBottom: 12 }}>Most urgent first.</p>
            {top.map((row, i) => (
              <QueueRow key={row.Shipment_ID} row={row} rank={i + 1} on={row.Shipment_ID === selected}
                logged={logged.has(row.Shipment_ID)} />
            ))}
          </section>
        </div>
      )}
    </>
  );
}
