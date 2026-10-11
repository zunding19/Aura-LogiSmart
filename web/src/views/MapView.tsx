import clsx from "clsx";
import RouteMap from "../RouteMap";
import { hours, pct, riskColor, riskFill, riskLevel, useApi, usd, verdict, type Data } from "../lib";
import { useStore } from "../store";
import NumberFlow from "@number-flow/react";
import { Banner, Blocks, Chip, KV, Meter } from "../ui";

function QueueRow({ row, rank, on, logged, best }: { row: Data; rank: number; on: boolean; logged: boolean; best: number }) {
  const { go, select } = useStore();
  const color = riskColor(row.p_failure);
  // Urgency is shown against the top row, so the first shipment reads 100.
  const share = best > 0 ? row.urgency_score / best : 0;
  const index = share >= 0.1 ? Math.round(share * 100) : Math.round(share * 1000) / 10;
  return (
    // A click anywhere on the row selects it on the map; the button opens the response.
    <div className={clsx("qrow", on && "on")} onClick={() => select(row.Shipment_ID)}>
      <div className="rank">{String(rank).padStart(2, "0")}</div>
      <div>
        <div className="id">{row.Shipment_ID}</div>
        <div className="route">
          {row.Origin_Port} to {row.Destination_Port} by {String(row.Transport_Mode).toLowerCase()}
        </div>
        <div className="meta">
          {row.Weather_Condition} · {usd(row.Cargo_Value_USD_Simulated)} ·{" "}
          {row.runway == null ? "powered container" : <span className={row.runway < 0 ? "red" : undefined}>runway {hours(row.runway)}</span>}
        </div>
        <div className="urgency">
          <span className="score">Urgency {index}</span>
          <span className="bar" aria-hidden="true"><i style={{ width: `${Math.max(share * 100, 2)}%` }} /></span>
        </div>
        {logged && <div className="chips"><Chip tone="teal">Action logged</Chip></div>}
      </div>
      <div className="risk" style={{ color }}>
        {pct(row.p_failure)}
        <div className="level">{riskLevel(row.p_failure).name} risk</div>
        <small><Blocks share={row.p_failure} color={riskFill(row.p_failure)} /></small>
      </div>
      <button type="button" className={clsx("btn", on && "primary")} aria-label={`Respond to ${row.Shipment_ID}`}
        onClick={(e) => { e.stopPropagation(); go("respond", row.Shipment_ID, "Detect"); }}>
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
      <KV style={{ marginTop: 16 }} items={[
        { label: "Trouble", value: data.failure_mode.mode, small: true },
        { label: "Thermal runway", value: hours(runway), small: true, color: (runway ?? 0) < 0 ? "var(--red)" : undefined },
        { label: "Recommended", value: advice.head, small: true },
      ]} />
    </section>
  );
}

/** The single most urgent shipment, stated as a sentence, with the way in. */
function Lead({ row }: { row: Data }) {
  const go = useStore((s) => s.go);
  const short = row.runway != null && row.runway < 0;
  return (
    <section className="lead" aria-label="Most urgent shipment">
      <div>
        <div className="eyebrow"><span className="live" aria-hidden="true" />Most urgent now</div>
        <div className="say">
          <span className="num">{row.Shipment_ID}</span>{" "}
          {short
            ? <>runs out of thermal buffer <em>{(-row.runway).toFixed(1)} hours before it lands</em></>
            : <>is at <em>{pct(row.p_failure)} disruption risk</em></>}
        </div>
        <div className="ctx">
          {row.Origin_Port} to {row.Destination_Port} by {String(row.Transport_Mode).toLowerCase()} ·{" "}
          {row.Weather_Condition} · {row.Hospital_Urgency_Tier_Simulated} · {pct(row.p_failure)} disruption risk
        </div>
      </div>
      <div className="stake">
        <div className="eyebrow" style={{ justifyContent: "flex-end" }}>Cargo at stake</div>
        <NumberFlow className="big" value={row.Cargo_Value_USD_Simulated}
          format={{ style: "currency", currency: "USD", notation: "compact", minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
      </div>
      <button type="button" className="btn primary" onClick={() => go("respond", row.Shipment_ID, "Detect")}>
        Respond now
      </button>
    </section>
  );
}

export default function MapView({ queue, coordinates }: { queue: Data; coordinates: Record<string, [number, number]> }) {
  const { selected, select } = useStore();
  const top: Data[] = queue.top;
  const stats = queue.stats;
  const critical = top.filter((row) => row.p_failure >= 0.8).length;
  const logged = new Set((queue.alerts as Data[]).filter((a) => a.logged).map((a) => a.id));

  return (
    <>
      <header className="hero">
      <h1 className="title">Which shipment needs you first</h1>
      <p className="subtitle">The ten most urgent pharmaceutical shipments.</p>
      </header>
      {top.length === 0 ? (
        <Banner title="No alerts at this line">Lower the alert line in the sidebar to see more shipments.</Banner>
      ) : (
        <>
        <Lead row={top[0]} />
        <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)" }}>
          <div>
            <section className="panel">
              <h2 className="h2">Routes at risk</h2>
              <p className="note">Red is critical, amber is high.</p>
              <RouteMap rows={top} coordinates={coordinates} selected={selected} onSelect={select} />
              <KV style={{ marginTop: 8 }} items={[
                { label: "Expected loss", value: usd(stats.expected_loss_top), sub: `top ${stats.shown}` },
                { label: "Buffer short", value: `${stats.buffer_short} of ${stats.shown}`, sub: "ends before arrival", color: stats.buffer_short ? "var(--red)" : undefined },
                { label: "SLA breaches", value: `${stats.sla_breaches} of ${stats.shown}`, sub: "payment on hold", color: stats.sla_breaches ? "var(--red)" : undefined },
              ]} />
            </section>
            {selected && <div style={{ marginTop: 16 }}><Brief id={selected} /></div>}
          </div>
          <section className="panel">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <h2 className="h2" style={{ margin: 0 }}>Triage queue</h2>
              <div className="chips">
                {critical > 0 && <Chip tone="red">{critical} critical</Chip>}
                {top.length - critical > 0 && <Chip tone="amber">{top.length - critical} high</Chip>}
              </div>
            </div>
            <p className="note" style={{ margin: "4px 0 12px" }}>
              Top {top.length} of {queue.count} flagged.
            </p>
            {top.map((row, i) => (
              <QueueRow key={row.Shipment_ID} row={row} rank={i + 1} on={row.Shipment_ID === selected}
                logged={logged.has(row.Shipment_ID)} best={top[0].urgency_score} />
            ))}
          </section>
        </div>
        </>
      )}
    </>
  );
}
