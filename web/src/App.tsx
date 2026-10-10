import { useEffect, useRef } from "react";
import { Toaster, toast } from "sonner";
import { pct, useApi } from "./lib";
import { ALERT_LINES, VIEW_NAME, useStore, type View } from "./store";
import { Banner, Chip } from "./ui";
import MapView from "./views/MapView";
import Inbox from "./views/Inbox";
import Guided from "./views/Guided";
import WhatIf from "./views/WhatIf";
import Evidence from "./views/Evidence";

const OPERATIONS: View[] = ["map", "inbox", "respond", "whatif"];

export default function App() {
  const { view, line, rev, selected, go, select, setLine } = useStore();
  const queue = useApi(`/api/queue?line=${line}`, rev);
  const meta = useApi("/api/meta");
  const data = queue.data;
  const open = (data?.alerts ?? []).filter((a: { kind: string }) => a.kind !== "Resolved");

  // Open on the most urgent shipment.
  useEffect(() => {
    if (!selected && data?.top?.length) select(data.top[0].Shipment_ID);
  }, [selected, data, select]);

  // Notification-driven: the top alert announces itself once, with a way in.
  // It stays quiet during a response, so it never covers the result of an action.
  const first = open[0];
  const announced = useRef("");
  useEffect(() => {
    if (view === "respond") {
      toast.dismiss("top-alert");
      return;
    }
    if (!first) return;
    const key = `${first.id}:${first.head}`;
    if (announced.current === key) return;
    announced.current = key;
    toast.error(`${first.id}: ${first.head}`, {
      id: "top-alert", // a stable id, so a re-run updates the toast and never doubles it
      description: first.route,
      duration: 8000,
      action: { label: "Respond", onClick: () => go("respond", first.id, "Detect") },
    });
  }, [first?.id, first?.head, view]); // eslint-disable-line react-hooks/exhaustive-deps

  const lineIndex = ALERT_LINES.indexOf(line);
  const row = data?.line;

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">LogiSmart</span>
        </div>
        <div className="eyebrow navlabel">Operations</div>
        {OPERATIONS.map((name) => (
          <button key={name} type="button" className="nav" aria-current={view === name ? "page" : undefined} onClick={() => go(name)}>
            {VIEW_NAME[name]}
            {name === "inbox" && open.length > 0 && <span className="count hot">{open.length}</span>}
          </button>
        ))}
        <div className="eyebrow navlabel">Proof</div>
        <button type="button" className="nav" aria-current={view === "evidence" ? "page" : undefined} onClick={() => go("evidence")}>
          {VIEW_NAME.evidence}
        </button>

        <div className="panel linebox" style={{ marginTop: 16, padding: "12px 16px 16px" }}>
          <div className="field" style={{ marginBottom: 8 }}>
            <label htmlFor="alert-line">
              <span className="eyebrow">Alert line</span>
              <output>{pct(line)}</output>
            </label>
            <input id="alert-line" type="range" min={0} max={ALERT_LINES.length - 1} step={1} value={lineIndex}
              aria-valuetext={pct(line)}
              onChange={(e) => setLine(ALERT_LINES[Number(e.target.value)])} />
          </div>
          {row && (
            <p className="note">
              <span className="num ink">{pct(row.precision)}</span> of alerts are real,{" "}
              <span className="num ink">{pct(row.recall)}</span> of disruptions caught on 1,000 test shipments.
            </p>
          )}
          <p className="fine">Lower line: more caught, more false alarms.</p>
        </div>
        <p className="fine" style={{ padding: "0 8px" }}>Shipments, cargo values and thermal buffers are simulated.</p>
      </aside>

      <main className="main">
        <div className="crumbs">
          <b>LogiSmart</b>
          <Chip tone="teal">Simulated shipments</Chip>
          <span className="sep">/</span>
          <b>{VIEW_NAME[view]}</b>
          {data && (
            <>
              <span className="push num">{data.count} pharma alerts at the {pct(line)} line</span>
              <span className="sep">/</span>
              <span className="num">{Math.round(row.false_alarms_per_1000)} false alarms per 1,000 tested</span>
            </>
          )}
        </div>

        {queue.error && <Banner tone="red" title="Cannot reach LogiSmart's server">{queue.error}</Banner>}
        {!data && queue.loading && <div className="skeleton" aria-busy="true" />}
        {data && view === "map" && <MapView queue={data} coordinates={meta.data?.coordinates ?? {}} />}
        {data && view === "inbox" && <Inbox queue={data} />}
        {data && view === "respond" && <Guided />}
        {data && view === "whatif" && meta.data && <WhatIf meta={meta.data} />}
        {data && view === "evidence" && <Evidence />}
      </main>
      <Toaster position="bottom-right" richColors closeButton toastOptions={{ style: { fontFamily: "var(--font)" } }} />
    </div>
  );
}
