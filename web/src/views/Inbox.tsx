import { useState } from "react";
import clsx from "clsx";
import type { Data } from "../lib";
import { useStore } from "../store";
import { Banner, Chip, Segmented } from "../ui";

export default function Inbox({ queue }: { queue: Data }) {
  const go = useStore((s) => s.go);
  const alerts: Data[] = queue.alerts;
  const kinds = ["All", ...Array.from(new Set(alerts.map((a) => a.kind as string))).sort()];
  const [pick, setPick] = useState("All");
  const filter = kinds.includes(pick) ? pick : "All";
  const shown = alerts.filter((a) => filter === "All" || a.kind === filter);

  return (
    <>
      <h1 className="title">Alerts that need a decision</h1>
      <p className="subtitle">
        Each alert says what went wrong and how long you have. Start a response to see the cause and the action that pays.
      </p>
      <div style={{ marginBottom: 16 }}>
        <Segmented label="Filter alerts" options={kinds} value={filter} onChange={setPick} />
      </div>
      {shown.length === 0 ? (
        <Banner title="Nothing here">No alerts of this kind right now.</Banner>
      ) : (
        <section className="panel">
          {shown.map((alert, index) => {
            const settled = alert.logged || alert.kind === "Resolved";
            return (
              <div className="alert" key={alert.id}>
                <div>
                  <div className="chips">
                    <Chip tone={alert.tone}>{alert.kind}</Chip>
                    <span className="num soft" style={{ fontSize: 13 }}>{alert.id}</span>
                    <span className="note">{alert.route}</span>
                    {alert.logged && <Chip tone="teal">Action logged</Chip>}
                  </div>
                  <div className="head">{alert.head}</div>
                  <div className="body">{alert.body}</div>
                </div>
                <div className="when">{Number(alert.hours_to_go).toFixed(1)} h to arrival</div>
                <button type="button" className={clsx("btn", index === 0 && !settled && "primary")}
                  onClick={() => go("respond", alert.id, "Detect")}>
                  {settled ? "Review" : "Start response"}
                </button>
              </div>
            );
          })}
        </section>
      )}
      <p className="fine">SLA notices and escrow are a simulation. No notice is sent and no payment is held.</p>
    </>
  );
}
