import { pct, useApi, type Data } from "../lib";
import { useStore } from "../store";
import { Banner, Blocks, KV } from "../ui";

export default function Evidence() {
  const { line, rev } = useStore();
  const { data, loading, error } = useApi(`/api/evidence?line=${line}`, rev);
  if (error) return <Banner tone="red" title="Cannot reach LogiSmart's server">{error}</Banner>;
  if (!data) return loading ? <div className="skeleton" aria-busy="true" /> : null;

  const row = data.line;
  const alerts = Math.round(row.alerts_per_1000);
  const falseAlarms = Math.round(row.false_alarms_per_1000);
  const missed = Math.round(row.missed_per_1000);
  const caught = alerts - falseAlarms;
  const quiet = 1000 - alerts - missed;
  const flagAll = 1000 - caught - missed; // false alarms if every shipment were flagged
  const groups = [
    { name: "Disruptions caught", count: caught, color: "var(--teal)" },
    { name: "False alarms", count: falseAlarms, color: "var(--red)" },
    { name: "Disruptions missed", count: missed, color: "var(--amber)" },
    { name: "Correctly left alone", count: quiet, color: "#cfdcdb" },
  ];
  const cells = groups.flatMap((group) => Array.from({ length: group.count }, () => group.color));
  const mix: Data[] = data.action_mix;
  const mixTotal = mix.reduce((sum, item) => sum + item.count, 0);
  const stages = [
    { verb: "Detect", fig: Number(data.total).toLocaleString("en-US"), text: "simulated shipments scored from 0 to 100% disruption risk." },
    { verb: "Categorize", fig: Number(data.pharma).toLocaleString("en-US"), text: "pharmaceutical shipments, each tagged as delay, cold-chain excursion or both." },
    { verb: "Prioritize", fig: data.queue_size, text: `air and road alerts at the ${pct(line)} line, ranked. ${data.top_id ?? "None"} is first.` },
    { verb: "Explain", fig: 3, text: "top drivers per shipment, in plain words with a what-if for each." },
    { verb: "Decide", fig: data.actions, text: "costed actions. The one with the highest net saving is recommended." },
  ];

  return (
    <>
      <h1 className="title">How LogiSmart decides, and how often it is right</h1>
      <p className="subtitle">
        Every figure here comes from the trained model and 1,000 held-out simulated shipments. None of it is real
        shipment data.
      </p>
      <div className="stage">
        {stages.map((stage, i) => (
          <div key={stage.verb}>
            <div className="n">{String(i + 1).padStart(2, "0")}</div>
            <div className="verb">{stage.verb}</div>
            <div className="fig">{stage.fig}</div>
            <p>{stage.text}</p>
          </div>
        ))}
      </div>

      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1.2fr) minmax(0, 1fr)", marginTop: 24 }}>
        <section className="panel">
          <h2 className="h2">1,000 test shipments at the {pct(line)} alert line</h2>
          <p className="note" style={{ marginBottom: 16 }}>
            One square is one shipment the model never saw in training. Move the alert line in the sidebar to see the trade.
          </p>
          <div className="waffle" role="img"
            aria-label={`${caught} disruptions caught, ${falseAlarms} false alarms, ${missed} missed, ${quiet} correctly left alone`}>
            {cells.map((color, i) => <span key={i} style={{ background: color }} />)}
          </div>
          <div className="legend">
            {groups.map((group) => (
              <div key={group.name} style={{ display: "contents" }}>
                <i style={{ background: group.color }} />
                <span>{group.name}</span>
                <b>{group.count}</b>
              </div>
            ))}
          </div>
        </section>

        <div>
          <section className="panel">
            <h2 className="h2">Fewer false alarms</h2>
            <p className="note" style={{ marginBottom: 16 }}>
              Against flagging every shipment, which is what a cautious team does today.
            </p>
            <div className="driver">
              <div className="top"><span>LogiSmart at {pct(line)}</span><span className="num">{falseAlarms}</span></div>
              <div style={{ marginTop: 8 }}><Blocks share={falseAlarms / flagAll} n={40} color="var(--red)" /></div>
            </div>
            <div className="driver">
              <div className="top"><span>Flag everything</span><span className="num">{flagAll}</span></div>
              <div style={{ marginTop: 8 }}><Blocks share={1} n={40} color="var(--quiet)" /></div>
            </div>
            <KV style={{ marginTop: 16 }} items={[
              { label: "Precision", value: pct(row.precision), sub: "of alerts are real" },
              { label: "Recall", value: pct(row.recall), sub: "of disruptions caught" },
              { label: "ROC AUC", value: Number(data.roc_auc).toFixed(2), sub: "ranking quality" },
            ]} />
            <p className="fine">At the default 60% line recall is about 7 in 10: it averages 69% over 30 random splits.</p>
          </section>
          <section className="panel">
            <h2 className="h2">A decision for every alert</h2>
            <p className="note" style={{ marginBottom: 16 }}>
              Best action across the {mixTotal} test alerts at the 60% line. LogiSmart says keep monitoring when no action pays.
            </p>
            {mix.map((item) => (
              <div className="driver" key={item.name}>
                <div className="top"><span>{item.name}</span><span className="num">{item.count}</span></div>
                <div style={{ marginTop: 8 }}>
                  <Blocks share={item.count / mixTotal} n={40} color={item.name === "Keep monitoring" ? "var(--quiet)" : "var(--teal)"} />
                </div>
              </div>
            ))}
          </section>
        </div>
      </div>
      <p className="fine">
        Assumptions: a disruption costs 10% of cargo value; cargo values, thermal buffers and urgency tiers are simulated;
        action fees are example figures.
      </p>
    </>
  );
}
