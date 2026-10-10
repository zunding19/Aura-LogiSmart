import { useState, type ReactNode } from "react";
import clsx from "clsx";
import NumberFlow from "@number-flow/react";
import { toast } from "sonner";
import { ThermalChart } from "../charts";
import { hours, pct, postJson, riskColor, riskLevel, SERVER_DOWN, shown, useApi, usd, verdict, type Data, type Tone } from "../lib";
import { STEPS, useStore, type Step } from "../store";
import { Banner, Blocks, Chip, KV, Meter, Problem, Segmented } from "../ui";

// The question each step answers, used as its heading and on the button that leads to it.
const QUESTION: Record<Step, string> = {
  Detect: "Will it be disrupted?",
  Categorize: "What kind of trouble?",
  Prioritize: "Which first?",
  Explain: "Why?",
  Decide: "What to do?",
};

const CORRECTIVE = ["cold_storage_transfer", "refrigerant_refresh", "refrigerated_ground"];

function Detect({ data }: { data: Data }) {
  const line = useStore((s) => s.line);
  const { shipment: ship, risk, scorecard } = data;
  const flagged = risk.p_failure >= line;
  return (
    <>
      <h2 className="h2">{QUESTION.Detect}</h2>
      <p className="note">
        LogiSmart scores this shipment at <b className="ink">{pct(risk.p_failure)}</b>. The alert line is {pct(line)}, so it is{" "}
        {flagged ? "flagged" : "not flagged"}.
        {risk.model_variant !== "primary" && " The backup model was used because weather is missing."}
      </p>
      <div className="eyebrow" style={{ marginTop: 24 }}>What LogiSmart looked at</div>
      <KV style={{ marginTop: 8, rowGap: 16 }} items={[
        { label: "Weather", value: ship.Weather_Condition ?? "Not entered" },
        { label: "Route risk", value: shown(ship.Geopolitical_Risk_Score, 1, " / 10"), sub: "political stability" },
        { label: "Carrier", value: shown(ship.Carrier_Reliability_Score, 2, " / 1"), sub: "reliability" },
        { label: "Lead time", value: shown(ship.Lead_Time_Days, 1, " d") },
        { label: "Distance", value: ship.Distance_km == null ? "Not entered" : `${Math.round(ship.Distance_km).toLocaleString("en-US")} km` },
      ]} />
      <div className="eyebrow" style={{ marginTop: 24 }}>Can you trust the alert?</div>
      <p className="note" style={{ marginTop: 8 }}>
        On {Number(scorecard.tested_on_shipments).toLocaleString("en-US")} held-out test shipments at the {pct(line)} line,{" "}
        {pct(data.line.precision)} of alerts were real disruptions and LogiSmart caught about{" "}
        {Math.round(data.line.recall * 10)} in 10 of them.
      </p>
    </>
  );
}

function Categorize({ data }: { data: Data }) {
  const signs: Data[] = data.explanation.warning_signs;
  return (
    <>
      <h2 className="h2">{QUESTION.Categorize}</h2>
      <div className="mid" style={{ marginTop: 12 }}>{data.failure_mode.mode}</div>
      <p className="note" style={{ marginTop: 8 }}>{data.failure_mode.reason}</p>
      <div className="chips" style={{ marginTop: 16 }}>
        <Chip>Risk category: {data.explanation.category}</Chip>
      </div>
      <div className="eyebrow" style={{ margin: "24px 0 12px" }}>Warning signs</div>
      {signs.length === 0 && <p className="note">No warning sign is above its warning level.</p>}
      {signs.length > 1 && (
        <p className="note" style={{ marginBottom: 12 }}>
          {signs.length} warning signs are present, so the category is combined. The Explain step shows how much each one
          moves the risk.
        </p>
      )}
      {signs.map((sign) => (
        <div className="driver" key={sign.feature}>
          <div className="top">
            <span>{sign.category}</span>
            <Blocks share={sign.severity} color={sign.severity >= 0.8 ? "var(--red)" : "var(--amber)"} />
          </div>
          <p>{sign.description}</p>
        </div>
      ))}
    </>
  );
}

function bufferLeft(value: number | null | undefined, countsDown: boolean): { value: string; color?: string } {
  if (value == null) return { value: "Not entered" };
  if (countsDown && value < 0.05) return { value: "Used up", color: "var(--red)" };
  return { value: `${Number(value).toFixed(1)} h` };
}

// What an action wins, in the words of the Decide table.
function payoff(o: Data): { text: string; win: boolean } {
  switch (o.payoff) {
    case "saves": return { text: `Saves ${usd(o.net_saving_usd)}`, win: true };
    case "protects": return { text: `Protects ${usd(o.cargo_protected_usd)} batch`, win: true };
    case "backup": return { text: "Covers the hospital", win: false };
    case "none": return { text: "No change", win: false };
    default: return { text: `Costs ${usd(o.action_cost_usd)}, no gain`, win: false };
  }
}

function Prioritize({ data, id }: { data: Data; id: string }) {
  const { priority, shipment: ship, risk, thermal } = data;
  const rows: Data[] = data.neighbours;
  // The four things the ranking weighs, as this shipment has them.
  const reasons = [
    { label: "Disruption risk", value: pct(risk.p_failure), sub: riskLevel(risk.p_failure).name, color: riskColor(risk.p_failure) },
    { label: "Cargo at stake", value: usd(ship.Cargo_Value_USD_Simulated) },
    { label: "Hospital need", value: ship.Hospital_Urgency_Tier_Simulated ?? "Not entered", small: true },
    { label: thermal.buffer_counts_down ? "Buffer left" : "Hold if power fails", ...bufferLeft(thermal.buffer_hours, thermal.buffer_counts_down) },
  ];
  return (
    <>
      <h2 className="h2">{QUESTION.Prioritize}</h2>
      <div className="mid" style={{ marginTop: 12 }}>
        {data.rank ? `Rank ${data.rank} of ${data.queue_size}` : "Not in the queue"}
      </div>
      <p className="note" style={{ marginTop: 8 }}>
        {data.rank
          ? "A shipment moves up the queue when its risk is higher, its cargo is worth more, the hospital needs it sooner or its buffer is shorter."
          : data.rank_note}
      </p>
      <div className="eyebrow" style={{ marginTop: 24 }}>Why it ranks here</div>
      <KV style={{ marginTop: 8 }} items={reasons} />
      <p className="note" style={{ marginTop: 16 }}>
        Expected loss if nothing is done: <b className="ink">{usd(priority.expected_loss_usd)}</b>. A disruption is assumed to
        cost {pct(data.loss_share)} of cargo value.
      </p>
      {rows.length > 0 && (
        <>
          <div className="eyebrow" style={{ margin: "24px 0 8px" }}>Next to it in the queue</div>
          <table className="matrix">
            <thead>
              <tr>
                {["Rank", "Shipment", "Risk", "Cargo", "Hospital need", "Buffer"].map((head) => (
                  <th key={head} className="eyebrow" scope="col">{head}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className={clsx(row.id === id && "on")}>
                  <td className="num">{String(row.rank).padStart(2, "0")}</td>
                  <td className="num name">{row.id}</td>
                  <td className="num" style={{ color: riskColor(row.p_failure) }}>{pct(row.p_failure)}</td>
                  <td className="num">{usd(row.cargo)}</td>
                  <td>{row.tier}</td>
                  <td className="num">{bufferLeft(row.buffer_hours, row.counts_down).value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="fine">Buffer is hours of cooling left. In a powered container it is how long the cargo holds if the power fails.</p>
        </>
      )}
    </>
  );
}

function Explain({ data }: { data: Data }) {
  // Inputs that moved the score by under 2 points are noise, not drivers. The first always shows.
  const top: Data[] = data.explanation.drivers.slice(0, 3);
  const drivers = top.filter((driver, i) => i === 0 || Math.abs(driver.contribution) >= 0.02);
  const dropped = top.length - drivers.length;
  return (
    <>
      <h2 className="h2">{QUESTION.Explain}</h2>
      <p className="note" style={{ marginTop: 8 }}>{data.explanation.summary}</p>
      <div className="eyebrow" style={{ margin: "24px 0 12px" }}>{drivers.length === 1 ? "Main driver" : `Top ${drivers.length} drivers`}</div>
      {drivers.map((driver) => (
        <div className="driver" key={driver.feature}>
          <div className="top">
            <span>{driver.label}</span>
            <span className="num soft">
              {driver.contribution >= 0 ? "+" : ""}{Math.round(driver.contribution * 100)} points{" "}
              <Blocks share={Math.abs(driver.contribution)} color={driver.contribution >= 0.2 ? "var(--red)" : "var(--quiet)"} />
            </span>
          </div>
          <p>{driver.description}</p>
        </div>
      ))}
      {dropped > 0 && (
        <p className="fine">
          {dropped === 1 ? "One other input" : `${dropped} other inputs`} moved the score by under 2 points.
        </p>
      )}
    </>
  );
}

function Decide({ data, id, tried }: { data: Data; id: string; tried: string | undefined }) {
  const tryAction = useStore((s) => s.tryAction);
  const actions = data.actions;
  const advice = verdict(actions, data.thermal);
  return (
    <>
      <h2 className="h2">{QUESTION.Decide}</h2>
      <Banner tone="teal" title={`Recommended: ${advice.head}`} style={{ margin: "12px 0 16px" }}>{advice.text}</Banner>
      <p className="note" style={{ marginBottom: 12 }}>
        Actions are ranked: what pays most first, then what protects the batch. Try one to see its result on the right.
        Nothing is sent to anyone until you commit.
      </p>
      <table className="matrix">
        <thead>
          <tr>
            {["Rank", "Action", "Fee", "Risk after", "Runway after", "What it wins", ""].map((head) => (
              <th key={head} className="eyebrow" scope="col">{head}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(actions.options as Data[]).map((o) => {
            const aid = o.action_id;
            const backup = aid === "stock_diversion";
            const moved = o.risk_after < o.risk_before - 0.001;
            const runway = o.thermal_runway_hours_after;
            const wins = payoff(o);
            return (
              <tr key={aid} className={clsx(tried === aid && "on")}>
                <td className="num">{o.rank ?? ""}</td>
                <td>
                  <div className="name">{o.title}</div>
                  <div className="chips" style={{ marginTop: 4 }}>
                    {aid === actions.recommended_action_id && o.payoff !== "protects" && (
                      <Chip tone="teal">{aid === "monitor" ? "Nothing pays more" : "Best net saving"}</Chip>
                    )}
                    {aid === actions.thermal_action_id && <Chip tone="teal">Cheapest way to cover the buffer</Chip>}
                    {backup && <Chip tone={aid === actions.backup_action_id ? "amber" : ""}>Backup plan</Chip>}
                  </div>
                </td>
                <td className="num">{backup ? "n/a" : usd(o.action_cost_usd)}</td>
                {o.available ? (
                  <>
                    <td className="num" style={{ color: moved ? riskColor(o.risk_after) : "var(--text-2)" }}>{pct(o.risk_after)}</td>
                    <td className="num" style={{ color: o.covers_thermal_gap ? "var(--teal)" : (runway ?? 0) < 0 ? "var(--red)" : "var(--text-2)" }}>
                      {hours(runway)}
                    </td>
                    <td style={{ color: wins.win ? "var(--teal)" : "var(--text-2)", fontWeight: wins.win ? 500 : undefined }}>{wins.text}</td>
                    <td>
                      <button type="button" aria-pressed={tried === aid}
                        className={clsx("btn", (tried === aid || (!tried && o.rank === 1)) && "primary")}
                        onClick={() => tryAction(id, aid)}>
                        {tried === aid ? "Shown" : "Try"}
                      </button>
                    </td>
                  </>
                ) : (
                  <>
                    <td colSpan={3} className="fine">{o.unavailable_reason}</td>
                    <td><button type="button" className="btn" disabled>Air only</button></td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="fine">
        A disruption is assumed to cost {pct(data.loss_share)} of cargo value. Fees are example figures. Savings are the
        model's estimate, not a guarantee. Cargo value and buffer hours are simulated.
      </p>
    </>
  );
}

/** Right-hand column: the live numbers that react to a tried action. */
function Rail({ data, id, tried }: { data: Data; id: string; tried: string | undefined }) {
  const bump = useStore((s) => s.bump);
  const [busy, setBusy] = useState(false);
  const { risk, thermal, shipment: ship } = data;
  const before: number = risk.p_failure;
  const option: Data | undefined = tried
    ? (data.actions.options as Data[]).find((o) => o.action_id === tried && o.available)
    : undefined;
  const now: number = option ? option.risk_after : before;
  const runway: number | null = option ? option.thermal_runway_hours_after : thermal.runway_hours;
  const level = riskLevel(now);
  // Compared as shown on screen, so a move under one point does not count as a change.
  const changed = Math.round(now * 100) !== Math.round(before * 100);
  const after = option?.after ? { ...option.after, title: option.title } : null;
  const logged = !!tried && (data.committed as string[]).includes(tried);
  const corrective = !!tried && CORRECTIVE.includes(tried);

  async function commit() {
    if (!option || busy) return;
    setBusy(true);
    try {
      const result = await postJson("/api/commit", {
        shipment_id: id,
        action_id: tried,
        buffer_after: option.after?.buffer_hours ?? 0,
        hub: ship.Origin_Port ?? "Origin",
      });
      if (result.ok) {
        toast.success(`${id}: ${result.message}`, {
          description: corrective ? "The fix is on record. Any held carrier payment is released." : undefined,
        });
        bump();
      } else {
        toast.error(result.message ?? "The action was not logged.");
      }
    } catch {
      toast.error(SERVER_DOWN);
    } finally {
      setBusy(false);
    }
  }

  // A win leads the rail as one large figure. Anything else stays a quiet note.
  let win: { label: string; figure: ReactNode; text: string } | null = null;
  let outcome: { head: string; text: string; tone?: Tone } | null = null;
  if (option) {
    const fee = usd(option.action_cost_usd);
    if ((option.net_saving_usd ?? 0) > 0) {
      const cover = (data.actions.options as Data[]).find((o) => o.action_id === data.actions.thermal_action_id);
      win = {
        label: "Net saving",
        figure: (
          <NumberFlow value={Math.round(option.net_saving_usd)}
            format={{ style: "currency", currency: "USD", maximumFractionDigits: 0 }} />
        ),
        text: `Risk falls from ${pct(before)} to ${pct(now)}. Expected loss avoided, less the ${fee} fee.`
          + (option.covers_thermal_gap ? " The buffer now covers the trip too."
            : runway != null && runway < 0 ? ` The buffer is still ${hours(runway)}${cover ? `: add ${cover.title}.` : "."}` : ""),
      };
    } else if (option.covers_thermal_gap) {
      win = {
        label: "Batch protected",
        figure: hours(runway),
        text: `Thermal runway was ${hours(thermal.runway_hours)}. The ${usd(option.cargo_protected_usd)} batch is protected for a ${fee} fee. `
          + (changed ? `The risk score moves only from ${pct(before)} to ${pct(now)}.` : "The risk score does not move: more cooling does not stop a storm."),
      };
    } else if (tried === "stock_diversion") {
      outcome = { head: "Hospital covered", text: option.effect };
    } else if (tried === "monitor") {
      outcome = (thermal.runway_hours ?? 0) < 0
        ? { head: "No action taken. Batch still exposed", text: `The thermal buffer ends ${(-thermal.runway_hours).toFixed(1)} hours before arrival. ${option.effect}`, tone: "red" }
        : { head: "No action taken", text: option.effect };
    } else {
      outcome = { head: `Costs ${fee}, no risk saving`, text: option.effect };
    }
  }

  const commitButton = option && (
    <button type="button" className="btn primary wide" disabled={logged || busy} onClick={commit}>
      {logged ? "Logged" : corrective ? "Commit and log handshake" : "Commit action"}
    </button>
  );

  return (
    <section className="panel rail" aria-label="Live impact">
      {win && (
        <div className="outcome" aria-live="polite">
          <div className="eyebrow">{win.label}</div>
          <span className="big">{win.figure}</span>
          <p>{win.text}</p>
          {commitButton}
        </div>
      )}
      <div className="eyebrow">Disruption risk{option ? " after action" : ""}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginTop: 8, flexWrap: "wrap" }} aria-live="polite">
        <NumberFlow className={win ? "mid" : "big"} style={{ color: riskColor(now) }} value={Math.round(now * 100) / 100}
          format={{ style: "percent", maximumFractionDigits: 0 }} />
        <Chip tone={level.tone}>{level.name}</Chip>
        {option && changed && <span className="note">was {pct(before)}</span>}
      </div>
      <Meter value={now} before={before} />
      <KV style={{ marginTop: 16 }} items={[
        { label: "Cargo", value: usd(ship.Cargo_Value_USD_Simulated) },
        {
          label: "Thermal runway",
          value: hours(runway),
          color: runway == null ? undefined : runway < 0 ? "var(--red)" : "var(--teal)",
          sub: runway == null ? undefined : runway < 0 ? "Batch exposed" : "Batch protected",
        },
      ]} />
      {outcome && (
        <>
          <Banner tone={outcome.tone} title={outcome.head} style={{ margin: "16px 0" }}>{outcome.text}</Banner>
          {commitButton}
        </>
      )}
      {thermal.curve && thermal.buffer_hours != null ? (
        <>
          <div className="eyebrow" style={{ marginTop: 16 }}>Thermal runway</div>
          <ThermalChart buffer={thermal.buffer_hours} toGo={thermal.hours_to_go} after={after} />
          <p className="fine">Thermal runway is buffer hours minus hours still to go. It counts hours, not temperature.</p>
        </>
      ) : (
        thermal.note && <p className="fine">{thermal.note}</p>
      )}
    </section>
  );
}

export default function Guided() {
  const { selected: id, step, setStep, line, rev, sim, go } = useStore();
  const { data, loading, error } = useApi(id ? `/api/shipment/${encodeURIComponent(id)}?line=${line}` : null, rev);

  if (!id) return <Banner title="No shipment selected">Pick one from the command map.</Banner>;
  if (error) return <Banner tone="red" title="Cannot reach LogiSmart's server">{error}</Banner>;
  if (!data) return loading ? <div className="skeleton" aria-busy="true" /> : null;
  if (data.status !== "ok") {
    return (
      <>
        <h1 className="title num">{id}</h1>
        <Problem result={data} />
      </>
    );
  }

  const ship = data.shipment;
  const sla = data.sla;
  const tried = step === "Decide" ? sim[id] : undefined;
  const index = STEPS.indexOf(step);
  const stormy = ship.Weather_Condition === "Storm" || ship.Weather_Condition === "Hurricane";

  return (
    <>
      <div className="shiphead">
        <h1 className="title num">{id}</h1>
        <div className="chips">
          <Chip tone={stormy ? "red" : ""}>{ship.Weather_Condition ?? "Weather not entered"}</Chip>
          <Chip tone={ship.Hospital_Urgency_Tier_Simulated === "Critical Emergency" ? "red" : ""}>
            {ship.Hospital_Urgency_Tier_Simulated ?? "No urgency tier"}
          </Chip>
          <Chip>Storage {ship.Storage_Standard_Simulated ?? "not entered"}</Chip>
          {data.flag_land_route && <Chip tone="amber">Data check: land route crosses an ocean</Chip>}
        </div>
      </div>
      <p className="subtitle" style={{ marginBottom: 12 }}>
        {ship.Origin_Port} to {ship.Destination_Port} by {String(ship.Transport_Mode).toLowerCase()},{" "}
        {Number(data.thermal.hours_to_go ?? 0).toFixed(1)} hours still to go.
      </p>

      {sla?.trigger_condition && (
        <details className={clsx("banner", sla.breach_warning ? "red" : "teal")} style={{ marginBottom: 12 }}>
          <summary>
            <b>{sla.status}</b>
          </summary>
          <p>{sla.breach_notice ? sla.breach_notice.text : "A cooling fix that restores at least 3 hours of buffer has been logged."}</p>
        </details>
      )}

      <Segmented label="Response steps" options={STEPS} value={step} onChange={setStep}
        render={(name, i) => (<><span className="n">{i + 1}</span>{name}</>)} />

      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1.7fr) minmax(0, 1fr)", marginTop: 16 }}>
        <div>
          <section className="panel">
            {step === "Detect" && <Detect data={data} />}
            {step === "Categorize" && <Categorize data={data} />}
            {step === "Prioritize" && <Prioritize data={data} id={id} />}
            {step === "Explain" && <Explain data={data} />}
            {step === "Decide" && <Decide data={data} id={id} tried={tried} />}
          </section>
          <div className="stepnav">
            {index > 0 ? (
              <button type="button" className="btn" onClick={() => setStep(STEPS[index - 1])}>Back</button>
            ) : <span />}
            {index < STEPS.length - 1 ? (
              <button type="button" className="btn primary" onClick={() => setStep(STEPS[index + 1])}>
                Next: {QUESTION[STEPS[index + 1]]}
              </button>
            ) : (
              <button type="button" className="btn" onClick={() => go("inbox")}>Back to alerts</button>
            )}
          </div>
        </div>
        <Rail data={data} id={id} tried={tried} />
      </div>
    </>
  );
}
