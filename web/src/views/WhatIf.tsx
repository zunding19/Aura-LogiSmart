import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import NumberFlow from "@number-flow/react";
import { toast } from "sonner";
import { CostChart } from "../charts";
import { getJson, hours, pct, postJson, riskColor, riskFill, riskLevel, SERVER_DOWN, usd, verdict, type Data } from "../lib";
import { useStore } from "../store";
import { Banner, Blocks, Chip, KV, Meter, Problem, Segmented } from "../ui";

type Form = {
  origin: string; destination: string; mode: "Air" | "Road"; weather: string;
  autoRisk: boolean; risk: number; carrier: number; lead: number; done: number;
  buffer: number; value: number; tier: string;
  // Kept from a loaded shipment until the route or buffer is changed by hand.
  distance?: number | null; context?: string | null;
};

const START: Form = {
  origin: "Frankfurt", destination: "Kuala Lumpur", mode: "Air", weather: "Storm",
  autoRisk: true, risk: 5, carrier: 0.75, lead: 3, done: 60, buffer: 12, value: 850_000,
  tier: "Critical Emergency",
};

const clamp = (v: number, low: number, high: number) => Math.min(Math.max(v, low), high);

function Slider({ id, label, value, min, max, step, show, onChange }: {
  id: string; label: string; value: number; min: number; max: number; step: number;
  show: string; onChange: (value: number) => void;
}) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}<output>{show}</output></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export default function WhatIf({ meta }: { meta: Data }) {
  const { selected, line } = useStore();
  const [form, setForm] = useState<Form>(START);
  const [source, setSource] = useState<string | null>(null);
  const [quick, setQuick] = useState<number | null>(null);
  const [result, setResult] = useState<Data>(null);
  const [pending, setPending] = useState(true);
  const turn = useRef(0);
  const set = (patch: Partial<Form>) => setForm((f) => ({ ...f, ...patch }));

  const order = useMemo(() => {
    const o: Record<string, string | number> = {
      Shipment_ID: "WHAT-IF",
      Place_of_Origin: form.origin, Place_of_Destination: form.destination,
      Transport_Mode: form.mode, Weather: form.weather,
      Carrier_Reliability_Score: form.carrier,
      "Lead-Time in days": form.lead,
      Transit_Duration: Math.round(form.lead * 24 * form.done) / 100,
      Buffer_of_Package: form.buffer, Cargo_Value_USD: form.value,
      Hospital_Urgency_Tier: form.tier,
    };
    if (!form.autoRisk) o.Geopolitical_Risk_Score = form.risk;
    if (form.distance != null) o.Distance_km = form.distance;
    if (form.context) o.Context_Source = form.context;
    return o;
  }, [form]);
  const samePlace = form.origin === form.destination;

  // The gauge follows the slider at once (a fast risk-only call); the full
  // answer with actions and money lands a moment after the hand stops.
  useEffect(() => {
    if (samePlace) return;
    const mine = ++turn.current;
    setPending(true);
    const fast = setTimeout(() => {
      postJson("/api/quick", order)
        .then((r) => { if (mine === turn.current && r.status === "ok") setQuick(r.p_failure); })
        .catch(() => undefined);
    }, 60);
    const full = setTimeout(() => {
      postJson(`/api/assess?line=${line}`, order)
        .then((r) => {
          if (mine !== turn.current) return;
          setResult(r);
          setQuick(r.status === "ok" ? r.risk.p_failure : null);
          setPending(false);
        })
        .catch(() => {
          if (mine !== turn.current) return;
          setResult({ status: "Cannot reach LogiSmart's server", reason: SERVER_DOWN });
          setPending(false);
        });
    }, 350);
    return () => { clearTimeout(fast); clearTimeout(full); };
  }, [order, samePlace, line]);

  async function loadSelected() {
    if (!selected) return;
    try {
      const data = await getJson(`/api/shipment/${encodeURIComponent(selected)}?line=${line}`);
      if (data.status !== "ok") throw new Error();
      const ship = data.shipment;
      const planned = Math.max(ship.Lead_Time_Days * 24, 1);
      setSource(null);
      setForm({
        origin: ship.Origin_Port, destination: ship.Destination_Port,
        mode: ship.Transport_Mode === "Road" ? "Road" : "Air",
        weather: ship.Weather_Condition ?? "Clear", autoRisk: false,
        risk: clamp(ship.Geopolitical_Risk_Score, 0, 10),
        carrier: clamp(ship.Carrier_Reliability_Score, 0.3, 1),
        lead: clamp(ship.Lead_Time_Days, 0.5, 15),
        done: clamp((100 * (ship.Elapsed_Transit_Hours_Simulated ?? 0)) / planned, 0, 100),
        distance: ship.Distance_km ?? null, context: data.thermal.buffer_counts_down ? null : "powered container",
        buffer: clamp(ship.Remaining_Buffer_Hours_Simulated ?? 0, 0, 120),
        value: clamp(ship.Cargo_Value_USD_Simulated ?? 50_000, 50_000, 6_000_000),
        tier: meta.tiers.includes(ship.Hospital_Urgency_Tier_Simulated) ? ship.Hospital_Urgency_Tier_Simulated : meta.tiers[0],
      });
    } catch {
      toast.error("That shipment could not be loaded.");
    }
  }

  async function liveWeather() {
    try {
      const data = await getJson(`/api/weather?origin=${encodeURIComponent(form.origin)}&destination=${encodeURIComponent(form.destination)}`);
      if (!data.condition) throw new Error();
      set({ weather: data.condition });
      setSource(data.source);
    } catch {
      toast.error("Live weather is unavailable. Pick the weather by hand.");
    }
  }

  const ok = !samePlace && result?.status === "ok";
  const p: number | null = quick ?? (ok ? result.risk.p_failure : null);
  const level = p == null ? null : riskLevel(p);
  const best = ok ? (result.actions.options as Data[]).find((o) => o.action_id === result.actions.recommended_action_id) : null;
  const advice = ok ? verdict(result.actions, result.thermal) : null;
  const assumed: string[] = ok ? (Object.values(result.inferred ?? {}).filter(Boolean) as string[]) : [];

  return (
    <>
      <h1 className="title">What if the order changes?</h1>
      <p className="subtitle">
        Move a control and LogiSmart re-scores the order: risk, thermal runway, the action that pays and what each choice
        leaves at stake.
      </p>
      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.5fr)" }}>
        <section className="panel" aria-label="Order">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <h2 className="h2" style={{ margin: 0 }}>Order</h2>
            {selected && <button type="button" className="btn" onClick={loadSelected}>Load {selected}</button>}
          </div>
          <div className="two">
            <div className="field">
              <label htmlFor="wi-origin">Origin</label>
              <select id="wi-origin" value={form.origin} onChange={(e) => { set({ origin: e.target.value, distance: null }); setSource(null); }}>
                {(meta.places as string[]).map((place) => <option key={place}>{place}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="wi-destination">Destination</label>
              <select id="wi-destination" value={form.destination} onChange={(e) => { set({ destination: e.target.value, distance: null }); setSource(null); }}>
                {(meta.places as string[]).map((place) => <option key={place}>{place}</option>)}
              </select>
            </div>
          </div>
          <div className="field">
            <span className="label">Mode</span>
            <div><Segmented label="Transport mode" options={["Air", "Road"] as const} value={form.mode} onChange={(mode) => set({ mode, distance: null, context: null })} /></div>
          </div>
          <div className="field">
            <span className="label">Weather on the route</span>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Segmented label="Weather on the route" options={meta.weather as string[]} value={form.weather}
                onChange={(weather) => { set({ weather }); setSource(null); }} />
              <button type="button" className="btn" onClick={liveWeather}>Use live weather</button>
            </div>
            {source && <p className="fine" style={{ marginTop: 0 }}>{source}</p>}
          </div>
          <Slider id="wi-carrier" label="Carrier reliability (0 poor, 1 perfect)" value={form.carrier} min={0.3} max={1} step={0.05}
            show={form.carrier.toFixed(2)} onChange={(carrier) => set({ carrier })} />
          <label className="check">
            <input type="checkbox" checked={form.autoRisk} onChange={(e) => set({ autoRisk: e.target.checked })} />
            Route risk from the World Bank stability index
          </label>
          {!form.autoRisk && (
            <Slider id="wi-risk" label="Route risk (0 calm, 10 unstable)" value={form.risk} min={0} max={10} step={0.1}
              show={form.risk.toFixed(1)} onChange={(risk) => set({ risk })} />
          )}
          <Slider id="wi-lead" label="Planned lead time (days)" value={form.lead} min={0.5} max={15} step={0.5}
            show={`${form.lead.toFixed(1)} d`} onChange={(lead) => set({ lead })} />
          <Slider id="wi-done" label="Trip completed" value={form.done} min={0} max={100} step={5}
            show={`${Math.round(form.done)}%`} onChange={(done) => set({ done })} />
          <Slider id="wi-buffer" label="Thermal buffer left (hours)" value={form.buffer} min={0} max={120} step={1}
            show={`${form.buffer} h`} onChange={(buffer) => set({ buffer, context: null })} />
          <Slider id="wi-value" label="Cargo value (USD)" value={form.value} min={50_000} max={6_000_000} step={50_000}
            show={usd(form.value)} onChange={(value) => set({ value })} />
          <div className="field" style={{ marginBottom: 0 }}>
            <label htmlFor="wi-tier">Hospital urgency</label>
            <select id="wi-tier" value={form.tier} onChange={(e) => set({ tier: e.target.value })}>
              {(meta.tiers as string[]).map((tier) => <option key={tier}>{tier}</option>)}
            </select>
          </div>
        </section>

        <div>
          {samePlace && <Problem result={{ reason: "Origin and destination are the same place." }} />}
          {!samePlace && result && result.status !== "ok" && <Problem result={result} />}
          {!samePlace && !result && <div className="skeleton" aria-busy="true" />}
          {ok && p != null && level && (
            <>
              <section className="panel" aria-label="Scenario result" aria-busy={pending}>
                <div className="eyebrow">Disruption risk</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
                  <NumberFlow className="big" style={{ color: riskColor(p) }} value={Math.round(p * 100) / 100}
                    format={{ style: "percent", maximumFractionDigits: 0 }} />
                  <Chip tone={level.tone}>{level.name}</Chip>
                  <span className="note">{result.failure_mode.mode}</span>
                </div>
                <Meter value={p} />
                <div className={clsx(pending && "pending")}>
                  <KV style={{ marginTop: 16 }} items={[
                    {
                      label: "Thermal runway", value: hours(result.thermal.runway_hours),
                      sub: `${Number(result.thermal.hours_to_go ?? 0).toFixed(1)} h still to go`,
                      color: (result.thermal.runway_hours ?? 0) < 0 ? "var(--red)" : undefined,
                    },
                    { label: "Expected loss", value: usd(result.priority.expected_loss_usd), sub: "if nothing is done" },
                    {
                      label: "Recommended", value: best?.title ?? "Keep monitoring", small: true,
                      sub: best?.payoff === "saves" ? <span className="teal">net saving {usd(best.net_saving_usd)}</span>
                        : best?.payoff === "protects" ? <span className="teal">protects the batch</span> : "nothing pays more",
                    },
                  ]} />
                  {advice && (
                    <Banner tone="teal" title={`Recommended: ${advice.head}`} style={{ marginTop: 16 }}>{advice.text}</Banner>
                  )}
                </div>
              </section>

              <section className={clsx("panel", pending && "pending")}>
                <h2 className="h2">What each choice leaves at stake</h2>
                <p className="note">
                  Each bar is the expected loss left after the action; the dark tip is its fee. Teal is the recommended action.
                </p>
                <CostChart options={result.actions.options} cargo={form.value} lossShare={meta.loss_share}
                  best={result.actions.recommended_action_id} />
              </section>

              {result.sweep && (
                <section className={clsx("panel", pending && "pending")}>
                  <h2 className="h2" style={{ marginBottom: 12 }}>Same order in each kind of weather</h2>
                  <KV items={(result.sweep as Data[]).map((s) => ({
                    label: s.weather, labelInk: s.weather === form.weather, value: pct(s.risk), color: riskColor(s.risk),
                    sub: <Blocks share={s.risk} color={riskFill(s.risk)} />,
                  }))} />
                </section>
              )}
              {assumed.length > 0 && (
                <>
                  <p className="fine">What LogiSmart assumed:</p>
                  <ul className="fine" style={{ paddingLeft: 16, marginTop: 4 }}>
                    {assumed.map((text) => <li key={text}>{text}</li>)}
                  </ul>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
