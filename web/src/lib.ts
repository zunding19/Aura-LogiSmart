import { useEffect, useState } from "react";

/* The API returns plain JSON shaped by the Python backend, so it is typed loosely. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Data = any;

const cache = new Map<string, Data>();

export async function getJson(url: string): Promise<Data> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export async function postJson(url: string, body: unknown): Promise<Data> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

export const SERVER_DOWN = "LogiSmart's server did not answer. Wait a moment, then try again.";

/** Fetch with a small cache. Old data stays on screen while the new answer loads. */
export function useApi(url: string | null, rev = 0) {
  const key = url ? `${url}#${rev}` : null;
  const [state, setState] = useState<{ data: Data; loading: boolean; error: string | null }>(() => ({
    data: key ? cache.get(key) ?? null : null,
    loading: !!key && !cache.has(key),
    error: null,
  }));

  useEffect(() => {
    if (!url || !key) return;
    if (cache.has(key)) {
      setState({ data: cache.get(key), loading: false, error: null });
      return;
    }
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    getJson(url)
      .then((data) => {
        cache.set(key, data);
        if (live) setState({ data, loading: false, error: null });
      })
      .catch(() => {
        if (live) setState({ data: null, loading: false, error: SERVER_DOWN });
      });
    return () => {
      live = false;
    };
  }, [url, key]);

  return state;
}

export const pct = (p: number | null | undefined) => (p == null ? "n/a" : `${Math.round(p * 100)}%`);

export function usd(value: number | null | undefined) {
  if (value == null) return "not estimated";
  const sign = value < 0 ? "-" : "";
  const v = Math.abs(value);
  if (v >= 1_000_000) return `${sign}$${(v / 1_000_000).toFixed(1)}M`;
  return `${sign}$${Math.round(v).toLocaleString("en-US")}`;
}

/** Thermal runway in words: spare or short. */
export function hours(value: number | null | undefined) {
  if (value == null) return "not a countdown";
  return value >= 0 ? `${value.toFixed(1)} h spare` : `${(-value).toFixed(1)} h short`;
}

/** A rounded figure for a sentence: "about $203,000". */
export function about(value: number) {
  const step = Math.abs(value) >= 10_000 ? 1_000 : 10;
  return `about ${usd(Math.round(value / step) * step)}`;
}

/** A number from the API, or plain words when it is missing. */
export function shown(value: unknown, digits: number, unit = "") {
  const n = Number(value);
  return value == null || !Number.isFinite(n) ? "Not entered" : `${n.toFixed(digits)}${unit}`;
}

/** The ranked actions in one recommendation: what to do first, what else, and why. */
// Short action names for headlines. The full titles stay in the Decide table.
const SHORT: Record<string, string> = {
  monitor: "Keep monitoring",
  cold_storage_transfer: "Cold storage",
  direct_flight: "Direct flight",
  refrigerant_refresh: "Re-icing",
  refrigerated_ground: "Refrigerated truck",
  stock_diversion: "Stock diversion",
};
const short = (o: Data): string => SHORT[o.action_id] ?? o.title;

export function verdict(actions: Data, thermal: Data): { head: string; text: string } {
  const options: Data[] = actions.options;
  const find = (id: string | null) => options.find((o) => o.action_id === id && o.available);
  const picked = find(actions.recommended_action_id);
  const best = picked?.payoff === "saves" ? picked : null;
  const cover = find(actions.thermal_action_id);
  const backup = find(actions.backup_action_id);
  const batch = (o: Data) => (o.cargo_protected_usd ? `the ${usd(o.cargo_protected_usd)} batch` : "the batch");
  const parts: string[] = [];
  let head = "Keep monitoring";

  if (best) {
    head = cover ? `${short(best)}, then ${short(cover).toLowerCase()}` : short(best);
    parts.push(
      `Risk ${pct(best.risk_before)} to ${pct(best.risk_after)}. Saves ${about(best.net_saving_usd)} after the ${usd(best.action_cost_usd)} fee.`,
    );
    if (cover) {
      const left = best.thermal_runway_hours_after;
      parts.push(
        `${left != null && left < 0 ? `Buffer still ${hours(left)}` : "Buffer short"}: ` +
        `add ${short(cover).toLowerCase()} (${usd(cover.action_cost_usd)}) to protect ${batch(cover)}.`,
      );
    } else if (best.covers_thermal_gap) {
      parts.push("The buffer then covers the trip.");
    }
  } else if (cover) {
    head = short(cover);
    parts.push(
      `Buffer ${hours(thermal.runway_hours)}. ${short(cover)} (${usd(cover.action_cost_usd)}) protects ${batch(cover)}.`,
      actions.flagged ? "No action cuts the risk enough to pay." : "Risk is below the alert line.",
    );
  } else {
    parts.push(actions.flagged ? "No action pays for itself." : "Risk is below the alert line.");
  }
  if (backup) parts.push(`Backup: ${short(backup).toLowerCase()}.`);
  return { head, text: parts.join(" ") };
}

export type Tone = "red" | "amber" | "teal" | "";

export const COLOR = {
  red: "var(--red)",
  amber: "var(--amber)",
  teal: "var(--teal)",
  ink: "var(--text)",
  soft: "#5f5045",
  quiet: "var(--quiet)",
};

export function riskColor(p: number | null | undefined) {
  if (p == null) return "var(--muted)";
  if (p >= 0.8) return COLOR.red;
  if (p >= 0.6) return COLOR.amber;
  if (p >= 0.3) return COLOR.soft;
  return COLOR.teal;
}

/** Fill for bars and map squares. High uses a yellower amber, so it stays apart from red on a projector. */
export function riskFill(p: number | null | undefined) {
  return p != null && p >= 0.6 && p < 0.8 ? "var(--amber-fill)" : riskColor(p);
}

export function riskLevel(p: number): { name: string; tone: Tone } {
  if (p >= 0.8) return { name: "Critical", tone: "red" };
  if (p >= 0.6) return { name: "High", tone: "amber" };
  if (p >= 0.3) return { name: "Medium", tone: "" };
  return { name: "Low", tone: "teal" };
}
