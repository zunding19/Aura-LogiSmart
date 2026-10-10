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

export const SERVER_DOWN = "Aura's server did not answer. Check that python server.py is running.";

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

export type Tone = "red" | "amber" | "teal" | "";

export const COLOR = {
  red: "var(--red)",
  amber: "var(--amber)",
  teal: "var(--teal)",
  ink: "var(--text)",
  soft: "#4b6468",
  quiet: "var(--quiet)",
};

export function riskColor(p: number | null | undefined) {
  if (p == null) return "var(--muted)";
  if (p >= 0.8) return COLOR.red;
  if (p >= 0.6) return COLOR.amber;
  if (p >= 0.3) return COLOR.soft;
  return COLOR.teal;
}

export function riskLevel(p: number): { name: string; tone: Tone } {
  if (p >= 0.8) return { name: "Critical", tone: "red" };
  if (p >= 0.6) return { name: "High", tone: "amber" };
  if (p >= 0.3) return { name: "Medium", tone: "" };
  return { name: "Low", tone: "teal" };
}
