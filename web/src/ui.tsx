import type { CSSProperties, ReactNode } from "react";
import clsx from "clsx";
import { riskColor, type Tone } from "./lib";
import { useStore } from "./store";

export function Chip({ tone = "", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={clsx("chip", tone)}>
      <i />
      {children}
    </span>
  );
}

/** A row of small squares, lit up to the given share (0 to 1). */
export function Blocks({ share, n = 10, color = "var(--text)" }: { share: number | null; n?: number; color?: string }) {
  const lit = Math.round(Math.min(Math.max(share ?? 0, 0), 1) * n);
  return (
    <span className="blocks" aria-hidden="true">
      {Array.from({ length: n }, (_, i) => (
        <span key={i} style={i < lit ? { background: color } : undefined} />
      ))}
    </span>
  );
}

/** Risk bar. When `before` differs, the old value stays as a faint ghost behind the new one. */
export function Meter({ value, before }: { value: number; before?: number }) {
  const line = useStore((s) => s.line);
  const from = before ?? value;
  return (
    <>
      <div className="meter" role="img" aria-label={`Disruption risk ${Math.round(value * 100)} percent`}>
        {Math.abs(from - value) > 0.001 && (
          <div className="ghost" style={{ transform: `scaleX(${from})`, background: riskColor(from) }} />
        )}
        <div className="fill" style={{ transform: `scaleX(${value})`, background: riskColor(value) }} />
        <div className="cuts" />
        <div className="mark-line" style={{ left: `${line * 100}%` }} />
      </div>
      <div className="scale">
        <span>0%</span>
        <span>Alert line {Math.round(line * 100)}%</span>
        <span>100%</span>
      </div>
    </>
  );
}

export type KVItem = {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  color?: string;
  small?: boolean;
  labelInk?: boolean;
};

export function KV({ items, style }: { items: KVItem[]; style?: CSSProperties }) {
  return (
    <div className="kv" style={style}>
      {items.map((item) => (
        <div key={item.label}>
          <div className="eyebrow" style={item.labelInk ? { color: "var(--text)" } : undefined}>
            {item.label}
          </div>
          <div className={clsx("v", item.small && "small")} style={{ color: item.color }}>
            {item.value}
          </div>
          {item.sub != null && <div className="s">{item.sub}</div>}
        </div>
      ))}
    </div>
  );
}

export function Banner({ tone = "", title, children, style }: {
  tone?: Tone;
  title: ReactNode;
  children?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div className={clsx("banner", tone)} style={style} role={tone === "red" ? "alert" : undefined}>
      <b>{title}</b>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange, label, render }: {
  options: readonly T[];
  value: T;
  onChange: (value: T) => void;
  label: string;
  render?: (option: T, index: number) => ReactNode;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((option, index) => (
        <button key={option} type="button" aria-pressed={option === value} onClick={() => onChange(option)}>
          {render ? render(option, index) : option}
        </button>
      ))}
    </div>
  );
}

/** Shown instead of a crash when the backend cannot score the input. */
export function Problem({ result }: { result: { status?: string; reason?: string } | null }) {
  return (
    <Banner tone="red" title={result?.status ?? "Insufficient data for reliable prediction"}>
      {result?.reason}
    </Banner>
  );
}
