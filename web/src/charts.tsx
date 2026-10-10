import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Line, ReferenceArea,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { usd, type Data } from "./lib";

const INK = "#0f2a2e";
const SOFT = "#3f5a5e";
const MUTED = "#4f676b";
const GRID = "#dce7e6";
const RED = "#d92d20";
const TEAL = "#0e7c73";
const QUIET = "#a9bcbb";
const TICK = { fontFamily: "Geist Mono Variable, monospace", fontSize: 13, fill: MUTED };
const LABEL = { fontFamily: "Geist Mono Variable, monospace", fontSize: 13 };

function depletion(buffer: number, span: number) {
  return buffer < span
    ? [{ x: 0, y: buffer }, { x: buffer, y: 0 }, { x: span, y: 0 }]
    : [{ x: 0, y: buffer }, { x: span, y: buffer - span }];
}

/** Buffer hours left against hours still to go. No temperature: the data has none. */
export function ThermalChart({ buffer, toGo, after }: {
  buffer: number;
  toGo: number;
  after?: { buffer_hours: number; hours_to_go: number; title: string } | null;
}) {
  const span = Math.max(toGo, after?.hours_to_go ?? 0, 1) * 1.35;
  const top = Math.max(buffer, after?.buffer_hours ?? 0, toGo, 1) * 1.15;
  const short = buffer < toGo;
  const stillShort = after ? after.buffer_hours < after.hours_to_go : false;

  return (
    <ResponsiveContainer width="100%" height={210}>
      <ComposedChart margin={{ top: 20, right: 12, bottom: 20, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis type="number" dataKey="x" domain={[0, span]} tickCount={6} tick={TICK} stroke={GRID}
          tickFormatter={(v: number) => v.toFixed(0)} allowDuplicatedCategory={false}
          label={{ value: "Hours from now", position: "insideBottom", offset: -12, fill: MUTED, fontSize: 13 }} />
        <YAxis type="number" domain={[0, top]} tickCount={5} tick={TICK} stroke={GRID} width={36}
          tickFormatter={(v: number) => v.toFixed(0)} />
        {short && (
          <ReferenceArea x1={buffer} x2={toGo} fill={RED} fillOpacity={after ? 0.06 : 0.14} stroke="none"
            label={after ? undefined : { value: `Unprotected ${(toGo - buffer).toFixed(1)} h`, position: "insideTopLeft", fill: RED, ...LABEL }} />
        )}
        {after && stillShort && (
          <ReferenceArea x1={after.buffer_hours} x2={after.hours_to_go} fill={RED} fillOpacity={0.14} stroke="none"
            label={{ value: `Still unprotected ${(after.hours_to_go - after.buffer_hours).toFixed(1)} h`, position: "insideTopLeft", fill: RED, ...LABEL }} />
        )}
        <ReferenceLine x={toGo} stroke={SOFT} strokeDasharray="4 4"
          label={{ value: "Arrival", position: "top", fill: SOFT, ...LABEL }} />
        {after && Math.abs(after.hours_to_go - toGo) > 0.05 && (
          <ReferenceLine x={after.hours_to_go} stroke={TEAL} strokeDasharray="4 4"
            label={{ value: "New arrival", position: after.hours_to_go < span / 2 ? "insideBottomLeft" : "insideBottomRight", fill: TEAL, ...LABEL }} />
        )}
        <Line data={depletion(buffer, span)} dataKey="y" name="Now" type="linear" dot={false} isAnimationActive={false}
          stroke={short ? RED : SOFT} strokeWidth={after ? 2 : 2.5} strokeDasharray={after ? "2 4" : undefined} />
        {after && (
          <Line data={depletion(after.buffer_hours, span)} dataKey="y" name={after.title} type="linear" dot={false}
            isAnimationActive={false} stroke={TEAL} strokeWidth={2.5} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Side by side: what each action leaves at stake (expected loss) plus its fee. */
export function CostChart({ options, cargo, lossShare }: { options: Data[]; cargo: number; lossShare: number }) {
  const rows = options
    .filter((o) => o.available && o.action_id !== "stock_diversion")
    .map((o) => {
      const loss = o.risk_after * lossShare * cargo;
      return { name: o.title, loss, fee: o.action_cost_usd, total: loss + o.action_cost_usd };
    })
    .sort((a, b) => a.total - b.total);
  const best = rows[0]?.name;

  return (
    <ResponsiveContainer width="100%" height={Math.max(rows.length * 48 + 48, 160)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 72, bottom: 20, left: 4 }} barCategoryGap="32%">
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" tick={TICK} stroke={GRID} tickFormatter={(v: number) => (v >= 1000 ? `$${Math.round(v / 1000)}k` : `$${v}`)}
          label={{ value: "Expected loss plus fee (USD)", position: "insideBottom", offset: -12, fill: MUTED, fontSize: 13 }} />
        <YAxis type="category" dataKey="name" width={232} tick={{ fontSize: 13, fill: SOFT }} stroke={GRID} />
        <Tooltip cursor={{ fill: "#eaf3f2" }} formatter={(value) => usd(Number(value))}
          contentStyle={{ borderRadius: 8, border: `1px solid ${GRID}`, fontSize: 13 }} />
        <Bar dataKey="loss" name="Expected loss" stackId="cost" isAnimationActive={false}>
          {rows.map((row) => <Cell key={row.name} fill={row.name === best ? TEAL : QUIET} />)}
        </Bar>
        <Bar dataKey="fee" name="Fee" stackId="cost" fill={INK} isAnimationActive={false} minPointSize={1}>
          <LabelList dataKey="total" position="right" formatter={(value) => usd(Number(value))} style={{ ...LABEL, fill: SOFT }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
