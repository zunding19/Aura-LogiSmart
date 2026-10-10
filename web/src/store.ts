import { create } from "zustand";

export const VIEWS = ["map", "inbox", "respond", "whatif", "evidence"] as const;
export type View = (typeof VIEWS)[number];
export const VIEW_NAME: Record<View, string> = {
  map: "Command map",
  inbox: "Alerts inbox",
  respond: "Guided response",
  whatif: "What-if simulator",
  evidence: "Evidence",
};
export const STEPS = ["Detect", "Categorize", "Prioritize", "Explain", "Decide"] as const;
export type Step = (typeof STEPS)[number];
export const ALERT_LINES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

type State = {
  view: View;
  selected: string | null;
  step: Step;
  line: number;
  sim: Record<string, string>; // shipment id -> action being tried
  rev: number; // bumped after a commit so alerts and SLA status reload
  go: (view: View, shipment?: string | null, step?: Step) => void;
  select: (shipment: string) => void;
  setStep: (step: Step) => void;
  setLine: (line: number) => void;
  tryAction: (shipment: string, action: string) => void;
  bump: () => void;
};

/* A link can open a screen directly, e.g. #/respond/SC-12132/Decide/direct_flight */
function fromHash(): Partial<State> {
  const [view, shipment, step, action] = decodeURIComponent(location.hash.replace(/^#\/?/, "")).split("/");
  const out: Partial<State> = {};
  if ((VIEWS as readonly string[]).includes(view)) out.view = view as View;
  if (shipment) out.selected = shipment;
  if ((STEPS as readonly string[]).includes(step)) out.step = step as Step;
  if (shipment && action) out.sim = { [shipment]: action };
  return out;
}

export const useStore = create<State>((set) => ({
  view: "map",
  selected: null,
  step: "Detect",
  line: 0.6,
  sim: {},
  rev: 0,
  ...fromHash(),
  go: (view, shipment, step) =>
    set((s) => ({ view, selected: shipment ?? s.selected, step: step ?? s.step })),
  select: (selected) => set({ selected }),
  setStep: (step) => set({ step }),
  setLine: (line) => set({ line }),
  tryAction: (shipment, action) => set((s) => ({ sim: { ...s.sim, [shipment]: action } })),
  bump: () => set((s) => ({ rev: s.rev + 1 })),
}));

// A new screen is a new history entry, so the browser's Back button stays inside the app.
useStore.subscribe((s, old) => {
  const parts: string[] = [s.view];
  if (s.view === "respond" && s.selected) parts.push(s.selected, s.step);
  const hash = `#/${parts.join("/")}`;
  if (location.hash === hash) return;
  if (s.view !== old.view) history.pushState(null, "", hash);
  else history.replaceState(null, "", hash);
});

window.addEventListener("popstate", () => useStore.setState(fromHash()));
