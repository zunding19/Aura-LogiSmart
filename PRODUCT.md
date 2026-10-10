# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Python 3.13 and Streamlit, with pandas, numpy, scikit-learn and plotly. Layouts map to Streamlit primitives (`st.columns`, `st.metric`, `st.dataframe`, `st.plotly_chart`, `st.tabs`). No login, no database; synthetic in-memory data.

## Users

A healthcare logistics coordinator watching live pharmaceutical cold-chain shipments (air to hub to land to hospital). Today they learn a batch has spoiled only after it arrives. Their job is to see which shipment is about to fail, understand why, and act before the thermal buffer runs out.

## Product Purpose

Aura predicts the probability that a pharmaceutical shipment will fail (late arrival, temperature excursion, or unusable on arrival) hours before it happens, ranks shipments by urgency, explains the main drivers, and lets the coordinator test an intervention and see the new risk and the money saved. Success for the hackathon: a judge can answer yes to "Does your model help someone make a better decision, and can you prove it?"

## Positioning

Existing cold-chain monitoring reports a breach after it has happened. Aura forecasts it and attaches a costed decision to every prediction.

## Operating Context

- Built for Dormathon 2026, Track 1 (Predict Early, Decide Better). Shown as a 3-minute live demo, usually on a projector.
- Five-step flow: Detect, Categorize, Prioritize, Explain, Decide.
- Three screens: triage priority queue, single-shipment deep dive with a temperature forecast curve, decision matrix with live risk and savings.

## Capabilities and Constraints

- Failure probability 0 to 100% with tiers: Low 0-30, Medium 31-60, High 61-80, Critical above 80.
- Risk categories: Cold-chain excursion, Transport delay, Handover delay, Combined risk. Top 3 drivers only.
- Urgency = P(fail) x cargo value x urgency weight / max(buffer hours, 0.5).
- Interventions: prioritise cold-storage transfer, move to next direct flight, re-ice at hub, refrigerated ground transport, hospital stock diversion.
- Missing or invalid input never crashes; show "Insufficient data for reliable prediction".
- Field names follow the shared data contract in CLAUDE.md.

## Brand Commitments

- Name: Aura.
- Feel chosen by the team: calm, trustworthy, medical. Light, clean hospital feel with soft teal and white; red reserved for shipments about to fail.
- Copy is direct and functional. No buzzwords, no emojis in headings, sparing em dashes.
- Visual bans listed in CLAUDE.md are binding (no gradients, glassmorphism, Inter-everywhere, colored left-border cards, generic 4-card KPI dashboards).

## Evidence on Hand

- Data is synthetic. Do not present it as real shipment records.
- Model metrics (precision, recall, F1, confusion matrix) must come from the trained model, not from the planning document's placeholder figures.
- No logo or brand assets yet (`assets/` is empty).

## Product Principles

1. Never stop at a prediction; every risk shown comes with an action and its payoff.
2. Show only what needs attention now, ranked.
3. Explain in plain terms a coordinator can act on.
4. Prove it: real evaluation numbers, before and after comparison.

## Accessibility & Inclusion

WCAG AA contrast minimum. Must stay readable on a projector.
