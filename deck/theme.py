"""Colours, type and motion for the command deck.

Motion values follow .github/skills (animate, emil-design-eng, apple-design):
strong ease-out curve, press feedback at scale(0.97) in 160ms, transform and
opacity only, hover gated to real pointers, and a reduced-motion fallback.
"""
import html as _html
import re

# Light, calm and medical: white surfaces on a soft teal tint, deep teal ink.
# Red is kept for shipments about to fail. Text colours pass WCAG AA on white.
BG = "#F3F8F8"
SURFACE = "#FFFFFF"
RAISED = "#EAF3F2"
LINE = "#DCE7E6"
LINE_STRONG = "#BFCFCE"
TEXT = "#0F2A2E"
TEXT_2 = "#3F5A5E"
MUTED = "#5B7377"
BLOCK_OFF = "#E2ECEB"
RED = "#D92D20"
AMBER = "#B25E09"
TEAL = "#0E7C73"
NEUTRAL = "#4B6468"
QUIET = "#A9BCBB"   # bars that are context, not the point

FONT = "'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'Geist Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"

LEVEL_COLOR = {"Critical": RED, "High": AMBER, "Medium": NEUTRAL, "Low": TEAL}


def risk_color(probability):
    if probability is None:
        return MUTED
    if probability >= .8:
        return RED
    if probability >= .6:
        return AMBER
    if probability >= .3:
        return NEUTRAL
    return TEAL


def esc(value):
    return _html.escape("" if value is None else str(value))


def flat(markup):
    """One line of HTML, so Markdown never reads an indented tag as a code block."""
    return re.sub(r"\n\s*", " ", markup.strip())


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');

:root {
  --bg: #F3F8F8; --surface: #FFFFFF; --raised: #EAF3F2;
  --line: #DCE7E6; --line-strong: #BFCFCE;
  --text: #0F2A2E; --text-2: #3F5A5E; --muted: #5B7377; --block-off: #E2ECEB;
  --red: #D92D20; --amber: #B25E09; --teal: #0E7C73; --teal-deep: #0A625B;
  --font: 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'Geist Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace;
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
}

html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp li, [data-testid="stMarkdownContainer"] {
  font-family: var(--font);
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton { display: none; }
.block-container { padding: 24px 32px 48px; max-width: 1480px; }
[data-testid="stSidebar"] { background: #E8F2F1; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 8px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 6px; }
[data-testid="stSidebarHeader"] { height: 24px; min-height: 24px; padding: 0; }
.stButton > button p { white-space: nowrap; }

/* Buttons: instant press feedback, colour change on hover, never a washed-out fade. */
.stButton > button, .stFormSubmitButton > button {
  border-radius: 8px; border: 1px solid var(--line-strong);
  background: var(--surface); color: var(--text); font-weight: 500;
  min-height: 36px; padding: 4px 8px;
  transition: transform 160ms var(--ease-out), background-color 150ms ease, border-color 150ms ease;
}
.stButton > button:active { transform: scale(0.97); }
.stButton > button:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--teal); color: #FFFFFF; border-color: var(--teal); }
.stButton > button[kind="primary"] p { color: #FFFFFF; font-weight: 600; }
.stButton > button:disabled { color: var(--muted); border-color: var(--line); background: var(--raised); }
.stButton > button[kind="primary"]:disabled p { color: var(--muted); }
@media (hover: hover) and (pointer: fine) {
  .stButton > button:hover { background: var(--raised); border-color: var(--teal); color: var(--text); }
  .stButton > button[kind="primary"]:hover { background: var(--teal-deep); border-color: var(--teal-deep); color: #FFFFFF; }
}

/* Sidebar navigation */
[data-testid="stSidebar"] .stButton > button {
  background: transparent; border-color: transparent; color: var(--text-2);
  justify-content: flex-start; text-align: left; min-height: 36px;
}
[data-testid="stSidebar"] .stButton > button > div { justify-content: flex-start; width: 100%; }
[data-testid="stSidebar"] .stButton > button p { color: var(--text-2); font-weight: 500; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: var(--surface); border-color: var(--line-strong); }
[data-testid="stSidebar"] .stButton > button[kind="primary"] p { color: var(--teal-deep); font-weight: 600; }
@media (hover: hover) and (pointer: fine) {
  [data-testid="stSidebar"] .stButton > button:hover { background: #DCEBEA; border-color: transparent; }
  [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: var(--surface); border-color: var(--teal); }
}

/* Panels are keyed containers: st.container(key="panel_...") */
[class*="st-key-panel"] {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 16px 20px 20px;
}
[class*="st-key-row_"] { border-top: 1px solid var(--line); padding-top: 12px; }

.brand { display: flex; align-items: center; gap: 8px; padding: 8px 4px 16px; }
.brand-mark { width: 16px; height: 16px; border-radius: 50%; border: 3px solid var(--teal); }
.brand-name { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.navlabel { padding: 12px 4px 20px; }
.crumbs { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2); padding-bottom: 16px; border-bottom: 1px solid var(--line); margin-bottom: 24px; flex-wrap: wrap; }
.crumbs b { color: var(--text); font-weight: 500; }
.crumbs .sep { color: var(--muted); }
.crumbs .push { margin-left: auto; }
.stApp h1.title { font-family: var(--font) !important; padding: 0; font-size: 30px; line-height: 1.1; letter-spacing: -0.02em; font-weight: 600; color: var(--text); margin: 0 0 8px; }
.stApp p.subtitle { font-size: 14px; line-height: 1.5; color: var(--text-2); margin: 0 0 24px; max-width: 760px; }
.h2 { font-size: 16px; line-height: 1.3; letter-spacing: -0.01em; font-weight: 600; color: var(--text); margin: 0 0 4px; }
.stApp p.note, .stApp .note { font-size: 13px; line-height: 1.5; color: var(--text-2); margin: 0; }
.stApp p.fine, .stApp ul.fine, .stApp ul.fine li, .stApp .fine { font-size: 12px; line-height: 1.5; color: var(--muted); margin: 8px 0 0; }
.stApp .num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.big { font-family: var(--font); font-size: 44px; line-height: 1; letter-spacing: -0.03em; font-weight: 600; font-variant-numeric: tabular-nums; }
.mid { font-size: 24px; line-height: 1.1; letter-spacing: -0.02em; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text); }

.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; line-height: 1; color: var(--text-2); border: 1px solid var(--line-strong); border-radius: 6px; padding: 4px 8px; white-space: nowrap; }
.chip i { width: 6px; height: 6px; background: currentColor; display: inline-block; }
.chip.red { color: var(--red); border-color: rgba(217, 45, 32, 0.4); }
.chip.amber { color: var(--amber); border-color: rgba(178, 94, 9, 0.4); }
.chip.teal { color: var(--teal); border-color: rgba(14, 124, 115, 0.4); }
.chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

/* Block meters, after the segmented bars in the reference boards */
.blocks { display: inline-flex; gap: 2px; vertical-align: middle; }
.blocks span { width: 6px; height: 12px; background: var(--block-off); border-radius: 1px; }
.blocks.tall span { width: 10px; height: 28px; border-radius: 2px; }

/* Queue rows enter once with a short stagger; they never block a click. */
.qrow { display: grid; grid-template-columns: 28px 1fr auto; gap: 12px; align-items: start;
  transition: opacity 180ms var(--ease-out), transform 180ms var(--ease-out);
  transition-delay: calc(var(--i, 0) * 30ms); }
@starting-style { .qrow { opacity: 0; transform: translateY(4px); } }
.qrow .rank { font-family: var(--mono); font-size: 12px; color: var(--muted); padding-top: 2px; }
.qrow .id { font-family: var(--mono); font-size: 14px; font-weight: 500; color: var(--text); }
.qrow .route { font-size: 13px; color: var(--text-2); margin-top: 2px; overflow-wrap: anywhere; }
.qrow .meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
.qrow .risk { text-align: right; font-family: var(--mono); font-size: 16px; font-weight: 500; }
.qrow .risk small { display: block; margin-top: 4px; }

/* Risk meter: the fill moves from the old value to the new one when an action is tried. */
.meter { position: relative; height: 28px; background: var(--block-off); border-radius: 2px; overflow: hidden; margin: 12px 0 8px; }
.meter .fill { position: absolute; inset: 0; transform-origin: left center; transform: scaleX(var(--v));
  transition: transform 480ms var(--ease-in-out), background-color 200ms ease; }
@starting-style { .meter .fill { transform: scaleX(var(--from)); } }
.meter .ghost { position: absolute; inset: 0; transform-origin: left center; opacity: 0.28; }
.meter .cuts { position: absolute; inset: 0; background-image: repeating-linear-gradient(90deg, transparent 0 10px, var(--surface) 10px 12px); }
.meter .line { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--text); }
.scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 11px; color: var(--muted); }

.kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 0; }
.kv > div { padding: 4px 16px 4px 0; border-left: 1px solid var(--line); padding-left: 16px; }
.kv > div:first-child { border-left: 0; padding-left: 0; }
.kv .v { font-size: 20px; letter-spacing: -0.02em; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; margin-top: 4px; }
.kv .s { font-size: 12px; color: var(--muted); margin-top: 2px; }

.banner { border: 1px solid var(--line-strong); border-radius: 12px; padding: 12px 16px; background: var(--surface); margin-bottom: 16px; }
.banner.red { border-color: rgba(217, 45, 32, 0.55); background: #FEF4F2; }
.banner.teal { border-color: rgba(14, 124, 115, 0.5); background: #EEF8F6; }
.banner b { color: var(--text); font-weight: 600; font-size: 14px; }
.stApp .banner p { font-size: 13px; color: var(--text-2); margin: 4px 0 0; line-height: 1.5; }

.driver { padding: 12px 0; border-top: 1px solid var(--line); }
.driver:first-child { border-top: 0; padding-top: 0; }
.driver .top { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; font-weight: 500; color: var(--text); }
.stApp .driver p { font-size: 13px; color: var(--text-2); margin: 4px 0 0; line-height: 1.5; }

.stage { display: grid; grid-template-columns: repeat(5, 1fr); border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.stage > div { padding: 16px 20px 20px; border-left: 1px solid var(--line); }
.stage > div:first-child { border-left: 0; }
.stage .n { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.08em; }
.stage .verb { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); margin-top: 8px; }
.stage .fig { font-size: 28px; font-weight: 600; letter-spacing: -0.02em; color: var(--text); margin-top: 12px; font-variant-numeric: tabular-nums; }
.stApp .stage p { font-size: 13px; color: var(--text-2); line-height: 1.5; margin: 4px 0 0; }
@media (max-width: 900px) { .stage { grid-template-columns: 1fr; } .stage > div { border-left: 0; border-top: 1px solid var(--line); } }

.waffle { display: grid; grid-template-columns: repeat(40, 1fr); gap: 2px; max-width: 520px; }
.waffle span { aspect-ratio: 1; border-radius: 1px; background: var(--block-off); }
.legend { display: grid; grid-template-columns: 12px 1fr auto; gap: 8px 12px; align-items: center; font-size: 13px; color: var(--text-2); }
.legend i { width: 10px; height: 10px; border-radius: 1px; display: block; }
.legend b { font-family: var(--mono); font-weight: 500; color: var(--text); text-align: right; }

.alert { display: grid; grid-template-columns: 1fr auto; gap: 8px 16px; }
.alert .head { font-size: 15px; font-weight: 600; color: var(--text); letter-spacing: -0.01em; }
.alert .body { font-size: 13px; color: var(--text-2); line-height: 1.5; margin-top: 4px; }
.alert .when { font-family: var(--mono); font-size: 12px; color: var(--muted); text-align: right; white-space: nowrap; }

.formula { font-family: var(--mono); font-size: 13px; color: var(--text-2); line-height: 1.8; overflow-wrap: anywhere; }
.formula b { color: var(--text); font-weight: 500; }

@media (prefers-reduced-motion: reduce) {
  .qrow { transition: opacity 160ms ease; transform: none !important; transition-delay: 0ms; }
  .meter .fill { transition: none; }
  .stButton > button, .stButton > button:active { transition: background-color 150ms ease; transform: none; }
}
"""


def blocks(share, n=10, color=TEXT, tall=False):
    """A row of small squares, lit up to the given share (0 to 1)."""
    share = 0 if share is None else min(max(float(share), 0), 1)
    lit = round(share * n)
    cells = "".join(
        f'<span style="background:{color}"></span>' if i < lit else "<span></span>"
        for i in range(n)
    )
    return f'<span class="blocks{" tall" if tall else ""}">{cells}</span>'


def chip(text, tone=""):
    return f'<span class="chip {tone}"><i></i>{esc(text)}</span>'


def usd(value):
    if value is None:
        return "not estimated"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f}M"
    return f"{sign}${value:,.0f}"


def hours(value):
    """Thermal runway in words: spare or short."""
    if value is None:
        return "not a countdown"
    return f"{value:.1f} h spare" if value >= 0 else f"{-value:.1f} h short"
