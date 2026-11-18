"""Glassmorphism theme: global CSS, HTML card helpers and a matching Plotly template."""

import html

import plotly.graph_objects as go
import plotly.io as pio

ACCENT = "#7dd3fc"
PALETTE = ["#7dd3fc", "#c4b5fd", "#f9a8d4", "#86efac", "#fcd34d", "#fdba74"]
GOOD, WARN, BAD = "#86efac", "#fcd34d", "#fca5a5"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --glass-bg: rgba(255, 255, 255, 0.07);
  --glass-bg-strong: rgba(255, 255, 255, 0.11);
  --glass-border: rgba(255, 255, 255, 0.16);
  --glass-highlight: rgba(255, 255, 255, 0.28);
  --glass-shadow: 0 8px 32px rgba(4, 6, 30, 0.45);
  --text: #eef2ff;
  --muted: #a5b4fc;
  --accent: #7dd3fc;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}

/* Aurora background with floating blurred orbs */
.stApp {
  background:
    radial-gradient(1200px 700px at 8% 0%, #3b1d77 0%, transparent 60%),
    radial-gradient(900px 600px at 100% 20%, #0f4c75 0%, transparent 55%),
    radial-gradient(900px 700px at 50% 110%, #5b1d5e 0%, transparent 60%),
    linear-gradient(160deg, #0b0a24 0%, #120f35 45%, #0a1630 100%);
  background-attachment: fixed;
  color: var(--text);
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; border-radius: 50%; filter: blur(90px); opacity: .45;
  z-index: 0; pointer-events: none; animation: drift 22s ease-in-out infinite alternate;
}
.stApp::before { width: 420px; height: 420px; top: 12%; left: 58%; background: #6d28d9; }
.stApp::after  { width: 360px; height: 360px; bottom: 6%; left: 12%; background: #0ea5e9; animation-delay: -11s; }
@keyframes drift { from { transform: translate(0,0) scale(1); } to { transform: translate(-80px, 60px) scale(1.15); } }
@media (prefers-reduced-motion: reduce) { .stApp::before, .stApp::after { animation: none; } }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > .main, .block-container { position: relative; z-index: 1; }
.block-container { padding-top: 2rem; max-width: 1400px; }

h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.02em; }
p, li, label, span { color: var(--text); }

/* Glass sidebar */
[data-testid="stSidebar"] {
  background: rgba(15, 12, 45, 0.45) !important;
  backdrop-filter: blur(22px) saturate(160%);
  -webkit-backdrop-filter: blur(22px) saturate(160%);
  border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] * { color: var(--text); }

/* Containers created with key="glass_..." become glass panels */
[class*="st-key-glass"],
div[data-testid="stExpander"] details,
[data-testid="stMetric"] {
  background: var(--glass-bg);
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  border: 1px solid var(--glass-border) !important;
  border-radius: 20px !important;
  box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-highlight);
}
[data-testid="stMetric"] { padding: 14px 18px; }
[class*="st-key-glass"] { padding: 18px 20px; }

/* Tabs as a glass pill bar */
.stTabs [role="tablist"] {
  gap: 6px; padding: 6px; border-radius: 16px; border-bottom: 1px solid var(--glass-border) !important;
  background: var(--glass-bg); border: 1px solid var(--glass-border);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  overflow-x: auto; margin-bottom: 14px;
}
.stTabs [role="tab"] {
  border-radius: 12px !important; padding: 8px 16px !important; background: transparent; color: var(--muted);
  white-space: nowrap; transition: background .2s ease; border-bottom: none !important; box-shadow: none;
}
.stTabs [role="tab"]::after { display: none !important; }
.stTabs [role="tab"]:hover { background: rgba(255,255,255,.06); }
.stTabs [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(125,211,252,.28), rgba(196,181,253,.28)) !important;
  color: #fff !important; box-shadow: inset 0 1px 0 var(--glass-highlight) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 700; }
[data-testid="stMetricValue"] p { font-size: inherit !important; }

/* Inputs */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  background: rgba(255,255,255,.06) !important; border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important; color: var(--text) !important;
}
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid var(--glass-border); }

/* Custom HTML cards */
.glass {
  background: var(--glass-bg);
  backdrop-filter: blur(18px) saturate(150%); -webkit-backdrop-filter: blur(18px) saturate(150%);
  border: 1px solid var(--glass-border); border-radius: 22px;
  box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-highlight);
  padding: 20px 22px; margin-bottom: 14px; position: relative; overflow: hidden;
}
.glass::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(135deg, rgba(255,255,255,.14) 0%, transparent 40%);
}
.hero h1 { font-size: clamp(1.6rem, 3.2vw, 2.5rem); font-weight: 800; margin: 0 0 6px; line-height: 1.15;
  background: linear-gradient(90deg, #e0f2fe, #c4b5fd 50%, #f9a8d4); -webkit-background-clip: text;
  background-clip: text; color: transparent !important; }
.hero p { color: var(--muted); margin: 0; font-size: 1rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 6px 0 18px; }
.kpi { margin: 0; }
.kpi .label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.kpi .value { font-size: 2rem; font-weight: 800; margin: 4px 0 2px; color: #fff; font-variant-numeric: tabular-nums; }
.kpi .sub { font-size: .82rem; color: var(--muted); }
.kpi .bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,.08); margin-top: 10px; overflow: hidden; }
.kpi .bar > span { display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, #7dd3fc, #c4b5fd); }
.section-title { font-size: 1.15rem; font-weight: 700; margin: 4px 0 10px; color: #fff; }
.muted { color: var(--muted); font-size: .9rem; }
.chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
  margin: 2px 4px 2px 0; border: 1px solid var(--glass-border); background: rgba(255,255,255,.07); }
.chip.good { color: #86efac; border-color: rgba(134,239,172,.4); }
.chip.warn { color: #fcd34d; border-color: rgba(252,211,77,.4); }
.chip.bad  { color: #fca5a5; border-color: rgba(252,165,165,.4); }
.alert { border-left: 4px solid; padding: 12px 16px; }
.alert.warn { border-left-color: #fcd34d; }
.alert.bad  { border-left-color: #fca5a5; }
.alert.good { border-left-color: #86efac; }
.alert b { color: #fff; }
.reco-row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.reco-rank { font-size: 1.4rem; font-weight: 800; width: 44px; height: 44px; border-radius: 14px; display: grid;
  place-items: center; background: linear-gradient(135deg, rgba(125,211,252,.35), rgba(196,181,253,.35)); color: #fff; }
.reco-main { flex: 1 1 240px; min-width: 0; }
.reco-main .name { font-weight: 700; color: #fff; }
.reco-stats { display: flex; gap: 18px; flex-wrap: wrap; }
.reco-stats div { font-size: .8rem; color: var(--muted); }
.reco-stats b { display: block; font-size: 1.05rem; color: #fff; font-variant-numeric: tabular-nums; }
</style>
"""


def esc(value) -> str:
    return html.escape(str(value))


def money(value: float) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def hero(title: str, subtitle: str) -> str:
    return f'<div class="glass hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'


def kpi_card(label: str, value: str, sub: str, fill_pct: float | None = None) -> str:
    bar = ""
    if fill_pct is not None:
        bar = f'<div class="bar"><span style="width:{max(0.0, min(100.0, fill_pct)):.0f}%"></span></div>'
    return (f'<div class="glass kpi"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div>{bar}</div>')


def kpi_grid(cards: list[str]) -> str:
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def alert(title: str, body: str, kind: str = "warn") -> str:
    return f'<div class="glass alert {kind}"><b>{esc(title)}</b><br><span class="muted">{esc(body)}</span></div>'


def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#e0e7ff", size=13),
        colorway=PALETTE,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)", linecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)", linecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="rgba(20,16,60,0.92)", bordercolor="rgba(255,255,255,0.2)", font=dict(color="#fff")),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    pio.templates["glass"] = template
    pio.templates.default = "plotly_dark+glass"


_register_plotly_template()
