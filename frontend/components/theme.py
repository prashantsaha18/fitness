"""
frontend/components/theme.py
─────────────────────────────
Single source of truth for the FitAI dark-fitness CSS theme.

Usage in any page:
    from frontend.components.theme import apply_theme
    apply_theme()          # call once at the top of every page

Design tokens match the CSS variables so Python code and HTML stay in sync.
"""
from __future__ import annotations

import streamlit as st

# ── Design tokens (mirrors :root CSS vars) ────────────────────────────────────
NEON_GREEN = "#39FF14"
ELECTRIC   = "#00D4FF"
FLAME      = "#FF6B35"
PURPLE     = "#8B5CF6"
AMBER      = "#F59E0B"
EMERALD    = "#10B981"
RED        = "#EF4444"
BLUE       = "#3B82F6"

DARK_BG    = "#0A0A0F"
CARD_BG    = "#12121A"
BORDER     = "#1E1E2E"
TEXT_MUTED = "#6B7280"
TEXT_DIM   = "#9CA3AF"
TEXT_MAIN  = "#F9FAFB"

WORKOUT_COLORS: dict[str, str] = {
    "HIIT":     FLAME,
    "Strength": PURPLE,
    "Yoga":     EMERALD,
    "Cardio":   BLUE,
    "Pilates":  AMBER,
    "Meal":     NEON_GREEN,
}

WORKOUT_ICONS: dict[str, str] = {
    "HIIT":     "🔥",
    "Strength": "💪",
    "Yoga":     "🧘",
    "Cardio":   "🏃",
    "Pilates":  "⚖️",
    "Meal":     "🥗",
}

# ── Zone mapping ──────────────────────────────────────────────────────────────
HR_ZONES: dict[str, tuple[int, int, str]] = {
    # zone_name → (hr_lo, hr_hi, color)
    "Resting":   (0,   100, TEXT_MUTED),
    "Fat Burn":  (100, 120, EMERALD),
    "Cardio":    (120, 150, BLUE),
    "Peak":      (150, 170, AMBER),
    "Anaerobic": (170, 999, RED),
}


def hr_zone(bpm: int) -> tuple[str, str]:
    """Return (zone_name, color) for a heart-rate reading."""
    for name, (lo, hi, color) in HR_ZONES.items():
        if lo <= bpm < hi:
            return name, color
    return "Anaerobic", RED


# ── Theme injector ────────────────────────────────────────────────────────────
_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --neon-green: #39FF14; --electric: #00D4FF; --flame: #FF6B35;
  --purple: #8B5CF6;     --amber: #F59E0B;    --emerald: #10B981;
  --dark-bg: #0A0A0F;    --card-bg: #12121A;  --border: #1E1E2E;
  --text-muted: #6B7280; --text-dim: #9CA3AF; --text-main: #F9FAFB;
}

/* ── Layout ──────────────────────────────────────────────────── */
.stApp                             { background: var(--dark-bg); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"]          { background: linear-gradient(180deg,#0D0D1A 0%,var(--dark-bg) 100%); border-right: 1px solid var(--border); }
#MainMenu, footer, header          { visibility: hidden; }
.block-container                   { padding-top: 1rem; max-width: 1400px; }
section[data-testid="stMain"]      { background: var(--dark-bg); }

/* ── Section headers ─────────────────────────────────────────── */
.section-header {
  font-size: 17px; font-weight: 700; color: var(--text-main);
  border-left: 3px solid var(--electric); padding-left: 12px;
  margin: 20px 0 14px;
}

/* ── Hero banner ─────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg,#0D0D2A 0%,#1A0A2E 50%,#0A1A0D 100%);
  border: 1px solid var(--border); border-radius: 20px;
  padding: 32px 40px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.hero::after {
  content: '⚡'; position: absolute; right: 40px; top: 50%;
  transform: translateY(-50%); font-size: 120px; opacity: 0.05; pointer-events: none;
}
.hero-title {
  font-size: 38px; font-weight: 900; line-height: 1.1;
  background: linear-gradient(90deg,#fff 0%,var(--electric) 50%,var(--neon-green) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}
.hero-sub { font-size: 15px; color: var(--text-muted); max-width: 600px; }

/* ── Metric cards ────────────────────────────────────────────── */
.metric-card {
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 14px; padding: 18px 22px; position: relative; overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #2E2E4E; }
.metric-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  border-radius: 14px 14px 0 0;
}
.metric-card.green::before  { background: linear-gradient(90deg,var(--neon-green),#00FF88); }
.metric-card.blue::before   { background: linear-gradient(90deg,var(--electric),#0088FF); }
.metric-card.orange::before { background: linear-gradient(90deg,var(--flame),#FFB347); }
.metric-card.purple::before { background: linear-gradient(90deg,var(--purple),#C084FC); }
.metric-card.amber::before  { background: linear-gradient(90deg,var(--amber),#FCD34D); }
.metric-label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 5px; }
.metric-value { font-size: 34px; font-weight: 900; line-height: 1; font-family: 'JetBrains Mono', monospace; color: var(--text-main); }
.metric-sub   { font-size: 12px; color: var(--text-muted); margin-top: 5px; }

/* ── Recommendation cards ────────────────────────────────────── */
.rec-card {
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 20px; margin-bottom: 10px;
  display: flex; align-items: center; gap: 14px;
  transition: all 0.2s; cursor: pointer;
}
.rec-card:hover { border-color: #3B3B5C; background: #161624; transform: translateX(3px); }
.rec-rank       { font-size: 13px; font-weight: 700; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; min-width: 24px; }
.rec-rank.top   { color: var(--neon-green); }
.rec-icon       { font-size: 28px; min-width: 38px; text-align: center; }
.rec-body       { flex: 1; }
.rec-title      { font-size: 14px; font-weight: 600; color: var(--text-main); margin-bottom: 3px; }
.rec-meta       { font-size: 12px; color: var(--text-muted); }
.rec-score      { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: var(--neon-green); min-width: 52px; text-align: right; }

/* ── Badges ──────────────────────────────────────────────────── */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }
.badge-hiit     { background:#FF6B3520; color:#FF6B35; border:1px solid #FF6B3540; }
.badge-strength { background:#8B5CF620; color:#A78BFA; border:1px solid #8B5CF640; }
.badge-yoga     { background:#10B98120; color:#34D399; border:1px solid #10B98140; }
.badge-cardio   { background:#3B82F620; color:#60A5FA; border:1px solid #3B82F640; }
.badge-pilates  { background:#F59E0B20; color:#FCD34D; border:1px solid #F59E0B40; }

/* ── Alert boxes ─────────────────────────────────────────────── */
.alert { border-radius: 10px; padding: 11px 15px; font-size: 13px; font-weight: 500; }
.alert-green  { background:#39FF1410; border:1px solid #39FF1430; color:var(--neon-green); }
.alert-blue   { background:#00D4FF10; border:1px solid #00D4FF30; color:var(--electric); }
.alert-orange { background:#FF6B3510; border:1px solid #FF6B3530; color:var(--flame); }
.alert-red    { background:#EF444410; border:1px solid #EF444430; color:#F87171; }

/* ── Progress bar ────────────────────────────────────────────── */
.prog-wrap { background:#1E1E2E; border-radius:8px; height:8px; overflow:hidden; margin-top:5px; }
.prog-bar  { height:100%; border-radius:8px; transition:width 0.5s ease; }

/* ── Streamlit overrides ─────────────────────────────────────── */
div[data-testid="stMetric"] {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 15px 18px;
}
div[data-testid="stMetric"] label              { color: var(--text-muted) !important; font-size: 12px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-main) !important; font-family: 'JetBrains Mono'; }

.stButton > button {
  background: linear-gradient(135deg,#1E1E3F,#2D2D5A); border: 1px solid #3B3B6B;
  color: var(--text-main); border-radius: 10px; font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg,#2D2D5A,#3B3B7A); border-color: var(--electric); color: var(--electric); transform: translateY(-1px); }

.stSelectbox > div > div  { background: var(--card-bg); border-color: var(--border); }
.stTextInput  > div > div { background: var(--card-bg); border-color: var(--border); }
.stSlider     > div > div > div { background: var(--electric); }
.stTabs [data-baseweb="tab"]             { background: var(--card-bg); color: var(--text-dim); border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"]           { background: #1E1E3F; color: var(--electric); border-bottom: 2px solid var(--electric); }
.stTabs [data-baseweb="tab-highlight"]   { background: var(--electric); }
"""

_PLOTLY_DEFAULTS = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font_color=TEXT_DIM,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(color=TEXT_DIM)),
    xaxis=dict(gridcolor=BORDER, color=TEXT_MUTED),
    yaxis=dict(gridcolor=BORDER, color=TEXT_MUTED),
)


def apply_theme() -> None:
    """Inject the shared FitAI CSS into the current Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def plotly_layout(**overrides) -> dict:
    """Return a Plotly layout dict pre-filled with dark-theme defaults."""
    layout = dict(_PLOTLY_DEFAULTS)
    layout.update(overrides)
    return layout


def hero(title: str, subtitle: str = "", extras_html: str = "") -> None:
    """Render the standard hero banner."""
    st.markdown(
        f"<div class='hero'>"
        f"<div class='hero-title'>{title}</div>"
        f"{'<div class=hero-sub>' + subtitle + '</div>' if subtitle else ''}"
        f"{extras_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    """Render a section header."""
    st.markdown(f"<div class='section-header'>{label}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", color: str = "blue") -> None:
    """Render a single metric card."""
    st.markdown(
        f"<div class='metric-card {color}'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{'<div class=metric-sub>' + sub + '</div>' if sub else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )


def alert(message: str, kind: str = "blue") -> None:
    """Render an alert box. kind ∈ {green, blue, orange, red}."""
    st.markdown(f"<div class='alert alert-{kind}'>{message}</div>", unsafe_allow_html=True)


def badge(label: str, workout_type: str) -> str:
    """Return inline badge HTML for a workout type."""
    css = f"badge-{workout_type.lower()}"
    return f"<span class='badge {css}'>{label}</span>"
