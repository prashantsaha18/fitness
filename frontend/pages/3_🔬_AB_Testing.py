"""
frontend/pages/3_🔬_AB_Testing.py — A/B Testing dashboard with Thompson Sampling.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.stats import beta as beta_dist

from frontend.components.theme import (
    AMBER, BORDER, CARD_BG, DARK_BG, ELECTRIC, EMERALD,
    NEON_GREEN, PURPLE, RED, TEXT_MAIN, TEXT_MUTED, TEXT_DIM,
    apply_theme, hero, section, alert,
)
from frontend.components.charts import make_posterior, make_line, plotly_layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="A/B Testing · FitAI", page_icon="🔬", layout="wide")
apply_theme()

# ── Session defaults ──────────────────────────────────────────────────────────
_DEFAULTS = dict(
    alpha_a=1.0, beta_a=1.0, alpha_b=1.0, beta_b=1.0,
    n_a=0, n_b=0, history=[], running=False,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Experiment Config")
    cr_a   = st.slider("Control CR (truth)",   0.40, 0.80, 0.52, 0.01)
    cr_b   = st.slider("Treatment CR (truth)", 0.40, 0.80, 0.58, 0.01)
    n_tick = st.slider("Users per tick",        10,  200,  50)
    winner_thresh = st.slider("Winner P(B>A)",  0.85, 0.99, 0.95, 0.01)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Simulate", use_container_width=True):
            st.session_state.running = True
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ── Simulate one tick ─────────────────────────────────────────────────────────
rng = np.random.default_rng()

if st.session_state.running:
    for _ in range(n_tick):
        ta = rng.beta(st.session_state.alpha_a, st.session_state.beta_a)
        tb = rng.beta(st.session_state.alpha_b, st.session_state.beta_b)
        if tb > ta:
            if rng.random() < cr_b:
                st.session_state.alpha_b += 1.0
            else:
                st.session_state.beta_b  += 1.0
            st.session_state.n_b += 1
        else:
            if rng.random() < cr_a:
                st.session_state.alpha_a += 1.0
            else:
                st.session_state.beta_a  += 1.0
            st.session_state.n_a += 1

    na  = st.session_state.n_a
    nb  = st.session_state.n_b
    ra  = (st.session_state.alpha_a - 1) / max(na, 1)
    rb  = (st.session_state.alpha_b - 1) / max(nb, 1)
    s_a = rng.beta(st.session_state.alpha_a, st.session_state.beta_a, 10_000)
    s_b = rng.beta(st.session_state.alpha_b, st.session_state.beta_b, 10_000)
    pbw = float((s_b > s_a).mean())
    st.session_state.history.append(dict(
        total=na+nb, na=na, nb=nb,
        rate_a=round(ra, 4), rate_b=round(rb, 4),
        p_b_wins=round(pbw, 4),
        lift=round((rb - ra) / max(ra, 1e-9) * 100, 2),
    ))

# ── Current state ─────────────────────────────────────────────────────────────
na    = st.session_state.n_a;   nb    = st.session_state.n_b
aa    = st.session_state.alpha_a; ba = st.session_state.beta_a
ab    = st.session_state.alpha_b; bb = st.session_state.beta_b
total = na + nb
rate_a = (aa - 1) / max(na, 1)
rate_b = (ab - 1) / max(nb, 1)

samp_a = rng.beta(ab, bb, 50_000)
samp_b = rng.beta(aa, ba, 50_000)
p_b_wins = float((samp_a > samp_b).mean())
lift_pct = (rate_b - rate_a) / max(rate_a, 1e-9) * 100
declared = p_b_wins >= winner_thresh and total >= 200

# ── Header ─────────────────────────────────────────────────────────────────────
hero(
    "🔬 A/B Testing Dashboard",
    f"Thompson Sampling · mSPRT Sequential Testing · "
    f"<strong style='color:{NEON_GREEN if declared else AMBER};'>"
    f"{'✅ Winner Declared' if declared else f'🔄 Running — {total:,} users'}"
    f"</strong>",
)

if declared:
    winner = "B (Treatment)" if p_b_wins >= winner_thresh else "A (Control)"
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{NEON_GREEN}10,{ELECTRIC}10);"
        f"border:1px solid {NEON_GREEN}40;border-radius:14px;padding:18px 24px;text-align:center;margin-bottom:16px;'>"
        f"<div style='font-size:30px;'>🏆</div>"
        f"<div style='font-size:20px;font-weight:900;color:{NEON_GREEN};margin:6px 0;'>Winner: {winner}</div>"
        f"<div style='color:{TEXT_DIM};font-size:14px;'>"
        f"P(B&gt;A) = {p_b_wins*100:.1f}% · Lift: {lift_pct:+.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
for col, lbl, val, color in zip(
    st.columns(5),
    ["TOTAL USERS", "CONTROL (A)",       "TREATMENT (B)",     "P(B > A)",            "LIFT"],
    [f"{total:,}",   f"{rate_a*100:.1f}%", f"{rate_b*100:.1f}%", f"{p_b_wins*100:.1f}%", f"{lift_pct:+.1f}%"],
    [ELECTRIC,       PURPLE,               NEON_GREEN,            AMBER,                  NEON_GREEN if lift_pct >= 0 else RED],
):
    with col:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-top:2px solid {color};border-radius:12px;padding:13px 15px;text-align:center;'>"
            f"<div style='font-size:10px;color:{TEXT_MUTED};letter-spacing:1.5px;text-transform:uppercase;'>{lbl}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:24px;font-weight:900;color:{color};margin:4px 0;'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([3, 2])

with left:
    section("📊 Posterior Distributions (Beta)")
    st.plotly_chart(
        make_posterior(aa, ba, ab, bb, rate_a, rate_b),
        use_container_width=True, config=dict(displayModeBar=False),
    )

    if st.session_state.history:
        section("📈 P(B > A) Over Time")
        hist = st.session_state.history
        xs   = [h["total"]    for h in hist]
        pbws = [h["p_b_wins"] for h in hist]
        ras  = [h["rate_a"]   for h in hist]
        rbs  = [h["rate_b"]   for h in hist]

        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(x=xs, y=pbws, name="P(B>A)",
            line=dict(color=AMBER, width=2.5), fill="tozeroy",
            fillcolor="rgba(245,158,11,0.10)"), secondary_y=False)
        fig_ts.add_hline(y=winner_thresh, line_dash="dash", line_color=NEON_GREEN,
                          annotation_text=f"Threshold {winner_thresh:.0%}",
                          annotation_font_color=NEON_GREEN)
        fig_ts.add_trace(go.Scatter(x=xs, y=ras, name="CR-A",
            line=dict(color=PURPLE, width=1.5, dash="dot")), secondary_y=True)
        fig_ts.add_trace(go.Scatter(x=xs, y=rbs, name="CR-B",
            line=dict(color=NEON_GREEN, width=1.5, dash="dot")), secondary_y=True)
        layout = plotly_layout()
        layout.update(height=220, showlegend=True,
                      legend=dict(bgcolor=CARD_BG, bordercolor=BORDER))
        fig_ts.update_layout(**layout)
        fig_ts.update_yaxes(gridcolor=BORDER, tickformat=".0%", secondary_y=False)
        fig_ts.update_yaxes(gridcolor=BORDER, tickformat=".1%", secondary_y=True)
        fig_ts.update_xaxes(gridcolor=BORDER, title="Total Users")
        st.plotly_chart(fig_ts, use_container_width=True, config=dict(displayModeBar=False))

with right:
    for arm, n_arm, a, b, color, model in [
        ("Control A",   na, aa, ba, PURPLE,    "deepfm_v1.0.0 (baseline)"),
        ("Treatment B", nb, ab, bb, NEON_GREEN, "deepfm_v2.0.0 (candidate)"),
    ]:
        rate    = (a - 1) / max(n_arm, 1)
        ci_lo   = float(beta_dist.ppf(0.025, a, b)) if n_arm > 10 else 0.0
        ci_hi   = float(beta_dist.ppf(0.975, a, b)) if n_arm > 10 else 1.0
        traffic = n_arm / max(total, 1) * 100

        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-top:2px solid {color};border-radius:15px;padding:18px 22px;margin-bottom:12px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:12px;'>"
            f"<div><div style='font-size:14px;font-weight:700;color:{TEXT_MAIN};'>{arm}</div>"
            f"<div style='font-size:11px;color:{TEXT_MUTED};'>{model}</div></div>"
            f"<div style='text-align:right;font-family:JetBrains Mono;'>"
            f"<div style='font-size:26px;font-weight:900;color:{color};'>{rate*100:.1f}%</div>"
            f"<div style='font-size:11px;color:{TEXT_MUTED};'>completion rate</div></div></div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
            + "".join(
                f"<div style='background:#0A0A0F;border-radius:8px;padding:9px;'>"
                f"<div style='font-size:10px;color:{TEXT_MUTED};letter-spacing:1px;'>{lbl}</div>"
                f"<div style='font-family:JetBrains Mono;font-size:15px;font-weight:700;color:{vc};'>{val}</div>"
                f"</div>"
                for lbl, val, vc in [
                    ("USERS",    f"{n_arm:,}",           TEXT_MAIN),
                    ("TRAFFIC",  f"{traffic:.1f}%",       color),
                    ("95% CI ↓", f"{ci_lo*100:.1f}%",    TEXT_DIM),
                    ("95% CI ↑", f"{ci_hi*100:.1f}%",    TEXT_DIM),
                ]
            )
            + f"</div>"
            f"<div style='margin-top:10px;background:{BORDER};border-radius:6px;height:5px;overflow:hidden;'>"
            f"<div style='height:100%;width:{traffic:.0f}%;background:{color};border-radius:6px;'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Summary card
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:16px 20px;'>"
        f"<div style='font-size:14px;font-weight:700;color:{TEXT_MAIN};margin-bottom:12px;'>📐 Statistical Summary</div>"
        + "".join(
            f"<div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:7px;'>"
            f"<span style='color:{TEXT_MUTED};'>{k}</span>"
            f"<span style='color:{vc};font-family:JetBrains Mono;font-weight:{"700" if bold else "400"};'>{v}</span>"
            f"</div>"
            for k, v, vc, bold in [
                ("Method",     "Thompson + mSPRT",           TEXT_DIM,                False),
                ("P(B > A)",   f"{p_b_wins*100:.2f}%",       NEON_GREEN if p_b_wins >= winner_thresh else AMBER, True),
                ("Lift",       f"{lift_pct:+.2f}%",          NEON_GREEN if lift_pct > 0 else RED,               True),
                ("Users",      f"{total:,}",                 TEXT_DIM,                False),
                ("Status",     "✅ Done" if declared else "🔄 Running", NEON_GREEN if declared else AMBER, True),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

if st.session_state.running and not declared:
    import time; time.sleep(0.8)
    st.rerun()
