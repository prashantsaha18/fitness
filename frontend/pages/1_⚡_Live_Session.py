"""
frontend/pages/1_⚡_Live_Session.py
Real-time workout session tracker.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

import streamlit as st

from frontend.components.theme import (
    AMBER, BLUE, CARD_BG, DARK_BG, BORDER, ELECTRIC, EMERALD,
    NEON_GREEN, RED, TEXT_MUTED, TEXT_DIM, TEXT_MAIN,
    apply_theme, section,
)
from frontend.components.charts import make_hr_chart, make_gauge
from frontend.components.db_utils import init_session_state_defaults, render_db_user_selector

st.set_page_config(page_title="Live Session · FitAI", page_icon="⚡", layout="wide")
apply_theme()
init_session_state_defaults()

# ── Workout definitions ───────────────────────────────────────────────────────
WORKOUTS: dict[str, dict[str, Any]] = {
    "HIIT": {
        "icon": "🔥", "met": 9.0,
        "intervals": [
            ("🔥 Sprint Burst",      30, 0.90, RED),
            ("🚶 Active Recovery",   15, 0.45, EMERALD),
            ("🔥 Sprint Burst",      30, 0.90, RED),
            ("🚶 Active Recovery",   15, 0.45, EMERALD),
            ("🔥 Max Effort",        20, 0.95, RED),
            ("🚶 Rest",              30, 0.35, EMERALD),
            ("🔥 Sprint Burst",      30, 0.88, RED),
            ("🚶 Active Recovery",   20, 0.40, EMERALD),
            ("🏁 Final Push",        45, 0.92, AMBER),
            ("🧘 Cool Down",         60, 0.30, BLUE),
        ],
    },
    "Strength": {
        "icon": "💪", "met": 6.0,
        "intervals": [
            ("🏋️ Warm-Up Sets",    180, 0.50, "#8B5CF6"),
            ("💪 Squat 5×5",        300, 0.75, RED),
            ("⏸ Rest",              120, 0.25, TEXT_MUTED),
            ("💪 Deadlift 3×5",     240, 0.80, RED),
            ("⏸ Rest",              180, 0.25, TEXT_MUTED),
            ("💪 Bench Press 4×8",  240, 0.70, AMBER),
            ("⏸ Rest",              120, 0.25, TEXT_MUTED),
            ("💪 Pull-ups 3×MAX",   120, 0.65, AMBER),
            ("🧘 Stretch",          180, 0.25, BLUE),
        ],
    },
    "Yoga": {
        "icon": "🧘", "met": 3.0,
        "intervals": [
            ("🧘 Child's Pose",      60, 0.25, EMERALD),
            ("🌅 Sun Salutation A",  90, 0.45, AMBER),
            ("🌅 Sun Salutation B",  90, 0.50, AMBER),
            ("⚡ Warrior Sequence", 120, 0.55, BLUE),
            ("🧘 Balance Poses",    120, 0.45, "#8B5CF6"),
            ("🌊 Hip Openers",      120, 0.35, EMERALD),
            ("🧘 Savasana",         120, 0.20, BLUE),
        ],
    },
    "Cardio": {
        "icon": "🏃", "met": 7.5,
        "intervals": [
            ("🚶 Warm-Up Walk",   300, 0.40, EMERALD),
            ("🏃 Easy Jog",       300, 0.55, BLUE),
            ("🏃 Steady Run",     600, 0.68, AMBER),
            ("🏃 Tempo Pace",     300, 0.78, RED),
            ("🏃 Steady Run",     300, 0.65, AMBER),
            ("🏃 Easy Jog",       300, 0.55, BLUE),
            ("🚶 Cool-Down",      300, 0.35, EMERALD),
        ],
    },
}

COACHING: dict[str, tuple[str, str]] = {
    "Resting":   ("You're warmed up and ready. Begin when you are. 🧘",                      TEXT_MUTED),
    "Fat Burn":  ("Stay in zone — fat adapts here. Breathe deeply. 💪",                      EMERALD),
    "Cardio":    ("Solid aerobic work! Maintain pace and check your form. 🏃",               BLUE),
    "Peak":      ("You're peaking! Push through — this is where gains happen. 🔥",           AMBER),
    "Anaerobic": ("⚡ MAX EFFORT! Everything you have — 10 seconds of everything!", RED),
}

def _hr_zone(bpm: int) -> tuple[str, str]:
    if bpm < 100: return "Resting",   TEXT_MUTED
    if bpm < 120: return "Fat Burn",  EMERALD
    if bpm < 150: return "Cardio",    BLUE
    if bpm < 170: return "Peak",      AMBER
    return "Anaerobic",               RED

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ── Session defaults ──────────────────────────────────────────────────────────
_SESS: dict[str, Any] = {
    "active": False, "start_ts": None, "elapsed": 0.0,
    "calories_live": 0.0, "hr_stream": [], "wt": "HIIT",
}
for k, v in _SESS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_db_user_selector()
    st.divider()

    st.markdown("### ⚡ Session Setup")
    wt     = st.selectbox("Workout", list(WORKOUTS.keys()))
    weight = st.slider("Weight (kg)", 45, 140, key="weight")
    age    = st.slider("Age", 18, 70, key="age")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        label = "▶ START" if not st.session_state.active else "⏹ STOP"
        if st.button(label, use_container_width=True):
            if not st.session_state.active:
                st.session_state.update(
                    active=True, start_ts=time.time(),
                    calories_live=0.0, hr_stream=[], wt=wt,
                )
                st.toast(f"🚀 {wt} started!", icon="⚡")
            else:
                st.session_state.active = False
                st.toast("Session saved! 💪", icon="✅")
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            for k, v in _SESS.items():
                st.session_state[k] = v
            st.rerun()
    if st.session_state.active:
        st.markdown(f"<div style='text-align:center;font-size:12px;color:{NEON_GREEN};'>● LIVE</div>",
                    unsafe_allow_html=True)

# ── Simulate live metrics ─────────────────────────────────────────────────────
workout   = WORKOUTS[wt]
intervals = workout["intervals"]
max_hr    = 220 - age

if st.session_state.active and st.session_state.start_ts:
    elapsed = time.time() - st.session_state.start_ts
    st.session_state.elapsed = elapsed
    # Find current interval
    cum = 0.0
    cur_idx, cur_name, cur_intens, cur_col, interval_remaining = 0, intervals[0][0], intervals[0][2], intervals[0][3], intervals[0][1]
    for i, (name, dur, intens, col) in enumerate(intervals):
        cum += dur
        if elapsed < cum or i == len(intervals) - 1:
            cur_idx, cur_name, cur_intens, cur_col = i, name, intens, col
            interval_remaining = max(0.0, cum - elapsed)
            break
    sim_hr   = int(np.clip(60 + (max_hr - 60) * cur_intens + random.gauss(0, 5), 50, max_hr))
    cal_rate = workout["met"] * cur_intens * weight / 3600.0
    st.session_state.calories_live += cal_rate * 2
    st.session_state.hr_stream.append(sim_hr)
    st.session_state.hr_stream = st.session_state.hr_stream[-120:]
else:
    elapsed = st.session_state.elapsed
    sim_hr  = 72; cur_intens = 0.0; cur_name = "Idle"
    cur_col = TEXT_MUTED; interval_remaining = 0.0; cur_idx = 0

import numpy as np  # noqa: E402
zone_name, zone_col = _hr_zone(sim_hr)
cue_text, cue_col   = COACHING.get(zone_name, COACHING["Resting"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='background:linear-gradient(135deg,#0A1A0D,#0D0D1A);border:1px solid {BORDER};"
    f"border-radius:20px;padding:28px 36px;margin-bottom:20px;'>"
    f"<div style='font-size:13px;color:{TEXT_MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;'>"
    f"{workout['icon']} {wt} · {'● LIVE' if st.session_state.active else '◉ IDLE'}</div>"
    f"<div style='font-family:JetBrains Mono;font-size:64px;font-weight:900;color:{TEXT_MAIN};line-height:1;'>"
    f"{_fmt(elapsed)}</div>"
    f"<div style='margin-top:12px;font-size:14px;color:{TEXT_DIM};'>"
    f"Current: <strong style='color:{TEXT_MAIN};'>{cur_name}</strong> · "
    f"{_fmt(interval_remaining)} remaining</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# KPI row
s1, s2, s3, s4, s5 = st.columns(5)
for col, lbl, val, color in [
    (s1, "HEART RATE",  f"{sim_hr} bpm",                                    zone_col),
    (s2, "CALORIES",    f"{st.session_state.calories_live:.0f} kcal",        "#FF6B35"),
    (s3, "INTENSITY",   f"{cur_intens*100:.0f}%",                            "#8B5CF6"),
    (s4, "INTERVAL",    f"{cur_idx+1}/{len(intervals)}",                     ELECTRIC),
    (s5, "ZONE",        zone_name,                                           zone_col),
]:
    with col:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-top:2px solid {color};"
            f"border-radius:12px;padding:14px 16px;text-align:center;'>"
            f"<div style='font-size:10px;color:{TEXT_MUTED};letter-spacing:1.5px;text-transform:uppercase;'>{lbl}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:18px;font-weight:900;color:{color};margin-top:4px;'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([3, 2])

with left:
    section("❤️ Live Heart Rate")
    hr_data = st.session_state.hr_stream or [72] * 30
    st.plotly_chart(
        make_hr_chart(hr_data, sim_hr, zone_col, max_hr=max_hr, height=270),
        use_container_width=True, config=dict(displayModeBar=False),
    )
    section("⏱ Interval Progress")
    if st.session_state.active and cur_idx < len(intervals):
        _, dur_s, _, _ = intervals[cur_idx]
        time_in = max(0.0, dur_s - interval_remaining)
        pct = min(time_in / max(dur_s, 1), 1.0) * 100
    else:
        pct = 0.0
    st.plotly_chart(
        make_gauge(pct / 100, cur_name[:28], cur_col, height=190),
        use_container_width=True, config=dict(displayModeBar=False),
    )

with right:
    section("📋 Interval Plan")
    for i, (name, dur, _, col) in enumerate(intervals):
        is_cur  = i == cur_idx and st.session_state.active
        is_done = i < cur_idx and st.session_state.active
        bg      = f"{col}20" if is_cur else CARD_BG
        border  = col if is_cur else BORDER
        suffix  = "  ←" if is_cur else "  ✓" if is_done else ""
        opacity = "0.5" if is_done else "1"
        st.markdown(
            f"<div style='background:{bg};border:1px solid {border};border-radius:11px;"
            f"padding:12px 16px;margin-bottom:7px;display:flex;align-items:center;gap:12px;opacity:{opacity};'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0;'></div>"
            f"<div style='flex:1;'>"
            f"<div style='font-size:13px;font-weight:{"700" if is_cur else "400"};color:{"#F9FAFB" if is_cur else TEXT_DIM};'>"
            f"{name}{suffix}</div>"
            f"<div style='font-size:11px;color:{TEXT_MUTED};'>{dur}s · {_*100:.0f}% intensity</div>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono;font-size:12px;color:{col if is_cur else TEXT_MUTED};'>{_fmt(dur)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0D1A0D,#0A0A0F);border:1px solid #39FF1430;"
        f"border-radius:13px;padding:18px 20px;margin-top:14px;'>"
        f"<div style='font-size:11px;color:{NEON_GREEN};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;'>"
        f"🧠 AI Coach</div>"
        f"<div style='font-size:14px;color:{TEXT_MAIN};line-height:1.5;'>{cue_text}</div>"
        f"<div style='margin-top:10px;font-size:12px;color:{TEXT_MUTED};'>"
        f"Zone: <strong style='color:{cue_col};'>{zone_name}</strong> · "
        f"HR: <strong style='color:{zone_col};'>{sim_hr} bpm</strong></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

if st.session_state.active:
    time.sleep(2)
    st.rerun()
