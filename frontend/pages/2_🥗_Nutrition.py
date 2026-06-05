"""
frontend/pages/2_🥗_Nutrition.py — Nutrition Intelligence page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from frontend.components.theme import (
    AMBER, BLUE, BORDER, CARD_BG, DARK_BG, ELECTRIC, EMERALD, FLAME,
    NEON_GREEN, PURPLE, RED, TEXT_MAIN, TEXT_MUTED, TEXT_DIM,
    apply_theme, hero, section, alert, metric_card,
)
from frontend.components.charts import make_bar, make_line, plotly_layout
from frontend.components.db_utils import init_session_state_defaults, render_db_user_selector

st.set_page_config(page_title="Nutrition · FitAI", page_icon="🥗", layout="wide")
apply_theme()
init_session_state_defaults()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_db_user_selector()
    st.divider()

    st.markdown("### 🥗 Nutrition Settings")
    age      = st.slider("Age", 18, 70, key="age")
    weight   = st.slider("Weight (kg)", 45, 140, key="weight")
    height   = st.slider("Height (cm)", 150, 210, key="height_cm")
    gender   = st.selectbox("Gender", ["Male", "Female"])
    goal     = st.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintenance", "Endurance", "Flexibility"], key="goal")
    activity = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
    )
    st.divider()
    htn   = st.toggle("Hypertension", key="htn")
    diab  = st.toggle("Diabetes",     key="diabetes")
    vegan = st.toggle("Vegan",        False)

# ── TDEE (Mifflin-St Jeor) ────────────────────────────────────────────────────
bmr = (10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161))
act = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55,
       "Very Active": 1.725, "Extremely Active": 1.9}[activity]
tdee       = int(bmr * act)
cal_target = {"Weight Loss": int(tdee * 0.80), "Muscle Gain": int(tdee * 1.12),
              "Maintenance": tdee, "Endurance": int(tdee * 1.05), "Flexibility": int(tdee * 0.95)}[goal]
protein_g  = int(weight * (2.0 if goal == "Muscle Gain" else 1.6))
fat_g      = int(cal_target * 0.28 / 9)
carb_g     = int((cal_target - protein_g * 4 - fat_g * 9) / 4)

# ── Header ─────────────────────────────────────────────────────────────────────
hero(
    "🥗 Nutrition Intelligence",
    f"TDEE: <strong style='color:#F9FAFB;'>{tdee:,} kcal</strong> · "
    f"Target: <strong style='color:{NEON_GREEN};'>{cal_target:,} kcal</strong> · "
    f"Goal: <strong style='color:{ELECTRIC};'>{goal}</strong>",
)

# ── Macro summary ─────────────────────────────────────────────────────────────
for col, lbl, val, sub, color in zip(
    st.columns(5),
    ["CALORIES",  "PROTEIN",    "CARBS",    "FAT",       "FIBER"],
    [f"{cal_target:,}", f"{protein_g}g", f"{carb_g}g", f"{fat_g}g", "35g"],
    ["kcal",       "+muscle",    "energy",   "hormones",  "gut health"],
    [ELECTRIC,     NEON_GREEN,   AMBER,      PURPLE,      EMERALD],
):
    with col:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-top:2px solid {color};border-radius:12px;padding:13px 15px;text-align:center;'>"
            f"<div style='font-size:10px;color:{TEXT_MUTED};letter-spacing:1.5px;text-transform:uppercase;'>{lbl}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:24px;font-weight:900;color:{color};margin:4px 0;'>{val}</div>"
            f"<div style='font-size:11px;color:{TEXT_MUTED};'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([2, 3])

# ── Macro donut ───────────────────────────────────────────────────────────────
with left:
    section("🔵 Macro Split")
    fig_donut = go.Figure(go.Pie(
        labels=["Protein", "Carbs", "Fat"],
        values=[protein_g * 4, carb_g * 4, fat_g * 9],
        hole=0.65,
        marker=dict(colors=[NEON_GREEN, AMBER, PURPLE]),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT_MAIN),
    ))
    fig_donut.add_annotation(
        text=f"<b>{cal_target:,}</b><br>kcal",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=TEXT_MAIN, family="JetBrains Mono"),
    )
    layout = plotly_layout()
    layout.update(height=260, showlegend=True,
                  legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(color=TEXT_DIM)))
    fig_donut.update_layout(**layout)
    st.plotly_chart(fig_donut, use_container_width=True, config=dict(displayModeBar=False))

    # ── Hydration tracker ─────────────────────────────────────────────────────
    section("💧 Hydration")
    water_target = round(weight * 0.033 + (0.5 if "Active" in activity else 0.0), 1)
    if "water_cups" not in st.session_state:
        st.session_state.water_cups = 0
    wa, wb = st.columns(2)
    with wa:
        if st.button("+ 250 ml", use_container_width=True):
            st.session_state.water_cups += 1
    with wb:
        if st.button("Reset", use_container_width=True):
            st.session_state.water_cups = 0
    water_l   = st.session_state.water_cups * 0.25
    water_pct = min(water_l / water_target, 1.0)
    drops     = "💧" * min(st.session_state.water_cups, 8)
    bar_col   = ELECTRIC if water_pct >= 0.5 else AMBER
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:13px;"
        f"padding:15px 18px;margin-top:8px;'>"
        f"<div style='font-size:22px;margin-bottom:6px;'>{drops or '🫙'}</div>"
        f"<div style='font-family:JetBrains Mono;font-size:26px;font-weight:900;color:{bar_col};'>"
        f"{water_l:.2f}L</div>"
        f"<div style='font-size:12px;color:{TEXT_MUTED};'>of {water_target}L target</div>"
        f"<div style='background:{BORDER};border-radius:8px;height:8px;margin-top:10px;overflow:hidden;'>"
        f"<div style='height:100%;width:{water_pct*100:.0f}%;"
        f"background:linear-gradient(90deg,{BLUE},{ELECTRIC});border-radius:8px;'></div></div>"
        f"<div style='font-size:11px;color:{NEON_GREEN if water_pct >= 1.0 else TEXT_MUTED};margin-top:5px;'>"
        f"{'✅ Goal reached!' if water_pct >= 1.0 else f'{water_target - water_l:.2f}L remaining'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

# ── Food log ──────────────────────────────────────────────────────────────────
with right:
    section("📓 Today's Food Log")
    ALL_MEALS = [
        {"n": "Greek Yogurt + Berries",    "t": "07:30", "cal": 220, "p": 18, "c": 24, "f": 5,  "e": "🥣"},
        {"n": "Oat Protein Pancakes",      "t": "07:30", "cal": 380, "p": 28, "c": 42, "f": 8,  "e": "🥞"},
        {"n": "Chicken & Quinoa Bowl",     "t": "12:30", "cal": 520, "p": 42, "c": 48, "f": 12, "e": "🍗"},
        {"n": "Tuna Salad Wrap",           "t": "12:30", "cal": 410, "p": 38, "c": 32, "f": 10, "e": "🌯"},
        {"n": "Banana + Peanut Butter",    "t": "15:00", "cal": 290, "p": 8,  "c": 36, "f": 12, "e": "🍌"},
        {"n": "Whey Protein Shake",        "t": "15:00", "cal": 180, "p": 30, "c": 12, "f": 2,  "e": "🥤"},
        {"n": "Salmon + Sweet Potato",     "t": "19:00", "cal": 550, "p": 44, "c": 46, "f": 14, "e": "🐟"},
        {"n": "Lentil Dal + Brown Rice",   "t": "19:00", "cal": 480, "p": 22, "c": 72, "f": 10, "e": "🍛"},
    ]
    GOAL_MEALS = {
        "Weight Loss":  [0, 3, 4, 6],
        "Muscle Gain":  [1, 2, 5, 6],
        "Maintenance":  [0, 2, 4, 7],
        "Endurance":    [0, 3, 4, 6],
        "Flexibility":  [0, 2, 4, 7],
    }
    idxs  = GOAL_MEALS[goal]
    meals = [m for m in [ALL_MEALS[i] for i in idxs]
             if not (vegan and m["e"] in ("🍗", "🐟", "🥤"))]
    if vegan:
        meals.append(ALL_MEALS[7])

    eaten_cal = sum(m["cal"] for m in meals)
    eaten_p   = sum(m["p"]   for m in meals)
    eaten_c   = sum(m["c"]   for m in meals)
    eaten_f   = sum(m["f"]   for m in meals)

    for meal in meals:
        pct = meal["cal"] / cal_target * 100
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:13px;"
            f"padding:14px 18px;margin-bottom:9px;transition:all .2s;'>"
            f"<div style='display:flex;align-items:center;gap:13px;'>"
            f"<div style='font-size:26px;'>{meal['e']}</div>"
            f"<div style='flex:1;'>"
            f"<div style='font-size:14px;font-weight:600;color:{TEXT_MAIN};'>{meal['n']}</div>"
            f"<div style='font-size:12px;color:{TEXT_MUTED};margin:4px 0;'>{meal['t']} · {meal['cal']} kcal</div>"
            f"<div>"
            f"<span style='display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;"
            f"font-weight:600;margin-right:5px;background:{NEON_GREEN}15;color:{NEON_GREEN};"
            f"border:1px solid {NEON_GREEN}30;'>P {meal['p']}g</span>"
            f"<span style='display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;"
            f"font-weight:600;margin-right:5px;background:{AMBER}15;color:{AMBER};"
            f"border:1px solid {AMBER}30;'>C {meal['c']}g</span>"
            f"<span style='display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;"
            f"font-weight:600;background:{PURPLE}15;color:#A78BFA;"
            f"border:1px solid {PURPLE}30;'>F {meal['f']}g</span>"
            f"</div></div>"
            f"<div style='font-family:JetBrains Mono;font-size:13px;color:{ELECTRIC};'>{pct:.0f}%</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Daily totals bar
    on_track = eaten_cal <= cal_target
    bar_col2 = f"linear-gradient(90deg,{NEON_GREEN},{ELECTRIC})" if on_track else f"linear-gradient(90deg,{AMBER},{RED})"
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:13px;padding:15px 18px;'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'>"
        f"<div style='font-size:13px;font-weight:700;color:{TEXT_MAIN};'>Daily Total</div>"
        f"<div style='font-family:JetBrains Mono;font-size:14px;"
        f"color:{NEON_GREEN if on_track else RED};'>{eaten_cal} / {cal_target} kcal</div></div>"
        f"<div style='background:{BORDER};border-radius:8px;height:10px;overflow:hidden;'>"
        f"<div style='height:100%;width:{min(eaten_cal/cal_target,1)*100:.0f}%;"
        f"background:{bar_col2};border-radius:8px;'></div></div>"
        f"<div style='display:flex;gap:18px;margin-top:10px;'>"
        f"<div><div style='font-size:10px;color:{TEXT_MUTED};'>PROTEIN</div>"
        f"<div style='color:{NEON_GREEN};font-weight:700;'>{eaten_p}g/{protein_g}g</div></div>"
        f"<div><div style='font-size:10px;color:{TEXT_MUTED};'>CARBS</div>"
        f"<div style='color:{AMBER};font-weight:700;'>{eaten_c}g/{carb_g}g</div></div>"
        f"<div><div style='font-size:10px;color:{TEXT_MUTED};'>FAT</div>"
        f"<div style='color:#A78BFA;font-weight:700;'>{eaten_f}g/{fat_g}g</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

# ── 7-day trend ───────────────────────────────────────────────────────────────
section("📈 7-Day Nutrition Trend")
days_lbl = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
rng      = np.random.default_rng(42 + age)
w_cal    = [int(rng.normal(cal_target, 150)) for _ in days_lbl]
w_prot   = [int(rng.normal(protein_g,   10)) for _ in days_lbl]

fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
fig_trend.add_trace(go.Bar(x=days_lbl, y=w_cal, name="Calories",
    marker=dict(color=[NEON_GREEN if c <= cal_target else RED for c in w_cal], opacity=0.8)),
    secondary_y=False)
fig_trend.add_trace(go.Scatter(x=days_lbl, y=w_prot, name="Protein (g)",
    line=dict(color=ELECTRIC, width=2), mode="lines+markers", marker=dict(size=7)),
    secondary_y=True)
fig_trend.add_hline(y=cal_target, line_dash="dot", line_color=TEXT_MUTED,
                     annotation_text="Cal Target", secondary_y=False)
layout = plotly_layout()
layout.update(height=230, showlegend=True,
              legend=dict(bgcolor=CARD_BG, bordercolor=BORDER))
fig_trend.update_layout(**layout)
fig_trend.update_yaxes(title_text="Calories", gridcolor=BORDER, secondary_y=False)
fig_trend.update_yaxes(title_text="Protein (g)", gridcolor=BORDER, secondary_y=True)
fig_trend.update_xaxes(gridcolor=BORDER)
st.plotly_chart(fig_trend, use_container_width=True, config=dict(displayModeBar=False))

# ── Alerts ─────────────────────────────────────────────────────────────────────
section("🛡️ Smart Alerts")
ac1, ac2, ac3 = st.columns(3)
with ac1:
    if eaten_p < protein_g * 0.8:
        alert(f"⚠️ Low protein ({eaten_p}g / {protein_g}g). Add a protein source.", "orange")
    else:
        alert(f"✅ Protein on track ({eaten_p}g / {protein_g}g).", "green")
with ac2:
    if htn:
        alert("ℹ️ HTN mode: sodium limit 1,500 mg/day. High-sodium recipes filtered.", "blue")
    elif diab:
        alert("ℹ️ Diabetes mode: low-GI foods prioritised, carbs capped at 150g.", "blue")
    else:
        alert("✅ No dietary restrictions active.", "green")
with ac3:
    if water_pct < 0.5:
        alert(f"💧 Hydration low ({water_l:.1f}L). Target: {water_target}L.", "orange")
    else:
        alert(f"💧 Hydration good ({water_l:.1f}L / {water_target}L).", "green")
