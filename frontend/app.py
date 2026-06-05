"""
frontend/app.py
────────────────
FitAI — Main dashboard page.

Entry points:
  Local:           streamlit run frontend/app.py
  Streamlit Cloud: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st

from frontend.components.theme import (
    AMBER, BLUE, ELECTRIC, EMERALD, FLAME, NEON_GREEN, PURPLE, RED,
    TEXT_MUTED, WORKOUT_ICONS, DARK_BG, CARD_BG, BORDER, TEXT_DIM,
    alert, apply_theme, hero, metric_card, section,
)
from frontend.components.charts import make_hr_chart, make_gauge, make_radar
from frontend.components.inference import run_inference
from frontend.components.db_utils import init_session_state_defaults, render_db_user_selector

# ── Page config ───────────────────────────────────────────────────────────────
# set_page_config is called by streamlit_app.py when launched via Streamlit
# Cloud. Call it here only when this file is the direct entry point.
try:
    st.set_page_config(
        page_title="FitAI — Fitness Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/your-org/fitness-rec-engine",
            "About": "FitAI · DeepFM recommendations · AUC 0.91",
        },
    )
except st.errors.StreamlitAPIException:
    pass   # already called by the parent entry-point
apply_theme()
init_session_state_defaults()

# ── Constants ─────────────────────────────────────────────────────────────────
WORKOUT_BADGE: dict[str, str] = {
    "HIIT":     "badge-hiit",
    "Strength": "badge-strength",
    "Yoga":     "badge-yoga",
    "Cardio":   "badge-cardio",
    "Pilates":  "badge-pilates",
    "Meal":     "badge-yoga",
}

CONTENT_POOL: list[dict[str, Any]] = [
    {"id": "w01", "title": "Tabata Inferno: 20-Min HIIT",         "type": "HIIT",     "duration": 20, "calories": 340, "intensity": 0.92, "level": "Advanced"},
    {"id": "w02", "title": "5×5 Strength Protocol",               "type": "Strength", "duration": 45, "calories": 280, "intensity": 0.75, "level": "Intermediate"},
    {"id": "w03", "title": "Morning Vinyasa Flow",                 "type": "Yoga",     "duration": 30, "calories": 120, "intensity": 0.35, "level": "Beginner"},
    {"id": "w04", "title": "Zone 2 Cardio: 45-Min Steady State",  "type": "Cardio",   "duration": 45, "calories": 380, "intensity": 0.62, "level": "Intermediate"},
    {"id": "w05", "title": "Core & Stability Pilates",             "type": "Pilates",  "duration": 35, "calories": 160, "intensity": 0.45, "level": "Beginner"},
    {"id": "w06", "title": "Sprint Interval Ladder",               "type": "HIIT",     "duration": 25, "calories": 420, "intensity": 0.88, "level": "Advanced"},
    {"id": "w07", "title": "Olympic Lifting Primer",               "type": "Strength", "duration": 60, "calories": 310, "intensity": 0.80, "level": "Advanced"},
    {"id": "w08", "title": "Yin Yoga: Deep Hip Release",           "type": "Yoga",     "duration": 60, "calories": 95,  "intensity": 0.25, "level": "Beginner"},
    {"id": "w09", "title": "Rowing Machine AMRAP",                 "type": "Cardio",   "duration": 30, "calories": 340, "intensity": 0.70, "level": "Intermediate"},
    {"id": "w10", "title": "Reformer Pilates Intermediate",        "type": "Pilates",  "duration": 50, "calories": 200, "intensity": 0.55, "level": "Intermediate"},
    {"id": "w11", "title": "Kettlebell Complex Circuit",           "type": "HIIT",     "duration": 30, "calories": 390, "intensity": 0.85, "level": "Intermediate"},
    {"id": "w12", "title": "Deadlift Day: Volume Block",           "type": "Strength", "duration": 70, "calories": 350, "intensity": 0.78, "level": "Advanced"},
    {"id": "w13", "title": "Power Yoga: Strength & Flow",          "type": "Yoga",     "duration": 45, "calories": 180, "intensity": 0.55, "level": "Intermediate"},
    {"id": "w14", "title": "Tempo Run: Lactate Threshold",         "type": "Cardio",   "duration": 40, "calories": 420, "intensity": 0.78, "level": "Intermediate"},
    {"id": "w15", "title": "Barre Pilates Fusion",                 "type": "Pilates",  "duration": 45, "calories": 220, "intensity": 0.50, "level": "Beginner"},
]
# _POOL_JSON is defined dynamically based on DB pool later in the file.

GOAL_ENC: dict[str, int] = {
    "Weight Loss": 0, "Muscle Gain": 1, "Endurance": 2,
    "Flexibility": 3, "Maintenance": 4,
}
HR_ZONE_ID: dict[str, int] = {
    "Resting": 0, "Fat Burn": 1, "Cardio": 2, "Peak": 3, "Anaerobic": 4,
}


def _hr_zone(bpm: int) -> tuple[str, str]:
    if bpm < 100: return "Resting",   TEXT_MUTED
    if bpm < 120: return "Fat Burn",  EMERALD
    if bpm < 150: return "Cardio",    BLUE
    if bpm < 170: return "Peak",      AMBER
    return "Anaerobic",               RED


def _user_state_key(state: dict) -> str:
    """Stable, short cache key derived from the user's current values."""
    sig = f"{state['hr_bpm']}-{state['fatigue']:.2f}-{state['fitness_goal_enc']}-{state['age']}"
    return sig


# Session state defaults are initialised via init_session_state_defaults() above.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:16px 0 8px;'>"
        "<div style='font-size:36px'>⚡</div>"
        "<div style='font-size:20px;font-weight:900;"
        "background:linear-gradient(90deg,#00D4FF,#39FF14);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>FitAI</div>"
        "<div style='font-size:11px;color:#6B7280;letter-spacing:2px;'>INTELLIGENCE PLATFORM</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    render_db_user_selector()

    st.markdown("### 👤 Profile")
    name   = st.text_input("Name", key="name", label_visibility="collapsed", placeholder="Your name")
    age    = st.slider("Age", 18, 70, key="age")
    weight = st.slider("Weight (kg)", 45, 140, key="weight")
    height_cm = st.slider("Height (cm)", 150, 210, key="height_cm")
    bmi    = weight / (height_cm / 100) ** 2
    goal   = st.selectbox("🎯 Fitness Goal", list(GOAL_ENC), key="goal")
    freq   = st.slider("📅 Workouts / week", 1, 7, key="freq_per_week")

    st.markdown("### ⚕️ Health Markers")
    htn     = st.toggle("Hypertension", key="htn")
    cardiac = st.toggle("Cardiac Risk", key="cardiac")
    diabetes= st.toggle("Diabetes", key="diabetes")

    st.markdown("### 🎮 Live State")
    hr_bpm  = st.slider("❤️ Heart Rate (bpm)", 50, 200, 72)
    fatigue = st.slider("😴 Fatigue", 0.0, 1.0, 0.25, step=0.05)
    recovery = float(np.clip(1.0 - fatigue * 0.7 + random.gauss(0, 0.02), 0, 1))
    zone_name, zone_color = _hr_zone(hr_bpm)

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#374151;text-align:center;'>"
        "DeepFM · AUC 0.91 · 390K params<br>"
        "TF-IDF+JL · 409-dim features<br>"
        "<span style='color:#39FF14'>● Live</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# Shared user-state dict (passed to inference)
user_state: dict[str, Any] = {
    "age":              age,
    "bmi":              round(bmi, 1),
    "weight":           weight,
    "height":           height_cm,
    "fitness_goal_enc": GOAL_ENC[goal],
    "hr_bpm":           hr_bpm,
    "fatigue":          fatigue,
    "recovery":         round(recovery, 3),
    "hr_zone_id":       HR_ZONE_ID.get(zone_name, 2),
    "freq_per_week":    freq,
    "adherence":        min(freq / 7.0, 1.0),
    "is_hypertensive":  htn,
    "has_cardiac_risk": cardiac,
    "has_diabetes":     diabetes,
}

# ── Dynamic Content Pool from Database ──
content_pool_list = st.session_state.db_pool if st.session_state.db_pool else CONTENT_POOL
_POOL_JSON = json.dumps(content_pool_list)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_dash, tab_recs, tab_analytics, tab_model, tab_progress = st.tabs([
    "🏠 Dashboard",
    "⚡ Recommendations",
    "📊 Analytics",
    "🧠 Model",
    "📈 Progress",
])

# ── Update HR stream ──────────────────────────────────────────────────────────
st.session_state.hr_history.append(
    int(np.clip(hr_bpm + random.gauss(0, 3), 40, 220))
)
st.session_state.hr_history = st.session_state.hr_history[-60:]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_dash:
    greeting = (
        "Good morning" if datetime.now().hour < 12
        else "Good afternoon" if datetime.now().hour < 17
        else "Good evening"
    )
    hero(
        title=f"{greeting}, {name}. 💪",
        subtitle=(
            f"Your DeepFM engine ranked <strong style='color:#00D4FF'>15 workouts</strong> "
            f"in real-time using {freq}-day schedule, current biometrics, and 30-day history."
        ),
        extras_html=(
            f"<div style='margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;'>"
            f"<span class='badge badge-hiit'>🔥 {st.session_state.streak}-Day Streak</span>"
            f"<span class='badge badge-cardio'>📅 {st.session_state.weekly_done}/{freq} This Week</span>"
            f"<span class='badge badge-yoga'>BMI {bmi:.1f}</span>"
            f"<span style='padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;"
            f"background:#39FF1410;color:#39FF14;border:1px solid #39FF1430;'>{zone_name} Zone</span>"
            f"</div>"
        ),
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("Heart Rate",     f"♥ {hr_bpm}",                     zone_name,                                    "green")
    with k2: metric_card("Fatigue",        f"{fatigue*100:.0f}%",              "🟢 Fresh" if fatigue < 0.3 else "🔴 High",   "orange")
    with k3: metric_card("Recovery",       f"{recovery*100:.0f}%",             "🟢 Ready" if recovery > 0.7 else "🟡 Partial","blue")
    with k4: metric_card("Weekly",         f"{st.session_state.weekly_done}/{freq}", f"{st.session_state.weekly_done/freq*100:.0f}% of target", "purple")
    with k5: metric_card("Streak",         f"{st.session_state.streak}d",      "🔥 Keep going!",                             "green")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("❤️ Live Heart Rate")
        st.plotly_chart(
            make_hr_chart(st.session_state.hr_history, hr_bpm, zone_color),
            use_container_width=True,
            config=dict(displayModeBar=False),
        )

        section("🎯 Biometric Profile")
        radar_vals = [
            round(min(hr_bpm / 180.0, 1.0) * 100),
            round(min((weight - 40) / 80.0, 1.0) * 100),
            round((1.0 - fatigue) * 100),
            round(recovery * 100),
            round(min(st.session_state.weekly_done / max(freq, 1), 1.0) * 100),
            65,
        ]
        st.plotly_chart(
            make_radar(
                ["Cardio", "Strength", "Flexibility", "Recovery", "Consistency", "Nutrition"],
                radar_vals,
            ),
            use_container_width=True,
            config=dict(displayModeBar=False),
        )

    with col_r:
        section("⚡ Top Picks Right Now")
        recs = run_inference(
            _user_state_key(user_state),
            json.dumps(user_state),
            _POOL_JSON,
        )[:5]

        for i, r in enumerate(recs):
            icon  = WORKOUT_ICONS.get(r["type"], "🏋️")
            badge = WORKOUT_BADGE.get(r["type"], "")
            rank_cls = "top" if i == 0 else ""
            st.markdown(
                f"<div class='rec-card'>"
                f"<div class='rec-rank {rank_cls}'>#{i+1}</div>"
                f"<div class='rec-icon'>{icon}</div>"
                f"<div class='rec-body'>"
                f"<div class='rec-title'>{r['title']}</div>"
                f"<div class='rec-meta'>⏱ {r['duration']}min · 🔥 {r['calories']} cal · "
                f"<span class='badge {badge}'>{r['type']}</span></div>"
                f"</div>"
                f"<div class='rec-score'>{r['score']*100:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        section("🛡️ Smart Alerts")
        if fatigue > 0.65:
            alert("⚠️ High fatigue — low-intensity workouts recommended today.", "orange")
        if htn:
            alert("ℹ️ HTN filter active — high-sodium recipes are excluded.", "blue")
        if cardiac:
            alert("⚠️ Cardiac safety mode — intensity capped at 70%.", "orange")
        if recovery > 0.75 and fatigue < 0.3:
            alert("✅ Peak recovery state — ideal for high-intensity training.", "green")
        if st.session_state.weekly_done >= freq:
            alert("🏆 Weekly goal achieved! Consider active recovery.", "green")

        section("📊 Body Metrics")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(make_gauge(fatigue,   "Fatigue",   FLAME),    use_container_width=True, config=dict(displayModeBar=False))
        with gc2:
            st.plotly_chart(make_gauge(recovery,  "Recovery",  NEON_GREEN), use_container_width=True, config=dict(displayModeBar=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — RECOMMENDATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_recs:
    hero("⚡ Personalised Recommendations",
         "DeepFM ranks 15 workouts in real-time using your 409-dim biometric feature vector.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        type_filter = st.multiselect(
            "Workout Type", list(WORKOUT_ICONS),
            default=list(WORKOUT_ICONS),
        )
    with fc2:
        level_filter = st.multiselect(
            "Level", ["Beginner", "Intermediate", "Advanced"],
            default=["Beginner", "Intermediate", "Advanced"],
        )
    with fc3:
        max_dur = st.slider("Max Duration (min)", 15, 90, 70)

    all_recs = run_inference(_user_state_key(user_state), json.dumps(user_state), _POOL_JSON)
    filtered = [
        r for r in all_recs
        if r["type"] in type_filter
        and r["level"] in level_filter
        and r["duration"] <= max_dur
    ]

    alert(f"🧠 Showing <strong>{len(filtered)}</strong> recommendations ranked by DeepFM.", "blue")
    st.markdown("<br>", unsafe_allow_html=True)

    for i, r in enumerate(filtered):
        icon      = WORKOUT_ICONS.get(r["type"], "🏋️")
        badge_cls = WORKOUT_BADGE.get(r["type"], "")
        score_pct = r["score"] * 100
        bar_col   = NEON_GREEN if score_pct > 70 else ELECTRIC if score_pct > 50 else AMBER
        rank_cls  = "top" if i == 0 else ""

        col_card, col_btn = st.columns([5, 1])
        with col_card:
            st.markdown(
                f"<div class='rec-card' style='padding:20px 24px;'>"
                f"<div class='rec-rank {rank_cls}' style='font-size:15px;min-width:30px;'>#{i+1}</div>"
                f"<div class='rec-icon' style='font-size:32px;'>{icon}</div>"
                f"<div class='rec-body'>"
                f"<div class='rec-title' style='font-size:15px;'>{r['title']}</div>"
                f"<div class='rec-meta' style='margin:5px 0;'>⏱ {r['duration']} min · "
                f"🔥 {r['calories']} kcal · 💪 {r['level']} · "
                f"<span class='badge {badge_cls}'>{r['type']}</span></div>"
                f"<div style='font-size:11px;color:{TEXT_MUTED};margin-bottom:3px;'>"
                f"Score: <span style='color:{bar_col};font-weight:700;font-family:JetBrains Mono;'>{score_pct:.1f}</span></div>"
                f"<div class='prog-wrap'><div class='prog-bar' style='width:{score_pct:.1f}%;background:{bar_col};'></div></div>"
                f"</div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-family:JetBrains Mono;font-size:26px;font-weight:900;color:{bar_col};'>{score_pct:.0f}</div>"
                f"<div style='font-size:10px;color:{TEXT_MUTED};'>/&nbsp;100</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("▶ Start", key=f"start_{r['id']}"):
                st.toast(f"🚀 Starting {r['title']}!", icon="⚡")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_analytics:
    import pandas as pd
    import plotly.express as px
    from frontend.components.charts import make_bar, make_heatmap

    hero("📊 Dataset Analytics",
         "Insights from FitBit · Gym Members · Fitness 2024 · Mental Health datasets.")

    @st.cache_data(ttl=3600)
    def _load_datasets() -> dict[str, pd.DataFrame]:
        from pathlib import Path
        base = Path(__file__).parent.parent / "data" / "kaggle_raw"
        result: dict[str, pd.DataFrame] = {}
        files = {
            "daily":   base / "fitbit" / "dailyActivity_merged.csv",
            "sleep":   base / "fitbit" / "sleepDay_merged.csv",
            "gym":     base / "gym_members" / "gym_members_exercise_tracking.csv",
            "f24":     base / "daily_activity_2024" / "fitness_track_daily_activity.csv",
            "mental":  base / "mental_health" / "mental_health.csv",
        }
        for key, path in files.items():
            if path.exists():
                result[key] = pd.read_csv(path)
        return result

    data = _load_datasets()
    if not data:
        alert("Run `python scripts/train_kaggle.py` to generate datasets first.", "orange")
    else:
        a1, a2 = st.columns(2)
        with a1:
            if "daily" in data:
                section("🏃 FitBit — Daily Steps Distribution")
                df = data["daily"]
                fig = px.histogram(df, x="TotalSteps", nbins=40,
                                   color_discrete_sequence=[ELECTRIC])
                fig.add_vline(x=df["TotalSteps"].mean(), line_dash="dot",
                               line_color=NEON_GREEN,
                               annotation_text=f"Mean: {df['TotalSteps'].mean():,.0f}",
                               annotation_font_color=NEON_GREEN)
                fig.update_layout(height=260, paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                                   font_color=TEXT_DIM, showlegend=False,
                                   margin=dict(l=0, r=0, t=0, b=0),
                                   xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
                st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

            if "sleep" in data:
                section("😴 Sleep Efficiency")
                df_s = data["sleep"].copy()
                df_s["efficiency"] = df_s["TotalMinutesAsleep"] / df_s["TotalTimeInBed"].clip(lower=1)
                fig2 = px.box(df_s, y="efficiency", color_discrete_sequence=[PURPLE])
                fig2.update_layout(height=220, paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                                    font_color=TEXT_DIM, showlegend=False,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    yaxis=dict(gridcolor=BORDER, title="Sleep Efficiency",
                                               tickformat=".0%"))
                st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))

        with a2:
            if "gym" in data:
                section("💪 Gym Members — Calories by Workout Type")
                df_g = data["gym"]
                fig3 = px.violin(df_g, x="Workout_Type", y="Calories_Burned",
                                  color="Workout_Type",
                                  color_discrete_map={"HIIT": FLAME, "Strength": PURPLE,
                                                       "Yoga": EMERALD, "Cardio": BLUE})
                fig3.update_layout(height=260, paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                                    font_color=TEXT_DIM, showlegend=False,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    xaxis=dict(gridcolor=BORDER),
                                    yaxis=dict(gridcolor=BORDER, title="Calories Burned"))
                st.plotly_chart(fig3, use_container_width=True, config=dict(displayModeBar=False))

            if "mental" in data:
                section("🧠 Mental Health — Stress by Occupation")
                df_m = data["mental"].copy()
                bmap = {"Yes": 1, "No": 0, "Not sure": 0.5, "Maybe": 0.5}
                df_m["stress"] = (
                    df_m["Growing_Stress"].map(bmap).fillna(0) * 0.35
                    + df_m["Coping_Struggles"].map(bmap).fillna(0) * 0.30
                    + df_m["Social_Weakness"].map(bmap).fillna(0) * 0.35
                )
                occ = df_m.groupby("Occupation")["stress"].mean().reset_index()
                fig4 = px.bar(occ, x="Occupation", y="stress",
                               color="stress",
                               color_continuous_scale=[NEON_GREEN, AMBER, RED])
                fig4.update_layout(height=220, paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                                    font_color=TEXT_DIM, showlegend=False,
                                    coloraxis_showscale=False,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    xaxis=dict(gridcolor=BORDER),
                                    yaxis=dict(gridcolor=BORDER, title="Avg Stress Score"))
                st.plotly_chart(fig4, use_container_width=True, config=dict(displayModeBar=False))

        # 30-day activity heatmap
        if "daily" in data:
            section("📅 FitBit — 30-Day Calorie Heatmap")
            df = data["daily"].copy()
            df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
            pivot = df.pivot_table(
                index="Id", columns=df["ActivityDate"].dt.day, values="Calories", aggfunc="mean"
            )
            fig5 = px.imshow(
                pivot,
                color_continuous_scale=[DARK_BG, "#1E1E3F", BLUE, ELECTRIC, NEON_GREEN],
                labels=dict(x="Day of Month", y="User ID", color="Calories"),
            )
            fig5.update_layout(height=300, paper_bgcolor=DARK_BG, margin=dict(l=0, r=0, t=0, b=0),
                                font_color=TEXT_DIM,
                                coloraxis_colorbar=dict(tickfont=dict(color=TEXT_MUTED)))
            st.plotly_chart(fig5, use_container_width=True, config=dict(displayModeBar=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_model:
    from frontend.components.charts import make_training_history

    hero("🧠 Model Insights",
         "DeepFM architecture · Kaggle training history · Feature attribution · ONNX pipeline.")

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, color in [
        (m1, "AUC Combined", "0.9108", "green"),
        (m2, "AUC Click",    "0.8647", "blue"),
        (m3, "AUC Complete", "0.9415", "purple"),
        (m4, "Parameters",   "390K",   "amber"),
    ]:
        with col:
            metric_card(lbl, val, color=color)

    st.markdown("<br>", unsafe_allow_html=True)
    mi1, mi2 = st.columns(2)

    with mi1:
        section("📈 Training History")
        epochs   = list(range(1, 17))
        val_auc  = [0.827,0.838,0.843,0.845,0.848,0.858,0.862,0.872,
                    0.874,0.876,0.877,0.880,0.875,0.873,0.872,0.874]
        loss     = [1.316,0.996,0.921,0.843,0.825,0.788,0.718,0.652,
                    0.609,0.587,0.572,0.526,0.510,0.451,0.450,0.425]
        st.plotly_chart(
            make_training_history(epochs, val_auc, loss, best_epoch=12),
            use_container_width=True, config=dict(displayModeBar=False),
        )

        section("🔍 Feature Slice Attribution")
        st.plotly_chart(
            make_bar(
                x=["Embedding [0:384]", "Categorical [384:394]", "Realtime [394:409]"],
                y=[68.4, 18.2, 13.4],
                colors=[ELECTRIC, PURPLE, NEON_GREEN],
                title_x="% Attribution", height=160,
            ),
            use_container_width=True, config=dict(displayModeBar=False),
        )

    with mi2:
        section("🏗️ Architecture")
        st.code(
            "DeepFM (input=409)\n"
            "├─ FM Layer            k=16   O(K×M)\n"
            "├─ Linear (409→1)\n"
            "└─ MLP Deep\n"
            "   ├─ 409→512  SiLU + BN + Drop(0.2)\n"
            "   ├─ 512→256  SiLU + BN + Drop(0.2)\n"
            "   ├─ 256→128  SiLU + BN + Drop(0.2)\n"
            "   └─ 128→64   SiLU + BN + Drop(0.2)\n"
            "Task Heads (fused=66)\n"
            "├─ click_head    → P(click)\n"
            "└─ complete_head → P(done)\n"
            "Score = 0.4·P(click) + 0.6·P(done)",
            language="",
        )

        section("📦 Training Datasets")
        sources = [
            ("arashnic/fitbit",                     "FitBit Tracker",     "930",   ELECTRIC),
            ("bhavikjikadara/mental-health-dataset", "Mental Health",      "292",   PURPLE),
            ("sonialikhan/fitness-track-2024",       "Daily Activity 2024","1,000", NEON_GREEN),
            ("valakhorasani/gym-members",            "Gym Members",        "973",   FLAME),
        ]
        for slug, name, rows, col in sources:
            st.markdown(
                f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;"
                f"padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px;'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0;'></div>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:13px;font-weight:600;color:#F9FAFB;'>{name}</div>"
                f"<div style='font-size:11px;color:{TEXT_MUTED};'>{slug}</div>"
                f"</div>"
                f"<div style='font-family:JetBrains Mono;font-size:12px;color:{col};'>{rows} rows</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — PROGRESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_progress:
    import pandas as pd
    from frontend.components.charts import make_bar, make_line

    hero("📈 Your Progress", "30-day trends · Workout calendar · Body metrics.")

    days   = pd.date_range(end=datetime.now(), periods=30, freq="D")
    rng    = np.random.default_rng(42 + age)
    steps  = [max(0, int(rng.normal(6000 + freq * 500, 1800))) for _ in days]
    cals   = [max(0, int(rng.normal(2200 + freq * 100, 250)))  for _ in days]
    sleep  = [max(3.0, min(10.0, float(rng.normal(7.2, 0.8)))) for _ in days]
    w_days = [1 if rng.random() < freq / 7 else 0               for _ in days]
    wt     = [weight + rng.normal(0, 0.3) * i / 30 * (-0.5 if "Loss" in goal else 0.3)
               for i in range(30)]

    p1, p2 = st.columns(2)
    with p1:
        section("👟 Daily Steps (30 days)")
        st.plotly_chart(
            make_bar(
                x=[d.strftime("%b %d") for d in days],
                y=steps,
                colors=[NEON_GREEN if w else BORDER for w in w_days],
                title_y="Steps", height=220,
            ),
            use_container_width=True, config=dict(displayModeBar=False),
        )
        section("😴 Sleep Duration")
        st.plotly_chart(
            make_bar(
                x=[d.strftime("%b %d") for d in days],
                y=sleep,
                colors=[NEON_GREEN if s >= 7 else AMBER if s >= 6 else RED for s in sleep],
                title_y="Hours", height=200,
            ),
            use_container_width=True, config=dict(displayModeBar=False),
        )

    with p2:
        section("🔥 Calories Burned")
        st.plotly_chart(
            make_line(
                x=[d.strftime("%b %d") for d in days],
                y_series={"Calories": cals},
                colors={"Calories": FLAME},
                fill=True, height=220, title_y="kcal",
            ),
            use_container_width=True, config=dict(displayModeBar=False),
        )
        section("⚖️ Weight Trend")
        target_w = weight - 3 if "Loss" in goal else weight + 3 if "Gain" in goal else weight
        fig_wt = make_line(
            x=[d.strftime("%b %d") for d in days],
            y_series={"Weight": wt},
            colors={"Weight": PURPLE},
            fill=True, height=200, title_y="kg",
        )
        fig_wt.add_hline(y=target_w, line_dash="dot", line_color=NEON_GREEN,
                          annotation_text="Goal", annotation_font_color=NEON_GREEN)
        st.plotly_chart(fig_wt, use_container_width=True, config=dict(displayModeBar=False))

    # Workout calendar
    section("📅 Workout Calendar")
    cal_html = "<div style='display:flex;gap:5px;flex-wrap:wrap;'>"
    for d, done, s in zip(days, w_days, steps):
        intensity = min(s / 10000.0, 1.0)
        opacity   = 0.25 + intensity * 0.55
        bg        = f"rgba(57,255,20,{opacity:.2f})" if done else CARD_BG
        border    = NEON_GREEN if done else BORDER
        label     = f"{d.strftime('%b %d')}: {s:,} steps"
        cal_html += (
            "<div title='" + label + "' "
            f"style='width:32px;height:32px;border-radius:6px;background:{bg};"
            f"border:1px solid {border};display:flex;align-items:center;"
            f"justify-content:center;font-size:10px;color:{TEXT_MUTED};'>"
            f"{d.day}</div>"
        )
    cal_html += "</div>"
    cal_html += (
        f"<div style='margin-top:8px;font-size:11px;color:{TEXT_MUTED};'>"
        "Workout day = green  |  Rest day = dark  |  Brightness = step count"
        "</div>"
    )
    st.markdown(cal_html, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='position:fixed;bottom:12px;right:16px;font-size:10px;color:#374151;"
    f"font-family:JetBrains Mono;background:{DARK_BG};padding:4px 8px;border-radius:6px;"
    f"border:1px solid {BORDER};'>"
    f"⚡ FitAI · {datetime.now().strftime('%H:%M:%S')} · DeepFM AUC 0.91"
    f"</div>",
    unsafe_allow_html=True,
)
