"""
frontend/pages/4_🔭_Data_Explorer.py — Interactive Kaggle dataset explorer.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components.theme import (
    AMBER, BLUE, BORDER, CARD_BG, DARK_BG,
    ELECTRIC, EMERALD, FLAME, NEON_GREEN, PURPLE,
    TEXT_MAIN, TEXT_MUTED, TEXT_DIM,
    apply_theme, hero, section, alert,
)
from frontend.components.charts import make_bar, make_heatmap, plotly_layout

st.set_page_config(page_title="Data Explorer · FitAI", page_icon="🔭", layout="wide")
apply_theme()

# ── Dataset registry ──────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent.parent / "data" / "kaggle_raw"
REGISTRY: dict[str, dict] = {
    "FitBit Daily Activity": {
        "path": _BASE / "fitbit" / "dailyActivity_merged.csv",
        "slug": "arashnic/fitbit", "icon": "🏃", "color": ELECTRIC,
        "desc": "30 users · 31 days · Steps, calories, activity zones",
    },
    "FitBit Sleep": {
        "path": _BASE / "fitbit" / "sleepDay_merged.csv",
        "slug": "arashnic/fitbit", "icon": "😴", "color": PURPLE,
        "desc": "~700 rows · Sleep minutes, time in bed, records",
    },
    "Gym Members Exercise": {
        "path": _BASE / "gym_members" / "gym_members_exercise_tracking.csv",
        "slug": "valakhorasani/gym-members-exercise-dataset", "icon": "💪", "color": NEON_GREEN,
        "desc": "973 rows · BPM, calories, workout type, experience",
    },
    "Fitness Daily 2024": {
        "path": _BASE / "daily_activity_2024" / "fitness_track_daily_activity.csv",
        "slug": "sonialikhan/fitness-track-daily-activity-dataset-2024", "icon": "⚡", "color": AMBER,
        "desc": "1,000 rows · Age, BMI, BPM, workout type, calories",
    },
    "Mental Health Survey": {
        "path": _BASE / "mental_health" / "mental_health.csv",
        "slug": "bhavikjikadara/mental-health-dataset", "icon": "🧠", "color": FLAME,
        "desc": "292 rows · Stress, coping, treatment, occupation",
    },
}


@st.cache_data(ttl=3600)
def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔭 Explorer")
    selected = st.selectbox("Dataset", list(REGISTRY))
    show_raw  = st.toggle("Show raw data", False)
    show_corr = st.toggle("Correlation matrix", True)

# ── Header ─────────────────────────────────────────────────────────────────────
hero("🔭 Kaggle Dataset Explorer",
     "4 datasets · 3,265 records · Interactive cross-dataset analysis")

# ── Dataset cards ─────────────────────────────────────────────────────────────
section("📦 Dataset Overview")
cols = st.columns(3)
for i, (name, meta) in enumerate(REGISTRY.items()):
    df = _load(meta["path"])
    n  = len(df) if df is not None else 0
    active_border = f"border-color:#3B3B6B;" if name == selected else ""
    with cols[i % 3]:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-top:2px solid {meta['color']};border-radius:13px;"
            f"padding:16px 18px;margin-bottom:12px;{active_border}'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
            f"<div style='font-size:22px;'>{meta['icon']}</div>"
            f"<div style='font-size:13px;font-weight:700;color:{TEXT_MAIN};'>{name}</div></div>"
            f"<div style='font-size:11px;color:{TEXT_MUTED};margin-bottom:5px;'>{meta['slug']}</div>"
            f"<div style='font-size:12px;color:{TEXT_DIM};'>{meta['desc']}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:14px;font-weight:700;"
            f"color:{meta['color']};margin-top:8px;'>"
            f"{'❌ Not found' if n == 0 else f'{n:,} rows'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Selected dataset exploration ──────────────────────────────────────────────
df = _load(REGISTRY[selected]["path"])
if df is None:
    alert("Run `python scripts/train_kaggle.py` to generate datasets.", "orange")
    st.stop()

section(f"🔍 {selected}")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
nulls    = int(df.isnull().sum().sum())

for col, lbl, val, color in zip(
    st.columns(4),
    ["ROWS",     "COLUMNS",         "NUMERIC",     "NULL VALUES"],
    [f"{len(df):,}", f"{len(df.columns)}", f"{len(num_cols)}", f"{nulls}"],
    [ELECTRIC,   NEON_GREEN,        AMBER,         RED if nulls > 0 else NEON_GREEN],
):
    with col:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-top:2px solid {color};border-radius:12px;padding:12px 15px;text-align:center;'>"
            f"<div style='font-size:10px;color:{TEXT_MUTED};letter-spacing:1.5px;text-transform:uppercase;'>{lbl}</div>"
            f"<div style='font-family:JetBrains Mono;font-size:24px;font-weight:900;color:{color};margin:4px 0;'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

if show_raw:
    section("📋 Raw Data (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True, height=280)

# ── Numeric distributions ─────────────────────────────────────────────────────
if num_cols:
    section("📊 Numeric Distributions")
    plot_cols = num_cols[:6]
    ncols_grid = min(3, len(plot_cols))
    nrows_grid = math.ceil(len(plot_cols) / ncols_grid)
    grid_cols  = st.columns(ncols_grid)
    cycle = [ELECTRIC, NEON_GREEN, AMBER, PURPLE, FLAME, RED]

    for i, col_name in enumerate(plot_cols):
        with grid_cols[i % ncols_grid]:
            vals = df[col_name].dropna()
            fig = px.histogram(vals, nbins=30, color_discrete_sequence=[cycle[i % len(cycle)]])
            fig.update_layout(height=200, showlegend=False,
                              title=dict(text=col_name, font=dict(size=12, color=TEXT_DIM)),
                              paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                              font_color=TEXT_MUTED,
                              margin=dict(l=0, r=0, t=30, b=0),
                              xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

# ── Correlation matrix ────────────────────────────────────────────────────────
if show_corr and len(num_cols) >= 2:
    section("🔗 Correlation Matrix")
    c_cols = num_cols[:10]
    corr   = df[c_cols].corr().values
    st.plotly_chart(
        make_heatmap(corr, c_cols, c_cols),
        use_container_width=True, config=dict(displayModeBar=False),
    )

# ── Categorical breakdown ─────────────────────────────────────────────────────
if cat_cols:
    section("📑 Categorical Breakdown")
    pick = st.selectbox("Column", cat_cols[:8])
    vc   = df[pick].value_counts().head(12)
    st.plotly_chart(
        make_bar(x=list(vc.index), y=list(vc.values),
                 colors=[ELECTRIC] * len(vc),
                 title_x=pick, title_y="Count", height=240),
        use_container_width=True, config=dict(displayModeBar=False),
    )

# ── Cross-dataset: BMI vs Calories ────────────────────────────────────────────
section("🔗 Cross-Dataset: BMI vs Calories Burned")
gym = _load(REGISTRY["Gym Members Exercise"]["path"])
f24 = _load(REGISTRY["Fitness Daily 2024"]["path"])

if gym is not None and f24 is not None:
    # Normalise calories column name
    f24_cal_col = next((c for c in f24.columns if "Calorie" in c), None)
    if f24_cal_col:
        gym["_src"] = "Gym Members"
        f24["_src"] = "Fitness 2024"
        g_sub = gym[["BMI", "Calories_Burned", "Workout_Type", "_src"]].rename(columns={"Calories_Burned": "Calories"})
        f_sub = f24[["BMI", f24_cal_col, "Workout_Type", "_src"]].rename(columns={f24_cal_col: "Calories"})
        comb  = pd.concat([g_sub, f_sub], ignore_index=True).dropna()
        fig_cross = px.scatter(
            comb, x="BMI", y="Calories",
            color="Workout_Type", symbol="_src",
            color_discrete_map={"HIIT": FLAME, "Strength": PURPLE,
                                 "Yoga": EMERALD, "Cardio": BLUE, "Pilates": AMBER},
            trendline="lowess", trendline_scope="overall",
            trendline_color_override=NEON_GREEN,
            opacity=0.65, height=300,
        )
        fig_cross.update_traces(marker=dict(size=7))
        layout = plotly_layout()
        layout.update(height=300, showlegend=True,
                      xaxis=dict(gridcolor=BORDER, title="BMI"),
                      yaxis=dict(gridcolor=BORDER, title="Calories Burned"))
        fig_cross.update_layout(**layout)
        st.plotly_chart(fig_cross, use_container_width=True, config=dict(displayModeBar=False))
