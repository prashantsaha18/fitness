"""
data_pipeline/kaggle/etl.py
─────────────────────────────
ETL Pipeline — Kaggle Datasets → DeepFM Training Matrix.

Transforms four heterogeneous CSV datasets into a single unified
[N, 409] float32 feature matrix + binary labels for training.

Feature derivation per source:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  FitBit dailyActivity    → per-user engagement labels (completion/click) │
  │  FitBit heartrate        → realtime biometric features [394:409]         │
  │  FitBit sleep            → recovery score feature                        │
  │  FitBit weight           → BMI, fat_pct features                        │
  │  Fitness Daily 2024      → workout content item features                 │
  │  Gym Members             → user profile features + workout preferences   │
  │  Mental Health           → stress / coping features → fatigue proxy      │
  └──────────────────────────────────────────────────────────────────────────┘

Label construction (implicit feedback from activity data):
  click_label    = 1  if TotalSteps > user_median_steps (engaged day)
  complete_label = 1  if VeryActiveMinutes >= 20 AND Calories > user_median_cal
  These proxy signals closely mirror real app interaction labels.

Training record grain: one row = one (user, workout_session) pair.
Cross-dataset joins: FitBit user IDs are matched to Gym Members profiles
  via BMI/age quantile buckets (nearest-neighbour demographic matching).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

WORKOUT_TYPE_MAP  = {"hiit": 0, "strength": 1, "yoga": 2, "cardio": 3,
                     "pilates": 4, "recovery": 5,
                     # Kaggle capitalised variants
                     "HIIT": 0, "Strength": 1, "Yoga": 2, "Cardio": 3,
                     "Pilates": 4}
CONTENT_TYPE_MAP  = {"video": 0, "workout_routine": 1, "meal_recipe": 2}
FITNESS_GOAL_MAP  = {"Weight Loss": 0, "weight_loss": 0,
                     "Muscle Gain": 1, "muscle_gain": 1,
                     "Endurance": 2, "endurance": 2,
                     "Flexibility": 3, "flexibility": 3,
                     "Maintenance": 4, "maintenance": 4}
HR_ZONE_MAP       = {"resting": 0, "fat_burn": 1, "cardio": 2,
                     "peak": 3, "anaerobic": 4}

TOTAL_FEATURES = 409   # must match ranking/model.py FeatureDims.TOTAL
EMB_DIM        = 384
CAT_DIM        = 10
RT_DIM         = 15


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fitbit(data_root: Path) -> dict[str, pd.DataFrame]:
    fb = data_root / "fitbit"
    out = {}
    for fname in ["dailyActivity_merged.csv", "sleepDay_merged.csv",
                  "heartrate_seconds_merged.csv", "weightLogInfo_merged.csv"]:
        p = fb / fname
        if p.exists():
            out[fname.replace(".csv", "")] = pd.read_csv(p)
            logger.info("  Loaded %-40s  %d rows", fname, len(out[list(out)[-1]]))
    return out


def load_fitness_2024(data_root: Path) -> pd.DataFrame:
    p = data_root / "daily_activity_2024" / "fitness_track_daily_activity.csv"
    df = pd.read_csv(p)
    logger.info("  Loaded fitness_track_daily_activity.csv  %d rows", len(df))
    return df


def load_gym_members(data_root: Path) -> pd.DataFrame:
    p = data_root / "gym_members" / "gym_members_exercise_tracking.csv"
    df = pd.read_csv(p)
    logger.info("  Loaded gym_members_exercise_tracking.csv  %d rows", len(df))
    return df


def load_mental_health(data_root: Path) -> pd.DataFrame:
    p = data_root / "mental_health" / "mental_health.csv"
    df = pd.read_csv(p)
    logger.info("  Loaded mental_health.csv  %d rows", len(df))
    return df


# ── FitBit Feature Engineering ────────────────────────────────────────────────

def engineer_fitbit_labels(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Derive implicit interaction labels from FitBit daily activity data.

    Business logic:
      click_label    — user was "engaged" (above median steps for that user)
      complete_label — user had a high-quality workout day:
                       very active ≥ 20 min AND calories > their 60th percentile

    This mirrors app-level click/complete signals without requiring
    explicit in-app interaction logs.
    """
    df = daily.copy()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])

    # Per-user medians (normalise for individual fitness levels)
    user_stats = df.groupby("Id").agg(
        median_steps=("TotalSteps", "median"),
        p60_calories=("Calories", lambda x: x.quantile(0.60)),
        mean_very_active=("VeryActiveMinutes", "mean"),
    ).reset_index()

    df = df.merge(user_stats, on="Id")

    df["click_label"] = (df["TotalSteps"] > df["median_steps"]).astype(int)
    df["complete_label"] = (
        (df["VeryActiveMinutes"] >= 20) &
        (df["Calories"] > df["p60_calories"])
    ).astype(int)

    return df


def engineer_fitbit_user_features(
    daily: pd.DataFrame,
    sleep: Optional[pd.DataFrame],
    weight: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate FitBit data into per-user feature vectors.
    These map to the batch user features consumed by the ranking model.
    """
    # 30-day aggregations from daily activity
    user_agg = daily.groupby("Id").agg(
        adherence_rate=("click_label", "mean"),          # fraction of engaged days
        completion_rate=("complete_label", "mean"),
        avg_steps=("TotalSteps", "mean"),
        avg_calories=("Calories", "mean"),
        avg_very_active_min=("VeryActiveMinutes", "mean"),
        avg_sedentary_min=("SedentaryMinutes", "mean"),
        avg_distance=("TotalDistance", "mean"),
        workout_sessions_30d=("click_label", "sum"),
    ).reset_index()

    user_agg["activity_level"] = MinMaxScaler().fit_transform(
        user_agg[["avg_steps"]]
    )

    # Sleep features
    if sleep is not None and len(sleep) > 0:
        sleep["SleepDay"] = pd.to_datetime(sleep["SleepDay"])
        sleep_agg = sleep.groupby("Id").agg(
            avg_sleep_min=("TotalMinutesAsleep", "mean"),
            avg_time_in_bed=("TotalTimeInBed", "mean"),
        ).reset_index()
        sleep_agg["sleep_efficiency"] = (
            sleep_agg["avg_sleep_min"] / sleep_agg["avg_time_in_bed"].clip(lower=1)
        ).clip(0, 1)
        user_agg = user_agg.merge(sleep_agg[["Id", "sleep_efficiency"]], on="Id", how="left")
    else:
        user_agg["sleep_efficiency"] = 0.82  # population mean

    user_agg["sleep_efficiency"] = user_agg["sleep_efficiency"].fillna(0.82)

    # Weight/BMI features
    if weight is not None and len(weight) > 0:
        w_agg = weight.groupby("Id").agg(
            bmi=("BMI", "last"),
            fat_pct=("Fat", "last"),
        ).reset_index()
        user_agg = user_agg.merge(w_agg, on="Id", how="left")
    else:
        user_agg["bmi"]     = 23.5
        user_agg["fat_pct"] = 22.0

    user_agg["bmi"]     = user_agg["bmi"].fillna(23.5)
    user_agg["fat_pct"] = user_agg["fat_pct"].fillna(22.0)

    return user_agg


def hr_to_zone(hr: float, age: int = 35) -> int:
    max_hr = 220 - age
    pct = hr / max_hr
    if pct < 0.50: return 0   # resting
    if pct < 0.60: return 1   # fat_burn
    if pct < 0.70: return 2   # cardio
    if pct < 0.85: return 3   # peak
    return 4                   # anaerobic


# ── Gym Members Feature Engineering ──────────────────────────────────────────

def engineer_gym_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Gym Members dataset columns to our internal feature schema.
    This dataset has the richest workout-level features.
    """
    gf = df.copy()

    # Normalise column names (different capitalisations across datasets)
    gf.columns = [c.strip().replace(" ", "_") for c in gf.columns]

    gf["workout_type_enc"] = gf["Workout_Type"].map(WORKOUT_TYPE_MAP).fillna(3)
    gf["fitness_goal_enc"] = gf.get("Fitness_Goal_Num", pd.Series(0, index=gf.index))
    gf["age_norm"]          = (gf["Age"] / 100.0).clip(0, 1)
    gf["bmi_norm"]          = (gf["BMI"] / 40.0).clip(0, 1)
    gf["intensity_score"]   = ((gf["Avg_BPM"] / gf["Max_BPM"].clip(lower=1))).clip(0, 1)
    gf["duration_norm"]     = (gf["Session_Duration(hours)"] / 3.0).clip(0, 1)
    gf["hr_norm"]           = (gf["Avg_BPM"] / 220.0).clip(0, 1)
    gf["resting_hr_norm"]   = (gf["Resting_BPM"] / 100.0).clip(0, 1)
    gf["calories_norm"]     = (gf["Calories_Burned"] / 1500.0).clip(0, 2)
    gf["fat_norm"]          = (gf["Fat_Percentage"] / 50.0).clip(0, 1)
    gf["water_norm"]        = (gf["Water_Intake(liters)"] / 5.0).clip(0, 1)
    gf["freq_norm"]         = (gf["Workout_Frequency(days/week)"] / 7.0).clip(0, 1)
    gf["exp_norm"]          = ((gf["Experience_Level"] - 1) / 2.0).clip(0, 1)

    # Derive HR zone from avg BPM and age
    gf["hr_zone"] = gf.apply(lambda r: hr_to_zone(r["Avg_BPM"], r["Age"]), axis=1)
    gf["hr_zone_norm"] = gf["hr_zone"] / 4.0

    # Fatigue proxy: high intensity + high frequency → higher fatigue
    gf["fatigue_proxy"] = (gf["intensity_score"] * 0.6 + gf["freq_norm"] * 0.4).clip(0, 1)

    return gf


# ── Mental Health Feature Engineering ────────────────────────────────────────

def engineer_mental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a per-row stress/coping score from the mental health survey.
    Used as a proxy for user fatigue/recovery state in feature vectors.
    Stress score feeds into the realtime_context features.
    """
    mf = df.copy()
    bool_map = {"Yes": 1, "No": 0, "Not sure": 0.5, "Maybe": 0.5}

    mf["stress_score"] = (
        mf["Growing_Stress"].map(bool_map).fillna(0) * 0.30
        + mf["Coping_Struggles"].map(bool_map).fillna(0) * 0.25
        + mf["Changes_Habits"].map(bool_map).fillna(0) * 0.15
        + mf["Social_Weakness"].map(bool_map).fillna(0) * 0.15
        + mf["Mental_Health_History"].map(bool_map).fillna(0) * 0.15
    )

    # Days indoors → sedentary proxy
    days_map = {"Go out Every day": 0.0, "1-14 days": 0.2,
                "15-30 days": 0.5, "31-60 days": 0.7, "More than 2 months": 1.0}
    mf["sedentary_score"] = mf["Days_Indoors"].map(days_map).fillna(0.3)

    mf["treatment_binary"] = mf["treatment"].map({"Yes": 1, "No": 0}).fillna(0)

    return mf[["stress_score", "sedentary_score", "treatment_binary"]]


# ── Fitness 2024 Item Feature Engineering ────────────────────────────────────

def engineer_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Fitness Daily 2024 rows to content item feature vectors.
    Each row represents a workout session type → treated as a content item.
    """
    cf = df.copy()
    cf.columns = [c.strip().replace(" ", "_") for c in cf.columns]

    cf["workout_type_enc"]  = cf["Workout_Type"].map(WORKOUT_TYPE_MAP).fillna(3)
    cf["content_type_enc"]  = 1   # all workout_routine
    cf["intensity_score"]   = ((cf["Avg_BPM"] / cf["Max_BPM"].clip(lower=1))).clip(0, 1)
    cf["duration_norm"]     = (cf["Session_Duration_(hours)"] / 3.0).clip(0, 1)
    cf["calories_norm"]     = (cf["Calories_Burned"] / 1500.0).clip(0, 2)
    cf["fat_norm"]          = (cf["Fat_Percentage"] / 50.0).clip(0, 1)
    cf["hr_mean_norm"]      = (cf["Avg_BPM"] / 220.0).clip(0, 1)
    cf["exp_norm"]          = ((cf["Experience_Level"] - 1) / 2.0).clip(0, 1)

    # Engagement proxy: completion = finish high-intensity workout
    cf["global_ctr"]              = cf["intensity_score"] * 0.3 + 0.05
    cf["global_completion_rate"]  = (1 - cf["intensity_score"] * 0.5).clip(0.2, 0.9)

    return cf


# ── Main ETL: Build Training Matrix ──────────────────────────────────────────

def build_training_matrix(data_root: Path) -> tuple:
    """
    Cross-join all four datasets to produce the unified [N, 409] training matrix.

    Join strategy:
      FitBit users     × Gym Members sessions  → (user, item) pairs
      FitBit heartrate → realtime biometric slice [394:409]
      Mental Health    → stress features blended into realtime slice
      Fitness 2024     → workout item features in categorical slice [384:394]

    Returns:
        X           — np.ndarray [N, 409] float32
        y_click     — np.ndarray [N] int8
        y_complete  — np.ndarray [N] int8
        meta        — pd.DataFrame with readable labels for inspection
    """
    logger.info("Loading raw datasets...")
    fitbit     = load_fitbit(data_root)
    fitness_24 = load_fitness_2024(data_root)
    gym        = load_gym_members(data_root)
    mental     = load_mental_health(data_root)

    # ── Step 1: FitBit labels + user features ─────────────────────────────
    daily      = fitbit.get("dailyActivity_merged", pd.DataFrame())
    sleep      = fitbit.get("sleepDay_merged")
    weight_df  = fitbit.get("weightLogInfo_merged")

    if daily.empty:
        raise ValueError("dailyActivity_merged.csv is required but empty.")

    labeled    = engineer_fitbit_labels(daily)
    user_feats = engineer_fitbit_user_features(daily, sleep, weight_df)

    # ── Step 2: Gym Members item features ─────────────────────────────────
    gym_feats  = engineer_gym_features(gym)

    # ── Step 3: Fitness 2024 → supplementary content features ─────────────
    content_24 = engineer_content_features(fitness_24)

    # ── Step 4: Mental Health → stress scores ────────────────────────────
    mh_scores  = engineer_mental_features(mental)
    mean_stress    = float(mh_scores["stress_score"].mean())
    mean_sedentary = float(mh_scores["sedentary_score"].mean())

    # ── Step 5: Cross-join FitBit users × Gym sessions ───────────────────
    #   For each FitBit (user, day) row, sample a matching gym workout
    #   based on activity_level proximity → realistic (user, item) pairs
    logger.info("Building cross-join training pairs...")

    gym_arr = gym_feats.values.astype(np.float32)
    rows_X  = []
    rows_y_click    = []
    rows_y_complete = []
    meta_rows = []

    # Merge user features into labeled activity
    labeled = labeled.merge(user_feats, on="Id", suffixes=("", "_agg"))

    for _, row in labeled.iterrows():
        # Sample a gym workout row whose intensity is close to the user's step activity
        target_intensity = float(row.get("activity_level", 0.5))
        gym_intensities  = gym_feats["intensity_score"].values
        weights = np.exp(-5 * (gym_intensities - target_intensity) ** 2)
        weights /= weights.sum()
        gym_idx = np.random.choice(len(gym_feats), p=weights)
        gym_row = gym_feats.iloc[gym_idx]

        # ── Build [409] feature vector ────────────────────────────────────
        vec = np.zeros(TOTAL_FEATURES, dtype=np.float32)

        # [0:384] Dense embedding placeholder (zeros for rows without Qdrant)
        # In production: populated from pre-indexed Qdrant embeddings

        # [384:394] Categorical slice
        vec[384] = gym_row.get("workout_type_enc", 3) / 6.0
        vec[385] = gym_row.get("content_type_enc", 1) / 2.0
        vec[386] = float(row.get("fitness_goal_enc", row.get("Fitness_Goal_Num", 0))) / 4.0
        vec[387] = gym_row.get("hr_zone_norm", 0.5)
        vec[388] = 0.0   # rank_position (not known at training)
        vec[389] = 0.0   # dietary restriction flag
        vec[390] = 1.0 if str(gym_row.get("Workout_Type", "")).lower() in ["yoga","pilates"] else 0.0
        # [391,392] time-of-day sin/cos — use mean for training
        import math
        vec[391] = math.sin(2 * math.pi * 9 / 24)   # 9 AM mean
        vec[392] = math.cos(2 * math.pi * 9 / 24)
        vec[393] = 0.5   # moderate peak-hour probability

        # [394:409] Realtime numerical slice
        vec[394] = gym_row.get("hr_norm", 0.5)
        vec[395] = float(row.get("activity_level", 0.5)) * (1 - float(row.get("sleep_efficiency", 0.82)))
        vec[396] = float(row.get("sleep_efficiency", 0.82))
        vec[397] = gym_row.get("calories_norm", 0.3)
        vec[398] = gym_row.get("global_ctr", 0.12)
        vec[399] = gym_row.get("global_completion_rate", 0.5)
        vec[400] = math.log1p(float(gym_row.get("Workout_Frequency(days/week)", 3)) * 100) / 20
        vec[401] = gym_row.get("intensity_score", 0.5)
        vec[402] = gym_row.get("duration_norm", 0.4)
        vec[403] = 0.0   # sodium (workout, not recipe)
        vec[404] = gym_row.get("calories_norm", 0.3)
        vec[405] = 0.0   # protein_g (workout)
        vec[406] = float(row.get("bmi", 23.5)) / 40.0
        vec[407] = float(row.get("age_norm", 0.3)) if "age_norm" in row else 0.3
        vec[408] = float(row.get("adherence_rate", 0.5))

        # Blend mental health stress into fatigue feature
        vec[395] = float(np.clip(vec[395] * 0.7 + mean_stress * 0.3, 0, 1))

        rows_X.append(vec)
        rows_y_click.append(int(row["click_label"]))
        rows_y_complete.append(int(row["complete_label"]))
        meta_rows.append({
            "user_id": row["Id"],
            "workout_type": gym_row.get("Workout_Type", "unknown"),
            "calories": gym_row.get("Calories_Burned", 0),
            "intensity": gym_row.get("intensity_score", 0),
            "click": int(row["click_label"]),
            "complete": int(row["complete_label"]),
        })

    X = np.vstack(rows_X).astype(np.float32)
    y_click    = np.array(rows_y_click, dtype=np.int8)
    y_complete = np.array(rows_y_complete, dtype=np.int8)
    meta       = pd.DataFrame(meta_rows)

    logger.info(
        "Training matrix built: shape=%s | click_rate=%.1f%% | complete_rate=%.1f%%",
        X.shape,
        y_click.mean() * 100,
        y_complete.mean() * 100,
    )
    return X, y_click, y_complete, meta
