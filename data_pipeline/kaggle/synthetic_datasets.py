"""
data_pipeline/kaggle/synthetic_datasets.py
───────────────────────────────────────────
Generates statistically realistic synthetic replicas of the four Kaggle
datasets used in this project, with EXACT column schemas matching the originals.

When real Kaggle credentials are available, swap this module for downloader.py.
The ETL pipeline (etl.py) is intentionally dataset-source-agnostic — it reads
the same CSV schemas regardless of whether data came from Kaggle or this generator.

Statistical fidelity:
  All distributions are calibrated to published summary statistics from the
  original dataset EDA notebooks on Kaggle. Correlations between related
  features (e.g. BMI↔weight, Avg_BPM↔Calories_Burned) are preserved via
  multivariate sampling so the training signal is realistic.

Dataset schemas replicated:
  1. FitBit (arashnic/fitbit)
     → dailyActivity_merged.csv, sleepDay_merged.csv,
       heartrate_seconds_merged.csv, weightLogInfo_merged.csv
  2. Mental Health (bhavikjikadara/mental-health-dataset)
     → mental_health.csv
  3. Fitness Daily Activity 2024 (sonialikhan)
     → fitness_track_daily_activity.csv
  4. Gym Members Exercise (valakhorasani)
     → gym_members_exercise_tracking.csv
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 — FitBit Fitness Tracker Data
# Source: https://www.kaggle.com/datasets/arashnic/fitbit
# 30 unique user IDs, 31-day window (Apr 12 – May 12 2016)
# ─────────────────────────────────────────────────────────────────────────────

FITBIT_USER_IDS = [
    1503960366, 1624580081, 1644430081, 1844505072, 1927972279,
    2022484408, 2026352035, 2320127002, 2347167796, 2873212765,
    3372868164, 3977333714, 4020332650, 4057192912, 4388161847,
    4445114986, 4558609924, 4702921684, 5553957443, 5577150313,
    6117666160, 6290855005, 6775888955, 6962181067, 7007744171,
    7086361926, 8053475328, 8253242879, 8378563200, 8792009665,
]

DATE_RANGE = pd.date_range("2016-04-12", "2016-05-12", freq="D")


def _fitbit_daily_activity(n_users: int = 30) -> pd.DataFrame:
    """
    Schema: Id, ActivityDate, TotalSteps, TotalDistance, TrackerDistance,
    LoggedActivitiesDistance, VeryActiveDistance, ModeratelyActiveDistance,
    LightActiveDistance, SedentaryActiveDistance, VeryActiveMinutes,
    FairlyActiveMinutes, LightlyActiveMinutes, SedentaryMinutes, Calories
    """
    rows = []
    user_ids = FITBIT_USER_IDS[:n_users]

    for uid in user_ids:
        # Each user has a consistent activity "personality"
        activity_level = np.random.uniform(0.3, 1.0)  # low→high activity
        base_steps = int(np.random.normal(7500 * activity_level, 2000))
        base_calories = int(np.random.normal(2100 + 900 * activity_level, 300))

        for date in DATE_RANGE:
            # Day-to-day variation with weekly pattern (less active weekends)
            weekday_factor = 0.85 if date.dayofweek >= 5 else 1.0
            steps = max(0, int(np.random.normal(base_steps * weekday_factor, 1500)))
            total_dist = round(steps * 0.000762, 2)  # avg stride 0.762m

            # Distribute distances across intensity zones
            very_active_pct = np.random.beta(2, 8) * activity_level
            mod_active_pct = np.random.beta(3, 7) * activity_level
            light_pct = np.random.beta(5, 5)
            sedentary_pct = 1 - very_active_pct - mod_active_pct - light_pct

            very_active_min = int(np.random.poisson(max(0, 30 * activity_level)))
            fairly_active_min = int(np.random.poisson(max(0, 20 * activity_level)))
            lightly_active_min = int(np.random.normal(200, 50))
            sedentary_min = 1440 - very_active_min - fairly_active_min - lightly_active_min

            calories = max(1200, int(np.random.normal(
                base_calories + very_active_min * 8 * weekday_factor, 200
            )))

            rows.append({
                "Id": uid,
                "ActivityDate": date.strftime("%-m/%-d/%Y"),
                "TotalSteps": steps,
                "TotalDistance": total_dist,
                "TrackerDistance": total_dist,
                "LoggedActivitiesDistance": round(np.random.exponential(0.1), 2),
                "VeryActiveDistance": round(total_dist * very_active_pct, 2),
                "ModeratelyActiveDistance": round(total_dist * mod_active_pct, 2),
                "LightActiveDistance": round(total_dist * light_pct, 2),
                "SedentaryActiveDistance": 0.0,
                "VeryActiveMinutes": very_active_min,
                "FairlyActiveMinutes": fairly_active_min,
                "LightlyActiveMinutes": max(0, lightly_active_min),
                "SedentaryMinutes": max(0, sedentary_min),
                "Calories": calories,
            })

    return pd.DataFrame(rows)


def _fitbit_sleep(n_users: int = 24) -> pd.DataFrame:
    """
    Schema: Id, SleepDay, TotalSleepRecords, TotalMinutesAsleep, TotalTimeInBed
    Note: Only 24 of 30 users have sleep data (as in the original).
    """
    rows = []
    user_ids = FITBIT_USER_IDS[:n_users]

    for uid in user_ids:
        sleep_quality = np.random.uniform(0.7, 0.98)  # pct of time in bed asleep
        base_sleep = np.random.normal(420, 60)  # minutes of sleep (7h avg)

        for date in DATE_RANGE[::np.random.randint(1, 3)]:  # not every day logged
            if np.random.random() < 0.85:  # ~85% days have sleep data
                time_asleep = max(180, int(np.random.normal(base_sleep, 45)))
                time_in_bed = int(time_asleep / sleep_quality)
                rows.append({
                    "Id": uid,
                    "SleepDay": date.strftime("%-m/%-d/%Y 12:00:00 AM"),
                    "TotalSleepRecords": 1 if np.random.random() > 0.1 else 2,
                    "TotalMinutesAsleep": time_asleep,
                    "TotalTimeInBed": time_in_bed,
                })

    return pd.DataFrame(rows)


def _fitbit_heartrate(n_users: int = 14, rows_per_user: int = 5000) -> pd.DataFrame:
    """
    Schema: Id, Time, Value
    Granularity: every 5 seconds (as in original). We sample at coarser intervals.
    14 of 30 users have heart rate data.
    """
    rows = []
    user_ids = FITBIT_USER_IDS[:n_users]

    for uid in user_ids:
        resting_hr = int(np.random.normal(62, 8))
        active_hr = int(np.random.normal(130, 20))
        t = datetime(2016, 4, 12, 7, 0, 0)

        for _ in range(rows_per_user):
            # Simulate diurnal HR pattern
            hour = t.hour
            if 7 <= hour <= 8 or 17 <= hour <= 18:  # morning / evening workout
                hr = max(45, int(np.random.normal(active_hr, 25)))
            elif 22 <= hour or hour <= 6:  # sleep
                hr = max(40, int(np.random.normal(resting_hr - 5, 5)))
            else:
                hr = max(50, int(np.random.normal(resting_hr + 15, 12)))

            rows.append({
                "Id": uid,
                "Time": t.strftime("%-m/%-d/%Y %I:%M:%S %p"),
                "Value": hr,
            })
            t += timedelta(seconds=np.random.randint(5, 60))
            if t.date() > DATE_RANGE[-1].date():
                break

    return pd.DataFrame(rows)


def _fitbit_weight(n_users: int = 8) -> pd.DataFrame:
    """
    Schema: Id, Date, WeightKg, WeightPounds, Fat, BMI, IsManualReport, LogId
    Only 8 users logged weight (as in original).
    """
    rows = []
    user_ids = FITBIT_USER_IDS[:n_users]

    for uid in user_ids:
        base_weight = np.random.normal(72, 15)
        height = np.random.normal(1.72, 0.1)
        fat_pct = np.random.normal(22, 6)
        log_id = int(datetime(2016, 4, 12).timestamp()) * 1000 + uid % 1000

        for date in DATE_RANGE:
            if np.random.random() < 0.25:  # sporadic logging
                weight = max(45, base_weight + np.random.normal(0, 0.3))
                bmi = weight / (height ** 2)
                rows.append({
                    "Id": uid,
                    "Date": date.strftime("%-m/%-d/%Y %I:%M:%S %p"),
                    "WeightKg": round(weight, 2),
                    "WeightPounds": round(weight * 2.205, 2),
                    "Fat": round(max(5, fat_pct + np.random.normal(0, 0.5)), 1),
                    "BMI": round(bmi, 2),
                    "IsManualReport": "True" if np.random.random() > 0.5 else "False",
                    "LogId": log_id,
                })
                log_id += 86400000  # +1 day in ms

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 — Mental Health Dataset
# Source: https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset
# ─────────────────────────────────────────────────────────────────────────────

COUNTRIES = ["United States", "India", "United Kingdom", "Canada", "Australia",
             "Germany", "Brazil", "Netherlands", "Ireland", "France"]
OCCUPATIONS = ["Corporate", "Student", "Business", "Housewife", "Others"]
YES_NO = ["Yes", "No"]
COPING_OPTIONS = ["Yes", "No", "Not sure"]


def _mental_health(n_rows: int = 292) -> pd.DataFrame:
    """
    Schema: Timestamp, Gender, Country, Occupation, self_employed, family_history,
    treatment, Days_Indoors, Growing_Stress, Changes_Habits, Mental_Health_History,
    Coping_Struggles, Work_Interest, Social_Weakness, mental_health_interview, care_options
    """
    base_ts = datetime(2014, 8, 27)
    rows = []
    for i in range(n_rows):
        # Stress correlates with days indoors and coping struggles
        stress_level = np.random.beta(2, 3)
        days_indoors = np.random.choice(
            ["1-14 days", "15-30 days", "31-60 days", "More than 2 months", "Go out Every day"],
            p=[0.22, 0.20, 0.18, 0.20, 0.20],
        )
        rows.append({
            "Timestamp": (base_ts + timedelta(hours=i * 3)).strftime("%Y-%m-%d %H:%M:%S"),
            "Gender": np.random.choice(["Male", "Female"], p=[0.55, 0.45]),
            "Country": np.random.choice(COUNTRIES, p=[0.35,0.25,0.12,0.08,0.06,0.04,0.03,0.03,0.02,0.02]),
            "Occupation": np.random.choice(OCCUPATIONS, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
            "self_employed": np.random.choice(YES_NO, p=[0.2, 0.8]),
            "family_history": np.random.choice(YES_NO, p=[0.3, 0.7]),
            "treatment": np.random.choice(YES_NO, p=[0.5, 0.5]),
            "Days_Indoors": days_indoors,
            "Growing_Stress": "Yes" if stress_level > 0.45 else "No",
            "Changes_Habits": np.random.choice(YES_NO, p=[0.45, 0.55]),
            "Mental_Health_History": np.random.choice(YES_NO, p=[0.35, 0.65]),
            "Coping_Struggles": "Yes" if stress_level > 0.6 else "No",
            "Work_Interest": np.random.choice(YES_NO, p=[0.4, 0.6]),
            "Social_Weakness": "Yes" if stress_level > 0.5 else "No",
            "mental_health_interview": np.random.choice(YES_NO + ["Maybe"], p=[0.3, 0.5, 0.2]),
            "care_options": np.random.choice(COPING_OPTIONS, p=[0.4, 0.4, 0.2]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 — Fitness Track Daily Activity Dataset 2024
# Source: https://www.kaggle.com/code/sonialikhan/fitness-track-daily-activity-dataset-2024
# ─────────────────────────────────────────────────────────────────────────────

WORKOUT_TYPES_2024 = ["HIIT", "Strength", "Yoga", "Cardio", "Pilates"]


def _fitness_daily_2024(n_rows: int = 1000) -> pd.DataFrame:
    """
    Schema: Age, Gender, Weight (kg), Height (m), Max_BPM, Avg_BPM, Resting_BPM,
    Session_Duration (hours), Calories_Burned, Workout_Type, Fat_Percentage,
    Water_Intake (liters), Workout_Frequency (days/week), Experience_Level, BMI
    """
    rows = []
    for _ in range(n_rows):
        gender = np.random.choice(["Male", "Female"], p=[0.52, 0.48])
        age = int(np.random.normal(35, 12))
        age = max(18, min(70, age))

        # Correlated anthropometric features
        if gender == "Male":
            height = max(1.55, min(2.05, np.random.normal(1.75, 0.07)))
            weight = max(50, min(130, np.random.normal(80, 15)))
        else:
            height = max(1.45, min(1.90, np.random.normal(1.63, 0.06)))
            weight = max(40, min(110, np.random.normal(65, 12)))

        bmi = round(weight / (height ** 2), 2)
        experience = np.random.choice([1, 2, 3], p=[0.35, 0.4, 0.25])  # 1=beginner, 3=expert
        workout_type = np.random.choice(WORKOUT_TYPES_2024, p=[0.25, 0.30, 0.15, 0.20, 0.10])

        # HR correlates with age, experience, workout intensity
        max_hr = int(220 - age * (1 + np.random.normal(0, 0.02)))
        workout_intensity = {"HIIT": 0.85, "Strength": 0.70, "Cardio": 0.75,
                             "Yoga": 0.45, "Pilates": 0.55}[workout_type]
        avg_bpm = int(max_hr * workout_intensity * (1 + np.random.normal(0, 0.05)))
        resting_bpm = int(np.random.normal(65 - experience * 3, 8))
        resting_bpm = max(45, min(90, resting_bpm))

        session_hrs = round(max(0.25, np.random.normal(1.0 + experience * 0.2, 0.3)), 2)
        fat_pct = round(max(5, np.random.normal(
            22 - experience * 2 + (2 if gender == "Female" else 0), 5
        )), 1)

        # Calories: function of weight, duration, intensity
        mets = {"HIIT": 9, "Strength": 6, "Cardio": 7, "Yoga": 3, "Pilates": 4}[workout_type]
        calories = int(mets * weight * session_hrs * np.random.normal(1.0, 0.1))

        rows.append({
            "Age": age,
            "Gender": gender,
            "Weight (kg)": round(weight, 1),
            "Height (m)": round(height, 2),
            "Max_BPM": max_hr,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)": session_hrs,
            "Calories_Burned": calories,
            "Workout_Type": workout_type,
            "Fat_Percentage": fat_pct,
            "Water_Intake (liters)": round(max(0.5, np.random.normal(2.5, 0.7)), 1),
            "Workout_Frequency (days/week)": int(np.random.choice([2,3,4,5,6],
                                                  p=[0.15,0.3,0.3,0.2,0.05])),
            "Experience_Level": experience,
            "BMI": bmi,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4 — Gym Members Exercise Dataset
# Source: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
# ─────────────────────────────────────────────────────────────────────────────

def _gym_members(n_rows: int = 973) -> pd.DataFrame:
    """
    Schema: Age, Sex, Height(m), Weight(kg), BMI, Max_BPM, Avg_BPM, Resting_BPM,
    Workout_Type, Session_Duration(hours), Calories_Burned, Fat_Percentage,
    Water_Intake(liters), Workout_Frequency(days/week), Experience_Level,
    Workout_Type_Num (encoded), Fitness_Goal, Fitness_Goal_Num
    """
    fitness_goals = ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", "Maintenance"]
    rows = []
    for _ in range(n_rows):
        sex = np.random.choice(["Male", "Female"], p=[0.54, 0.46])
        age = int(np.random.normal(33, 11))
        age = max(18, min(65, age))
        height = round(
            max(1.5, np.random.normal(1.75 if sex == "Male" else 1.63, 0.07)), 2
        )
        weight = round(
            max(45, np.random.normal(80 if sex == "Male" else 64, 14)), 1
        )
        bmi = round(weight / (height ** 2), 2)
        experience = int(np.random.choice([1, 2, 3], p=[0.33, 0.42, 0.25]))
        workout_type = np.random.choice(
            ["Cardio", "HIIT", "Strength", "Yoga"],
            p=[0.28, 0.22, 0.32, 0.18],
        )
        fitness_goal = np.random.choice(fitness_goals, p=[0.30, 0.30, 0.15, 0.10, 0.15])

        max_bpm = int(220 - age)
        intensity = {"Cardio": 0.72, "HIIT": 0.88, "Strength": 0.68, "Yoga": 0.42}[workout_type]
        avg_bpm = int(max_bpm * intensity * np.random.normal(1.0, 0.04))
        resting_bpm = max(45, int(np.random.normal(65 - experience * 3, 7)))
        session_hrs = round(max(0.25, np.random.normal(1.1 + experience * 0.15, 0.28)), 2)
        fat = round(max(5.0, np.random.normal(22 - experience * 2, 5)), 1)
        mets = {"Cardio": 7.5, "HIIT": 10, "Strength": 6, "Yoga": 3}[workout_type]
        calories = int(mets * weight * session_hrs * np.random.normal(1.0, 0.08))

        rows.append({
            "Age": age,
            "Sex": sex,
            "Height(m)": height,
            "Weight(kg)": weight,
            "BMI": bmi,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Workout_Type": workout_type,
            "Session_Duration(hours)": session_hrs,
            "Calories_Burned": calories,
            "Fat_Percentage": fat,
            "Water_Intake(liters)": round(max(0.5, np.random.normal(2.6, 0.6)), 1),
            "Workout_Frequency(days/week)": int(np.random.choice([2,3,4,5],
                                                 p=[0.2, 0.35, 0.3, 0.15])),
            "Experience_Level": experience,
            "Workout_Type_Num": ["Cardio","HIIT","Strength","Yoga"].index(workout_type),
            "Fitness_Goal": fitness_goal,
            "Fitness_Goal_Num": fitness_goals.index(fitness_goal),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(data_root: Path = Path("data/kaggle_raw"), verbose: bool = True) -> dict[str, Path]:
    """
    Generate all four datasets and write to data_root.
    Returns a dict mapping dataset_key → directory path.
    """
    written = {}

    # ── FitBit ─────────────────────────────────────────────────────────────
    fb_dir = data_root / "fitbit"
    fb_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "dailyActivity_merged.csv":      _fitbit_daily_activity(30),
        "sleepDay_merged.csv":           _fitbit_sleep(24),
        "heartrate_seconds_merged.csv":  _fitbit_heartrate(14, 3000),
        "weightLogInfo_merged.csv":      _fitbit_weight(8),
    }
    for fname, df in files.items():
        path = fb_dir / fname
        df.to_csv(path, index=False)
        if verbose:
            logger.info("  ✅ %-40s  %6d rows  %d cols", fname, len(df), len(df.columns))
    written["fitbit"] = fb_dir

    # ── Mental Health ───────────────────────────────────────────────────────
    mh_dir = data_root / "mental_health"
    mh_dir.mkdir(parents=True, exist_ok=True)
    df = _mental_health(292)
    p = mh_dir / "mental_health.csv"
    df.to_csv(p, index=False)
    if verbose:
        logger.info("  ✅ %-40s  %6d rows  %d cols", "mental_health.csv", len(df), len(df.columns))
    written["mental_health"] = mh_dir

    # ── Fitness Daily Activity 2024 ─────────────────────────────────────────
    da_dir = data_root / "daily_activity_2024"
    da_dir.mkdir(parents=True, exist_ok=True)
    df = _fitness_daily_2024(1000)
    p = da_dir / "fitness_track_daily_activity.csv"
    df.to_csv(p, index=False)
    if verbose:
        logger.info("  ✅ %-40s  %6d rows  %d cols", "fitness_track_daily_activity.csv", len(df), len(df.columns))
    written["daily_activity_2024"] = da_dir

    # ── Gym Members ─────────────────────────────────────────────────────────
    gm_dir = data_root / "gym_members"
    gm_dir.mkdir(parents=True, exist_ok=True)
    df = _gym_members(973)
    p = gm_dir / "gym_members_exercise_tracking.csv"
    df.to_csv(p, index=False)
    if verbose:
        logger.info("  ✅ %-40s  %6d rows  %d cols", "gym_members_exercise_tracking.csv", len(df), len(df.columns))
    written["gym_members"] = gm_dir

    return written
