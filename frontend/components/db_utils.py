"""
frontend/components/db_utils.py
────────────────────────────────
Helper functions to fetch users, completed workouts, streaks, and content pool
from NeonDB (Postgres) synchronously for the Streamlit dashboard.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone, date
import uuid
import nest_asyncio
import streamlit as st

from sqlalchemy import select, func, and_
from data_pipeline.database import AsyncSessionLocal
from data_pipeline.schemas import User, UserInteraction, ContentItem

# Enable nested event loops for Streamlit thread environments
nest_asyncio.apply()


async def _get_all_users_async() -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).order_by(User.username))
        users = result.scalars().all()
        return [{
            "id": str(u.id),
            "username": u.username,
            "email": u.email,
            "age": u.age or 28,
            "weight_kg": u.weight_kg or 74.0,
            "height_cm": u.height_cm or 178.0,
            "fitness_goal": u.fitness_goal.value if u.fitness_goal else "maintenance",
            "is_hypertensive": u.is_hypertensive,
            "has_cardiac_risk": u.has_cardiac_risk,
            "has_diabetes": u.has_diabetes,
            "preferred_workout_types": u.preferred_workout_types or [],
        } for u in users]


async def _get_weekly_completed_async(user_id: str) -> int:
    async with AsyncSessionLocal() as session:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        user_uuid = uuid.UUID(user_id)
        stmt = select(func.count(UserInteraction.id)).where(
            and_(
                UserInteraction.user_id == user_uuid,
                UserInteraction.interaction_type == "complete",
                UserInteraction.created_at >= seven_days_ago
            )
        )
        result = await session.execute(stmt)
        return result.scalar() or 0


async def _get_user_streak_async(user_id: str) -> int:
    async with AsyncSessionLocal() as session:
        user_uuid = uuid.UUID(user_id)
        stmt = select(func.date(UserInteraction.created_at)).where(
            UserInteraction.user_id == user_uuid
        ).distinct().order_by(func.date(UserInteraction.created_at).desc())
        result = await session.execute(stmt)
        dates = [r[0] for r in result.all()]
        if not dates:
            return 0
        
        today = date.today()
        parsed_dates = []
        for d in dates:
            if isinstance(d, date):
                parsed_dates.append(d)
            elif isinstance(d, datetime):
                parsed_dates.append(d.date())
            else:
                try:
                    parsed_dates.append(datetime.strptime(str(d), "%Y-%m-%d").date())
                except Exception:
                    pass
        
        if not parsed_dates:
            return 0
            
        latest = parsed_dates[0]
        if (today - latest).days > 1:
            return 0
            
        streak = 0
        current_date = latest
        for d in parsed_dates:
            diff = (current_date - d).days
            if diff == 0:
                continue
            elif diff == 1:
                streak += 1
                current_date = d
            else:
                break
        return streak + 1


async def _get_db_content_pool_async() -> list[dict]:
    async with AsyncSessionLocal() as session:
        stmt = select(ContentItem).where(ContentItem.is_published == True)
        result = await session.execute(stmt)
        items = result.scalars().all()
        
        pool = []
        for item in items:
            t = "Cardio"
            if item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe":
                t = "Meal"
            elif item.workout_type:
                val = item.workout_type.value if hasattr(item.workout_type, "value") else str(item.workout_type)
                t = val.capitalize() if val else "Cardio"
                if val == "hiit":
                    t = "HIIT"
            
            duration = item.duration_minutes or 30
            calories = 0
            if item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe":
                calories = item.calories_kcal or 0
            else:
                calories = item.calories_burned_estimate or 0
                
            intensity = item.intensity_score or 0.5
            
            level = "Intermediate"
            if intensity > 0.8:
                level = "Advanced"
            elif intensity < 0.4:
                level = "Beginner"
                
            pool.append({
                "id": str(item.id),
                "title": item.title,
                "type": t,
                "duration": int(duration),
                "calories": int(calories),
                "intensity": float(intensity),
                "level": level,
                "sodium_mg": float(item.sodium_mg or 0.0) if (item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe") else None,
                "calories_kcal": float(item.calories_kcal or 0.0) if (item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe") else None,
                "protein_g": float(item.protein_g or 0.0) if (item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe") else None,
                "carbs_g": float(item.carbs_g or 0.0) if (item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe") else None,
                "fat_g": float(item.fat_g or 0.0) if (item.content_type == "meal_recipe" or item.content_type.value == "meal_recipe") else None,
                "dietary_tags": item.dietary_tags or [],
            })
        return pool


def run_async(coro):
    """Bridge to run coroutines synchronously in Streamlit's environment."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)


def get_all_users() -> list[dict]:
    return run_async(_get_all_users_async())


def get_weekly_completed(user_id: str) -> int:
    return run_async(_get_weekly_completed_async(user_id))


def get_user_streak(user_id: str) -> int:
    return run_async(_get_user_streak_async(user_id))


def get_db_content_pool() -> list[dict]:
    return run_async(_get_db_content_pool_async())


def init_session_state_defaults():
    """Initialise global defaults for biometric and user state."""
    if "db_users" not in st.session_state:
        try:
            st.session_state.db_users = get_all_users()
        except Exception:
            st.session_state.db_users = []
            
    if "db_pool" not in st.session_state:
        try:
            st.session_state.db_pool = get_db_content_pool()
        except Exception:
            st.session_state.db_pool = []
            
    defaults = {
        "name": "Alex",
        "age": 28,
        "weight": 74,
        "height_cm": 178,
        "goal": "Maintenance",
        "freq_per_week": 4,
        "htn": False,
        "cardiac": False,
        "diabetes": False,
        "streak": 5,
        "weekly_done": 2,
        "db_user_select": "Custom (Sliders)",
        "prev_db_user_select": "Custom (Sliders)",
        "hr_history": [72 for _ in range(30)],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_db_user_selector():
    """Render the dropdown user selector in the sidebar."""
    st.markdown("### 👤 Select DB User")
    db_usernames = [u["username"] for u in st.session_state.db_users]
    
    st.selectbox(
        "User Selection",
        ["Custom (Sliders)"] + db_usernames,
        key="db_user_select",
        label_visibility="collapsed"
    )
    
    if st.session_state.db_user_select != st.session_state.prev_db_user_select:
        username = st.session_state.db_user_select
        if username == "Custom (Sliders)":
            st.session_state.name = "Alex"
            st.session_state.age = 28
            st.session_state.weight = 74
            st.session_state.height_cm = 178
            st.session_state.goal = "Maintenance"
            st.session_state.freq_per_week = 4
            st.session_state.htn = False
            st.session_state.cardiac = False
            st.session_state.diabetes = False
            st.session_state.streak = 5
            st.session_state.weekly_done = 2
        else:
            user = next((u for u in st.session_state.db_users if u["username"] == username), None)
            if user:
                st.session_state.name = user["username"].split("_")[0].capitalize()
                st.session_state.age = int(user["age"])
                st.session_state.weight = int(user["weight_kg"])
                st.session_state.height_cm = int(user["height_cm"])
                
                db_goal = user["fitness_goal"]
                goal_mapping = {
                    "weight_loss": "Weight Loss",
                    "muscle_gain": "Muscle Gain",
                    "endurance": "Endurance",
                    "flexibility": "Flexibility",
                    "maintenance": "Maintenance"
                }
                st.session_state.goal = goal_mapping.get(db_goal, "Maintenance")
                st.session_state.htn = bool(user["is_hypertensive"])
                st.session_state.cardiac = bool(user["has_cardiac_risk"])
                st.session_state.diabetes = bool(user["has_diabetes"])
                
                # Fetch streak & completed workouts
                st.session_state.streak = get_user_streak(user["id"])
                st.session_state.weekly_done = get_weekly_completed(user["id"])
                st.session_state.freq_per_week = len(user.get("preferred_workout_types", [])) * 2 or 4
                
        st.session_state.prev_db_user_select = st.session_state.db_user_select
        st.rerun()
