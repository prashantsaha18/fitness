"""
scripts/seed_data.py
─────────────────────
Synthetic data seeding pipeline for NeonDB.

Generates statistically realistic user profiles and content items at scale.
Designed to be idempotent — safe to re-run; existing records are skipped.

Seeding targets:
  • 10,000  User records with varied health profiles and goals
  • 100,000 ContentItem records (videos, workouts, recipes)
  • 500,000 Interaction events (sparse engagement matrix, ~5 interactions/user)

Content distribution mirrors a real fitness platform catalogue:
  40% workout_routine | 35% video | 25% meal_recipe
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on the path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import settings
from data_pipeline.database import AsyncSessionLocal, init_db
from data_pipeline.schemas import (
    ContentItem,
    ContentType,
    FitnessGoal,
    InteractionType,
    User,
    UserInteraction,
    WorkoutType,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Synthetic Data Generators ─────────────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Neha",
    "Karan", "Pooja", "Dev", "Riya", "Aditya", "Nisha", "Rahul", "Kavya",
    "Alex", "Sarah", "Jordan", "Morgan", "Casey", "Taylor", "Jamie", "Riley",
]
LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Mehta", "Shah", "Joshi",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Martinez",
]

WORKOUT_TITLES = {
    WorkoutType.HIIT: [
        "30-Minute Fat-Torching HIIT Blast", "Tabata Protocol: Full Body Ignition",
        "HIIT Cardio Shred: Beginner to Advanced", "Athletic HIIT: Speed & Power Circuit",
        "Zero Equipment HIIT: Home Warrior Series", "HIIT Metabolic Conditioning Block",
    ],
    WorkoutType.STRENGTH: [
        "Progressive Overload: Compound Lifts", "Upper Body Hypertrophy Protocol",
        "Leg Day: Squat & Deadlift Focus", "Push-Pull-Legs: Full Program Week 1",
        "Functional Strength: Olympic Lifts Intro", "Bodyweight Strength: Calisthenics Fundamentals",
    ],
    WorkoutType.YOGA: [
        "Morning Sun Salutation Flow", "Yin Yoga: Deep Tissue Release 60min",
        "Power Yoga: Strength & Flexibility", "Restorative Yoga: Recovery & Calm",
        "Yoga for Athletes: Hip Flexor Focus", "Breathwork & Mindfulness Integration",
    ],
    WorkoutType.CARDIO: [
        "Zone 2 Steady State: Fat Adaptation", "Treadmill Interval Protocol",
        "Cycling LISS: 45-Min Endurance Ride", "Jump Rope Cardio: Footwork & Conditioning",
        "Rowing Machine Technique & Endurance", "Stairmaster Glute Activation Cardio",
    ],
    WorkoutType.PILATES: [
        "Core Stability: Pilates Fundamentals", "Reformer Flow: Intermediate Series",
        "Mat Pilates: Postural Correction", "Pilates for Back Pain Relief",
        "Athletic Pilates: Cross-Training Stability", "Pilates 100s & Classical Sequence",
    ],
    WorkoutType.RECOVERY: [
        "Active Recovery: Foam Rolling Protocol", "Mobility Work: Joint Health Routine",
        "Static Stretching: Full Body 20min", "Cold/Heat Contrast Recovery Guide",
        "Sleep & Recovery Optimisation Talk", "Breathwork for HRV Improvement",
    ],
}

RECIPE_TITLES = [
    "High-Protein Greek Chicken Bowl", "Overnight Oats: Muscle Recovery Formula",
    "Avocado Salmon Protein Wrap", "Quinoa Stir-Fry: Macro-Balanced Plate",
    "Low-Carb Egg White Frittata", "Sweet Potato & Black Bean Power Bowl",
    "Post-Workout Banana Protein Smoothie", "Mediterranean Tuna Salad Jar",
    "Lean Turkey Meatball Meal Prep", "Chia Seed Pudding: Omega-3 Boost",
    "Zucchini Noodles with Pesto Chicken", "Lentil Dal with Brown Rice",
    "Green Smoothie: Alkaline Reset", "Baked Cod with Roasted Vegetables",
    "Tempeh Stir-Fry: Plant Protein", "Acai Bowl with Granola & Berries",
    "Cottage Cheese Pancakes (25g Protein)", "Edamame & Brown Rice Onigiri",
]

VIDEO_TITLES = [
    "Understanding Periodisation for Hypertrophy", "Macros 101: Tracking for Body Composition",
    "How to Read a Nutrition Label", "Sleep Science: Optimising Recovery Windows",
    "VO2 Max Explained: Training Your Engine", "Injury Prevention: Prehab Essentials",
    "Progressive Overload: The Only Rule That Matters", "Heart Rate Zones Deep Dive",
    "Gut Health & Athletic Performance", "Supplement Stack: Evidence-Based Guide",
    "Breathing Mechanics for Performance", "Mental Performance: Flow State Training",
]

MUSCLE_GROUPS = ["quads", "hamstrings", "glutes", "chest", "back", "shoulders",
                 "biceps", "triceps", "core", "calves", "hip_flexors", "lats"]
EQUIPMENT = ["barbell", "dumbbells", "kettlebell", "resistance_bands",
             "pull_up_bar", "bench", "cable_machine", "none"]
DIETARY_TAGS = ["high-protein", "low-carb", "keto", "vegan", "vegetarian",
                "gluten-free", "dairy-free", "paleo", "low-sodium", "nut-free"]


def _random_user() -> dict:
    """Generate a statistically realistic user profile."""
    age = random.randint(18, 65)
    height_cm = random.gauss(170, 12)
    weight_kg = random.gauss(75, 18)
    bmi = weight_kg / ((height_cm / 100) ** 2)

    # Health markers — realistic prevalence rates
    is_hypertensive = random.random() < 0.28       # 28% global prevalence
    has_cardiac_risk = random.random() < 0.07
    has_diabetes = random.random() < 0.11

    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    username = f"{name.replace(' ', '_').lower()}_{random.randint(100, 9999)}"

    return {
        "id": str(uuid.uuid4()),
        "username": username,
        "email": f"{username}@example.com",
        "hashed_password": "$2b$12$placeholder_hashed_password_for_seed_data",
        "age": age,
        "weight_kg": round(weight_kg, 1),
        "height_cm": round(height_cm, 1),
        "fitness_goal": random.choice(list(FitnessGoal)).value,
        "preferred_workout_types": random.sample(
            [w.value for w in WorkoutType], k=random.randint(1, 3)
        ),
        "is_hypertensive": is_hypertensive,
        "has_cardiac_risk": has_cardiac_risk,
        "has_diabetes": has_diabetes,
        "dietary_restrictions": {
            tag: True for tag in random.sample(DIETARY_TAGS, k=random.randint(0, 2))
        },
        "last_active_at": datetime.now(timezone.utc) - timedelta(
            hours=random.randint(0, 72)
        ),
        "is_active": True,
    }


def _random_workout(workout_type: WorkoutType) -> dict:
    titles = WORKOUT_TITLES[workout_type]
    intensity = random.uniform(0.2, 0.95)
    duration = random.choice([15, 20, 30, 45, 60, 75, 90])

    return {
        "id": str(uuid.uuid4()),
        "title": random.choice(titles) + f" #{random.randint(1, 99)}",
        "description": (
            f"A {_intensity_label(intensity)} {workout_type.value} session "
            f"targeting {', '.join(random.sample(MUSCLE_GROUPS, k=2))}. "
            f"Duration: {duration} minutes. "
            f"This evidence-based protocol is designed to optimise "
            f"{'strength and power' if intensity > 0.7 else 'endurance and aerobic capacity'}. "
            f"Suitable for {'advanced' if intensity > 0.8 else 'intermediate'} athletes."
        ),
        "content_type": ContentType.WORKOUT_ROUTINE.value,
        "workout_type": workout_type.value,
        "duration_minutes": duration,
        "intensity_score": round(intensity, 3),
        "calories_burned_estimate": round(intensity * duration * 8.5, 1),
        "target_muscle_groups": random.sample(MUSCLE_GROUPS, k=random.randint(2, 5)),
        "required_equipment": random.sample(EQUIPMENT, k=random.randint(0, 3)),
        "global_ctr": round(random.betavariate(2, 5), 4),
        "global_completion_rate": round(random.betavariate(3, 3), 4),
        "total_interactions": random.randint(50, 50000),
        "is_published": True,
    }


def _random_recipe() -> dict:
    has_sodium_issue = random.random() < 0.3
    sodium_mg = (
        random.uniform(600, 2400) if has_sodium_issue
        else random.uniform(50, 400)
    )
    protein_g = random.uniform(15, 60)
    carbs_g = random.uniform(5, 120)
    fat_g = random.uniform(5, 45)
    calories = (protein_g * 4) + (carbs_g * 4) + (fat_g * 9)

    return {
        "id": str(uuid.uuid4()),
        "title": random.choice(RECIPE_TITLES) + f" (v{random.randint(1, 5)})",
        "description": (
            f"A nutrient-dense meal delivering {protein_g:.0f}g protein "
            f"and {calories:.0f} kcal. "
            f"Prep time: {random.choice([5, 10, 15, 20, 30])} minutes. "
            f"Designed for {'muscle recovery' if protein_g > 35 else 'sustained energy'}. "
            f"Dietary tags: {', '.join(random.sample(DIETARY_TAGS, k=2))}."
        ),
        "content_type": ContentType.MEAL_RECIPE.value,
        "sodium_mg": round(sodium_mg, 1),
        "calories_kcal": round(calories, 1),
        "protein_g": round(protein_g, 1),
        "carbs_g": round(carbs_g, 1),
        "fat_g": round(fat_g, 1),
        "dietary_tags": random.sample(DIETARY_TAGS, k=random.randint(1, 4)),
        "global_ctr": round(random.betavariate(2, 5), 4),
        "global_completion_rate": round(random.betavariate(4, 2), 4),
        "total_interactions": random.randint(20, 20000),
        "is_published": True,
    }


def _random_video() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": random.choice(VIDEO_TITLES) + f" | Ep.{random.randint(1, 50)}",
        "description": (
            "An evidence-based educational deep-dive for serious athletes. "
            f"Runtime: {random.choice([8, 12, 15, 20, 25, 35])} minutes. "
            "Backed by peer-reviewed research and expert practitioner insights. "
            "Subtitles available in 12 languages."
        ),
        "content_type": ContentType.VIDEO.value,
        "duration_minutes": random.choice([8, 12, 15, 20, 25, 35]),
        "intensity_score": 0.1,  # videos are low-intensity
        "global_ctr": round(random.betavariate(3, 4), 4),
        "global_completion_rate": round(random.betavariate(2, 4), 4),
        "total_interactions": random.randint(100, 100000),
        "is_published": True,
    }


def _intensity_label(score: float) -> str:
    if score < 0.3: return "low-intensity"
    if score < 0.6: return "moderate-intensity"
    if score < 0.8: return "high-intensity"
    return "extreme-intensity"


# ── Seeding Functions ─────────────────────────────────────────────────────────

async def seed_users(session, count: int = 10_000) -> list[str]:
    """
    Batch-insert users using PostgreSQL upsert (ON CONFLICT DO NOTHING).
    Returns list of user UUIDs for interaction seeding.
    """
    logger.info("Seeding %d users...", count)

    # Check existing count
    existing = (await session.execute(select(func.count(User.id)))).scalar()
    if existing >= count:
        logger.info("Users already seeded (%d records). Skipping.", existing)
        result = await session.execute(select(User.id).limit(count))
        return [str(r[0]) for r in result.all()]

    remaining = count - existing
    user_records = [_random_user() for _ in range(remaining)]

    BATCH = 500
    inserted_ids = []
    for i in range(0, len(user_records), BATCH):
        batch = user_records[i : i + BATCH]
        stmt = pg_insert(User).values(batch).on_conflict_do_nothing(
            index_elements=["email"]
        )
        await session.execute(stmt)
        inserted_ids.extend(r["id"] for r in batch)
        if (i + BATCH) % 2000 == 0:
            await session.flush()
            logger.info("  Users: %d / %d", min(i + BATCH, remaining), remaining)

    await session.flush()
    logger.info("✅ Seeded %d new user records", len(inserted_ids))

    result = await session.execute(select(User.id).limit(count))
    return [str(r[0]) for r in result.all()]


async def seed_content(session, count: int = 100_000) -> list[str]:
    """
    Batch-insert content items with realistic distribution:
    40% workouts, 35% videos, 25% recipes.
    """
    logger.info("Seeding %d content items...", count)

    existing = (await session.execute(select(func.count(ContentItem.id)))).scalar()
    if existing >= count:
        logger.info("Content already seeded (%d records). Skipping.", existing)
        result = await session.execute(select(ContentItem.id).limit(count))
        return [str(r[0]) for r in result.all()]

    remaining = count - existing
    n_workouts = int(remaining * 0.40)
    n_videos = int(remaining * 0.35)
    n_recipes = remaining - n_workouts - n_videos

    content_records = []
    workout_types = list(WorkoutType)
    for _ in range(n_workouts):
        content_records.append(_random_workout(random.choice(workout_types)))
    for _ in range(n_videos):
        content_records.append(_random_video())
    for _ in range(n_recipes):
        content_records.append(_random_recipe())

    random.shuffle(content_records)

    all_keys = {
        "id": None,
        "title": None,
        "description": None,
        "content_type": None,
        "workout_type": None,
        "duration_minutes": None,
        "intensity_score": None,
        "calories_burned_estimate": None,
        "target_muscle_groups": None,
        "required_equipment": None,
        "sodium_mg": None,
        "calories_kcal": None,
        "protein_g": None,
        "carbs_g": None,
        "fat_g": None,
        "dietary_tags": None,
        "global_ctr": 0.0,
        "global_completion_rate": 0.0,
        "total_interactions": 0,
        "is_published": True,
    }
    content_records = [{**all_keys, **r} for r in content_records]

    BATCH = 1000
    for i in range(0, len(content_records), BATCH):
        batch = content_records[i : i + BATCH]
        stmt = pg_insert(ContentItem).values(batch).on_conflict_do_nothing(
            index_elements=["id"]
        )
        await session.execute(stmt)
        if (i + BATCH) % 10_000 == 0:
            await session.flush()
            logger.info("  Content: %d / %d", min(i + BATCH, remaining), remaining)

    await session.flush()
    logger.info("✅ Seeded %d content items", remaining)

    result = await session.execute(select(ContentItem.id).limit(count))
    return [str(r[0]) for r in result.all()]


async def seed_interactions(
    session,
    user_ids: list[str],
    content_ids: list[str],
    count: int = 500_000,
) -> None:
    """
    Generate a realistic sparse interaction matrix.
    Interaction types follow a power-law distribution:
      ~50% click, ~20% complete, ~20% skip, ~10% save/share/abandon
    """
    existing = (
        await session.execute(select(func.count(UserInteraction.id)))
    ).scalar()
    if existing >= count:
        logger.info("Interactions already seeded (%d records). Skipping.", existing)
        return

    remaining = count - existing
    logger.info("Seeding %d interaction events...", remaining)

    interaction_weights = [
        (InteractionType.CLICK.value, 0.50),
        (InteractionType.COMPLETE.value, 0.20),
        (InteractionType.SKIP.value, 0.20),
        (InteractionType.SAVE.value, 0.05),
        (InteractionType.SHARE.value, 0.02),
        (InteractionType.ABANDON.value, 0.03),
    ]
    types, weights = zip(*interaction_weights)

    model_versions = ["deepfm_v1.0.0", "deepfm_v0.9.5", "ann_fallback_v0"]

    BATCH = 2000
    records = []
    for _ in range(remaining):
        interaction_type = random.choices(types, weights=weights, k=1)[0]
        completion_pct = (
            random.uniform(0.8, 1.0) if interaction_type == "complete"
            else random.uniform(0.0, 0.4) if interaction_type in ("skip", "abandon")
            else None
        )
        records.append({
            "id": str(uuid.uuid4()),
            "user_id": random.choice(user_ids),
            "content_id": random.choice(content_ids),
            "interaction_type": interaction_type,
            "session_id": str(uuid.uuid4())[:8],
            "rank_position": random.randint(1, 20),
            "dwell_time_seconds": random.uniform(5, 3600) if interaction_type != "skip" else random.uniform(0.5, 5),
            "completion_pct": round(completion_pct, 3) if completion_pct else None,
            "heart_rate_bpm": random.randint(60, 180) if random.random() > 0.4 else None,
            "fatigue_level": round(random.uniform(0.1, 0.9), 3) if random.random() > 0.5 else None,
            "active_calories": round(random.uniform(0, 800), 1) if random.random() > 0.5 else None,
            "model_version": random.choice(model_versions),
            "inference_score": round(random.uniform(0.1, 0.95), 4),
        })

        if len(records) >= BATCH:
            stmt = pg_insert(UserInteraction).values(records).on_conflict_do_nothing(
                index_elements=["id"]
            )
            await session.execute(stmt)
            await session.flush()
            records = []

    if records:
        stmt = pg_insert(UserInteraction).values(records).on_conflict_do_nothing(
            index_elements=["id"]
        )
        await session.execute(stmt)
        await session.flush()

    logger.info("✅ Seeded %d interaction events", remaining)


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def main(
    num_users: int = 10_000,
    num_content: int = 100_000,
    num_interactions: int = 500_000,
) -> None:
    logger.info("=" * 60)
    logger.info("FITNESS REC ENGINE — DATA SEEDING PIPELINE")
    logger.info("Target: %s", settings.DATABASE_URL.split("@")[-1])
    logger.info("=" * 60)

    await init_db()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            user_ids = await seed_users(session, count=num_users)
            content_ids = await seed_content(session, count=num_content)
            await seed_interactions(
                session, user_ids, content_ids, count=num_interactions
            )

    logger.info("=" * 60)
    logger.info("✅ SEEDING COMPLETE")
    logger.info("  Users:        %d", num_users)
    logger.info("  Content:      %d", num_content)
    logger.info("  Interactions: %d", num_interactions)
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed NeonDB with synthetic data")
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--content", type=int, default=100_000)
    parser.add_argument("--interactions", type=int, default=500_000)
    args = parser.parse_args()

    asyncio.run(
        main(
            num_users=args.users,
            num_content=args.content,
            num_interactions=args.interactions,
        )
    )
