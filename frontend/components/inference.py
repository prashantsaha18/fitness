"""
frontend/components/inference.py
──────────────────────────────────
Centralised model loading and inference for the Streamlit frontend.

All pages import from here — the DeepFM class lives ONLY in ranking/model.py.
This module:
  • Loads the trained checkpoint once via @st.cache_resource
  • Rebuilds the TF-IDF + Random Projection embedder (deterministic)
  • Exposes run_inference() for scoring a list of content items
  • Exposes build_feature_vec() for single-item feature construction
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

# Import the canonical DeepFM — never redefined here or in pages
from ranking.model import DeepFMRanker

ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifacts"
EMB_DIM      = 384
TOTAL_DIM    = 409


# ── Workout corpus for TF-IDF ─────────────────────────────────────────────────
_WORKOUT_CORPUS: dict[str, str] = {
    "HIIT": (
        "high intensity interval training hiit anaerobic sprint burst cardio "
        "metabolic conditioning fat loss calorie burn plyometrics tabata circuit "
        "explosive power speed agility endurance short rest periods"
    ),
    "Strength": (
        "strength training resistance hypertrophy muscle building compound lifts "
        "progressive overload barbell squat deadlift bench press pull up "
        "powerlifting bodybuilding power force tension mechanical load "
        "protein synthesis muscle mass volume intensity"
    ),
    "Yoga": (
        "yoga mindfulness flexibility balance posture alignment breathing pranayama "
        "sun salutation warrior poses meditation stress relief mental wellness "
        "recovery mobility joint health parasympathetic rest digest stretch fascia"
    ),
    "Cardio": (
        "cardio cardiovascular endurance aerobic running cycling rowing swimming "
        "steady state zone two fat adaptation vo2 max mitochondria aerobic base "
        "heart health blood pressure longevity distance pacing tempo marathon"
    ),
    "Pilates": (
        "pilates core stability postural alignment spine health pelvic floor "
        "deep core transverse abdominis mat reformer controlled movement "
        "low impact rehabilitation posture correction body control precision"
    ),
}


def _remap_state_dict(state: dict) -> dict:
    """
    Remap legacy checkpoint key names → canonical DeepFMRanker key names.

    Training scripts written before ranking/model.py was stabilised saved
    checkpoints with abbreviated attribute names (ls, lk, lin, ch, kh,
    deep.N.n.*).  This function makes those checkpoints loadable into the
    canonical DeepFMRanker without retraining.
    """
    KEY_MAP = {
        "ls":  "log_sigma_click",
        "lk":  "log_sigma_complete",
        "lin": "linear",
        "ch":  "click_head",
        "kh":  "complete_head",
    }
    remapped = {}
    for k, v in state.items():
        # Replace prefix (e.g. "lin.weight" → "linear.weight")
        new_key = k
        for old_pfx, new_pfx in KEY_MAP.items():
            if k == old_pfx or k.startswith(old_pfx + "."):
                new_key = new_pfx + k[len(old_pfx):]
                break
        # Replace ".n." → ".block." inside deep MLP blocks
        new_key = new_key.replace(".n.0.", ".block.0.").replace(".n.1.", ".block.1.")
        remapped[new_key] = v
    return remapped


@st.cache_resource   # no show_spinner — works both inside and outside Streamlit
def load_model() -> tuple[DeepFMRanker, Any, Any]:
    """
    Load the trained DeepFM checkpoint and rebuild the TF-IDF embedder.

    Returns:
        (model, tfidf_vectorizer, random_projection)

    Cache policy: resource-level cache — loaded once per Streamlit process.
    Falls back to MemoryCacheStorageManager when called outside Streamlit.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.random_projection import SparseRandomProjection

    # ── Build TF-IDF + Random Projection (deterministic) ─────────────────
    corpus = [text for text in _WORKOUT_CORPUS.values()] * 40
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, sublinear_tf=True)
    tfidf.fit(corpus)
    rp = SparseRandomProjection(n_components=EMB_DIM, random_state=42, density=0.1)
    rp.fit(tfidf.transform(corpus))

    # ── Load DeepFM checkpoint ────────────────────────────────────────────
    model = DeepFMRanker()
    ckpt_path = ARTIFACT_DIR / "ranking_model_kaggle.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(_remap_state_dict(state["model_state_dict"]), strict=False)
    model.eval()

    return model, tfidf, rp


def _embed(workout_type: str) -> np.ndarray:
    """Produce a 384-dim L2-normalised embedding for a workout type string."""
    _, tfidf, rp = load_model()
    text   = _WORKOUT_CORPUS.get(workout_type, _WORKOUT_CORPUS["Cardio"])
    sparse = tfidf.transform([text])
    vec    = rp.transform(sparse).toarray().astype(np.float32)[0]
    norm   = np.linalg.norm(vec)
    return vec / max(norm, 1e-8)


def build_feature_vec(workout: dict[str, Any], user_state: dict[str, Any]) -> np.ndarray:
    """
    Construct the canonical [409] feature vector for a (user, workout) pair.

    Layout mirrors ranking/features.py FeatureDims exactly:
      [0:384]   dense TF-IDF embedding
      [384:394] categorical features
      [394:409] realtime biometric features
    """
    wt_map = {"HIIT": 0, "Strength": 1, "Yoga": 2, "Cardio": 3, "Pilates": 4}
    hour   = datetime.now().hour

    vec = np.zeros(TOTAL_DIM, dtype=np.float32)

    # [0:384] embedding
    vec[:EMB_DIM] = _embed(workout.get("type", "Cardio"))

    # [384:394] categorical
    vec[384] = wt_map.get(workout.get("type", "Cardio"), 3) / 6.0
    vec[385] = 1.0 / 2.0                                              # workout_routine
    vec[386] = float(user_state.get("fitness_goal_enc", 0)) / 4.0
    vec[387] = float(user_state.get("hr_zone_id", 2)) / 4.0
    vec[388] = 0.0                                                     # rank_position (unknown)
    vec[389] = 0.0                                                     # dietary restriction flag
    vec[390] = 1.0 if workout.get("type", "") in ("Yoga", "Pilates") else 0.0
    vec[391] = math.sin(2.0 * math.pi * hour / 24.0)
    vec[392] = math.cos(2.0 * math.pi * hour / 24.0)
    vec[393] = 1.0 if (6 <= hour <= 9 or 17 <= hour <= 20) else 0.0

    # [394:409] realtime biometric + content stats
    vec[394] = min(float(user_state.get("hr_bpm", 70)) / 220.0, 1.0)
    vec[395] = float(user_state.get("fatigue", 0.3))
    vec[396] = float(user_state.get("recovery", 0.75))
    vec[397] = min(float(workout.get("calories", 300)) / 1500.0, 2.0)
    vec[398] = 0.12                                                    # global CTR placeholder
    vec[399] = max(0.3, 0.85 - workout.get("intensity", 0.5) * 0.4)
    vec[400] = math.log1p(float(user_state.get("freq_per_week", 3)) * 100.0) / 20.0
    vec[401] = float(workout.get("intensity", 0.5))
    vec[402] = min(float(workout.get("duration", 30)) / 120.0, 1.0)
    vec[403] = 0.0                                                     # sodium (not a recipe)
    vec[404] = min(float(workout.get("calories", 300)) / 1500.0, 2.0)
    vec[405] = 0.0                                                     # protein_g
    vec[406] = float(user_state.get("bmi", 23.5)) / 40.0
    vec[407] = float(user_state.get("age", 30)) / 100.0
    vec[408] = float(user_state.get("adherence", 0.65))

    return vec


@st.cache_data(ttl=30, show_spinner=False)
def run_inference(
    user_state_key: str,
    user_state_json: str,
    content_pool_json: str,
) -> list[dict[str, Any]]:
    """
    Score every item in content_pool against the current user state.

    Args:
        user_state_key:   Cache key (user fingerprint string — changes when state changes)
        user_state_json:  JSON-serialised user_state dict
        content_pool_json: JSON-serialised list of content dicts

    Returns:
        Content pool sorted by engagement score (descending), with 'score' key added.

    Cache strategy:
        ttl=30s balances freshness with inference cost.
        The cache key is the user_state_key, so a slider move triggers re-inference.
    """
    import json

    user_state   = json.loads(user_state_json)
    content_pool = json.loads(content_pool_json)
    model, _, _  = load_model()

    results: list[dict[str, Any]] = []
    for item in content_pool:
        vec   = build_feature_vec(item, user_state)
        x     = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            score = float(model.predict_proba(x)[0].item())

        # Safety adjustment: penalise high-intensity items for fatigued users
        fatigue = user_state.get("fatigue", 0.3)
        if fatigue > 0.65 and item.get("intensity", 0.5) > 0.75:
            score *= 1.0 - (fatigue - 0.65) * 0.8

        results.append({**item, "score": round(score, 6)})

    return sorted(results, key=lambda r: r["score"], reverse=True)
