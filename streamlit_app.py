"""
streamlit_app.py  —  Streamlit Cloud entry point.

Set as "Main file path" in Streamlit Cloud app settings.
Local run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ── Page config must be the very first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="FitAI — Fitness Intelligence",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path bootstrap (editable install OR direct execution) ─────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("fitai.bootstrap")


@st.cache_resource(show_spinner=False)
def _bootstrap() -> dict[str, bool]:
    """
    First-run bootstrap: generate datasets + train model.
    Cached at resource level — runs once per Streamlit process lifetime.
    Returns a dict of what was generated.
    """
    status: dict[str, bool] = {"data": False, "model": False}

    data_csv = _ROOT / "data" / "kaggle_raw" / "fitbit" / "dailyActivity_merged.csv"
    if not data_csv.exists():
        _log.info("Generating synthetic Kaggle datasets...")
        from data_pipeline.kaggle.synthetic_datasets import generate_all
        generate_all(_ROOT / "data" / "kaggle_raw", verbose=False)
        status["data"] = True
        _log.info("Datasets ready.")

    model_pt = _ROOT / "artifacts" / "ranking_model_kaggle.pt"
    if not model_pt.exists():
        _log.info("Training DeepFM (5 epochs)...")
        subprocess.run(
            [sys.executable, "scripts/train_kaggle.py", "--epochs", "5"],
            cwd=str(_ROOT), check=False,
        )
        status["model"] = True
        _log.info("Model ready.")

    return status


# ── Run bootstrap with user-visible progress ──────────────────────────────────
_data_csv  = _ROOT / "data" / "kaggle_raw" / "fitbit" / "dailyActivity_merged.csv"
_model_pt  = _ROOT / "artifacts" / "ranking_model_kaggle.pt"
_need_boot = not _data_csv.exists() or not _model_pt.exists()

if _need_boot:
    with st.spinner("\u26a1 FitAI first-run setup (generating data + training model ~60s)..."):
        _bootstrap()
else:
    _bootstrap()   # populates cache instantly — no spinner needed

# ── Hand off to main app ──────────────────────────────────────────────────────
# Import the app module.  Because streamlit_app.py already called
# set_page_config(), we patch app.py to skip its own call.
import importlib, types

# Load frontend.app without executing the top-level set_page_config call.
# We monkey-patch st.set_page_config to a no-op for this import only.
_real_spc = st.set_page_config
st.set_page_config = lambda **_kw: None   # suppress duplicate call
try:
    import frontend.app   # noqa: F401 — runs the Streamlit page
finally:
    st.set_page_config = _real_spc        # restore
