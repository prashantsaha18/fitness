# FitAI — Streamlit Cloud Deployment Guide

## What You're Deploying

A 5-page AI-powered fitness app backed by a **DeepFM recommendation model**
(AUC 0.91) trained on four Kaggle datasets. All heavy computation is pre-baked
into the repo — Streamlit Cloud just serves the UI.

---

## Step 1 — Push the Code to GitHub

Unzip `fitai_streamlit_deploy.zip`, then:

```bash
# The zip extracts into a fitai/ folder
unzip fitai_streamlit_deploy.zip
cd fitai

# Initialise git
git init
git add .
git commit -m "Initial FitAI deployment"

# Create a new GitHub repo at https://github.com/new
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/fitai.git
git branch -M main
git push -u origin main
```

> **Important:** The `artifacts/` folder (model checkpoint + ONNX) and
> `data/kaggle_raw/` (CSV datasets) are committed to the repo so the app
> boots instantly without retraining. Total repo size: ~5 MB.

---

## Step 2 — Create the Streamlit Cloud App

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Fill in the form:

| Field | Value |
|-------|-------|
| Repository | `YOUR_USERNAME/fitai` |
| Branch | `main` |
| Main file path | `streamlit_app.py` ← **exactly this** |
| App URL | choose any slug, e.g. `fitai-demo` |

4. Click **"Deploy"**

---

## Step 3 — Watch the Build Log

First deploy takes **3–5 minutes** (pip installs ~220 MB of dependencies).
Subsequent deploys are instant (cached).

Expected build log:
```
[pip] Installing torch==2.3.0+cpu  ← CPU wheel ~220MB
[pip] Installing streamlit, plotly, scikit-learn...
[app] ⚡ Bootstrap: datasets already present
[app] ⚡ Bootstrap: model already present
[app] 🟢 FitAI ready
```

---

## File Structure Streamlit Cloud Reads

```
fitai/                          ← repo root
├── streamlit_app.py            ← ENTRY POINT (set this in App Settings)
├── requirements.txt            ← pip dependencies (CPU torch)
├── packages.txt                ← apt packages (libgomp1)
├── .streamlit/
│   └── config.toml             ← dark theme + server settings
├── frontend/
│   ├── app.py                  ← main dashboard (5 tabs)
│   ├── components/             ← shared theme, charts, inference
│   └── pages/                  ← 4 additional pages
├── ranking/model.py            ← canonical DeepFMRanker
├── artifacts/
│   ├── ranking_model_kaggle.pt ← trained checkpoint (pre-baked)
│   └── ranking_model.onnx      ← ONNX graph (pre-baked)
└── data/kaggle_raw/            ← 4 Kaggle datasets (pre-baked CSVs)
```

---

## Troubleshooting

### "Module not found" errors
The app uses `pip install -e .` (editable install via `setup.py`).
Streamlit Cloud runs `pip install -r requirements.txt` which does NOT do
an editable install. The `streamlit_app.py` handles this by adding the
repo root to `sys.path` before any imports.

If you see import errors, add this at the top of `streamlit_app.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```
(Already present — just confirming.)

### "Memory limit exceeded"
Streamlit Community Cloud has ~1 GB RAM. The app is designed to fit:
- `torch` CPU wheel: ~220 MB
- `scikit-learn` + `scipy`: ~80 MB
- Model in memory: ~6 MB
- App overhead: ~100 MB
- **Total: ~406 MB** — safely under the 1 GB limit.

If you hit memory issues, upgrade to Streamlit Cloud Teams ($25/mo, 2 GB RAM).

### App is slow on first page load
The `@st.cache_resource` decorators ensure the model is loaded once and
reused across all user sessions. Cold start (first visitor after deploy)
takes ~3 seconds. Subsequent loads are instant.

### "set_page_config can only be called once"
Already handled — `streamlit_app.py` calls it first, then monkey-patches
it to a no-op before importing `frontend.app`. If you see this error,
check that you haven't added a second `st.set_page_config()` call anywhere.

---

## Environment Variables (Optional)

Set these in Streamlit Cloud → **App Settings → Secrets**:

```toml
# .streamlit/secrets.toml format

[app]
jwt_secret = "your-secret-here"   # not needed for frontend-only mode

# Only needed if connecting to real NeonDB / Qdrant / Redis:
# [database]
# url = "postgresql+asyncpg://..."
# [redis]
# url = "redis://..."
# [qdrant]
# host = "..."
```

The frontend works fully without any secrets — it uses the pre-trained
model and synthetic data. Secrets are only needed if you wire up the
FastAPI backend (`api/main.py`).

---

## Local Development

```bash
cd fitai

# Install dependencies
pip install -r requirements.txt
pip install -e .              # editable install (eliminates sys.path hacks)

# Run locally
streamlit run streamlit_app.py
# → http://localhost:8501

# Or run individual pages directly
streamlit run frontend/app.py
```

---

## Pages Available After Deploy

| Page | URL path | Description |
|------|----------|-------------|
| 🏠 Dashboard | `/` | Live HR, radar profile, top-5 recommendations |
| ⚡ Live Session | `/?page=Live_Session` | Real-time workout tracker with interval timer |
| 🥗 Nutrition | `/?page=Nutrition` | TDEE calculator, macro tracker, hydration |
| 🔬 A/B Testing | `/?page=AB_Testing` | Thompson Sampling experiment dashboard |
| 🔭 Data Explorer | `/?page=Data_Explorer` | Interactive Kaggle dataset explorer |

---

*Built with DeepFM · TF-IDF+JL Embeddings · Streamlit · Plotly*
