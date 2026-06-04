"""
data_pipeline/kaggle/downloader.py
────────────────────────────────────
Kaggle Dataset Downloader — authenticates via Kaggle API and pulls all
four source datasets into a structured local cache.

Datasets acquired:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  1. FitBit Fitness Tracker (arashnic/fitbit)                             │
  │     30 users · 18 CSV files · Apr–May 2016                              │
  │     Key files: dailyActivity_merged, heartrate_seconds_merged,           │
  │                sleepDay_merged, weightLogInfo_merged                      │
  │                                                                          │
  │  2. Mental Health Dataset (bhavikjikadara/mental-health-dataset)         │
  │     Mental fitness scores, stress indicators, country-level wellness     │
  │                                                                          │
  │  3. Fitness Track Daily Activity 2024 (yaminh/fitness-track-...)         │
  │     973 rows · Age, BMI, BPM, Workout_Type, Calories_Burned, Experience │
  │                                                                          │
  │  4. Gym Members Exercise Dataset (valakhorasani/gym-members-exercise)    │
  │     973 rows · overlapping schema with Dataset 3 — used as test split    │
  └──────────────────────────────────────────────────────────────────────────┘

Authentication:
  Set KAGGLE_USERNAME and KAGGLE_KEY as environment variables, OR
  place ~/.kaggle/kaggle.json with {"username":"...","key":"..."}.

  Get your API key at: https://www.kaggle.com/settings → API → Create Token
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dataset Registry ──────────────────────────────────────────────────────────

DATASETS = {
    "fitbit": {
        "slug": "arashnic/fitbit",
        "target_dir": "fitbit",
        "required_files": [
            "dailyActivity_merged.csv",
            "sleepDay_merged.csv",
            "heartrate_seconds_merged.csv",
            "weightLogInfo_merged.csv",
            "hourlyCalories_merged.csv",
            "hourlySteps_merged.csv",
            "hourlyIntensities_merged.csv",
        ],
        "description": "FitBit Fitness Tracker Data — 30 users, biometric timeseries",
    },
    "mental_health": {
        "slug": "bhavikjikadara/mental-health-dataset",
        "target_dir": "mental_health",
        "required_files": ["mental_health.csv"],
        "description": "Mental Health Dataset — wellness scores, stress indicators",
    },
    "daily_activity_2024": {
        "slug": "yaminh/fitness-track-daily-activity-dataset",
        "target_dir": "daily_activity_2024",
        "required_files": ["fitness_track_daily_activity.csv"],
        "description": "Fitness Track Daily Activity 2024 — workout metrics, BMI, BPM",
    },
    "gym_members": {
        "slug": "valakhorasani/gym-members-exercise-dataset",
        "target_dir": "gym_members",
        "required_files": ["gym_members_exercise_tracking.csv"],
        "description": "Gym Members Exercise Tracking — workout type, calories, experience",
    },
}

DEFAULT_DATA_ROOT = Path("data/kaggle_raw")


# ── Downloader ────────────────────────────────────────────────────────────────

class KaggleDownloader:
    """
    Downloads and validates Kaggle datasets using the official kaggle-api client.

    Error handling:
      - Missing credentials → clear instructions printed, graceful exit
      - Partial download → re-download triggered (idempotent)
      - Missing required files → specific file names reported
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._api = None

    def _get_api(self):
        """Lazy-initialise the Kaggle API client."""
        if self._api is not None:
            return self._api
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
        except ImportError:
            raise RuntimeError(
                "kaggle package not installed.\n"
                "Run: pip install kaggle\n"
                "Then set KAGGLE_USERNAME and KAGGLE_KEY env vars,\n"
                "or place ~/.kaggle/kaggle.json."
            )

        # Validate credentials exist before initialising
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        has_file = kaggle_json.exists()

        if not has_env and not has_file:
            raise RuntimeError(
                "Kaggle credentials not found.\n\n"
                "Option 1: Set environment variables:\n"
                "  export KAGGLE_USERNAME=your_username\n"
                "  export KAGGLE_KEY=your_api_key\n\n"
                "Option 2: Download kaggle.json from kaggle.com/settings → API\n"
                "  and place it at ~/.kaggle/kaggle.json\n\n"
                "Get your API key at: https://www.kaggle.com/settings"
            )

        api = KaggleApiExtended()
        api.authenticate()
        self._api = api
        return api

    def download_dataset(
        self,
        dataset_key: str,
        force: bool = False,
    ) -> Path:
        """
        Download a single dataset by its registry key.

        Args:
            dataset_key: One of 'fitbit', 'mental_health', 'daily_activity_2024', 'gym_members'
            force: Re-download even if files exist

        Returns:
            Path to the dataset directory
        """
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS)}")

        spec = DATASETS[dataset_key]
        target_dir = self.data_root / spec["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if not force and self._is_complete(target_dir, spec["required_files"]):
            logger.info(
                "✅ %s already downloaded at %s", dataset_key, target_dir
            )
            return target_dir

        logger.info(
            "⬇️  Downloading %s (%s)...", dataset_key, spec["slug"]
        )
        api = self._get_api()

        # Download zip to temp location
        zip_path = target_dir / "download.zip"
        api.dataset_download_files(
            dataset=spec["slug"],
            path=str(target_dir),
            unzip=False,
            quiet=False,
        )

        # Find and extract the zip
        zips = list(target_dir.glob("*.zip"))
        if zips:
            logger.info("Extracting %s...", zips[0].name)
            with zipfile.ZipFile(zips[0], "r") as zf:
                # Flatten nested zip structure — some Kaggle datasets have subdirs
                for member in zf.namelist():
                    filename = Path(member).name
                    if filename and not member.endswith("/"):
                        source = zf.open(member)
                        target = target_dir / filename
                        with open(target, "wb") as f:
                            shutil.copyfileobj(source, f)
            zips[0].unlink()  # delete zip after extraction

        # Validate
        missing = self._get_missing_files(target_dir, spec["required_files"])
        if missing:
            logger.warning(
                "⚠️  Some expected files not found in %s: %s\n"
                "The dataset may have been restructured on Kaggle. "
                "Proceeding with available files.",
                dataset_key, missing
            )
        else:
            logger.info("✅ %s downloaded and validated", dataset_key)

        return target_dir

    def download_all(self, force: bool = False) -> dict[str, Path]:
        """Download all four datasets. Returns map of key → directory path."""
        results = {}
        for key in DATASETS:
            try:
                results[key] = self.download_dataset(key, force=force)
            except Exception as exc:
                logger.error("Failed to download %s: %s", key, exc)
                results[key] = None
        return results

    def _is_complete(self, directory: Path, required_files: list[str]) -> bool:
        return len(self._get_missing_files(directory, required_files)) == 0

    def _get_missing_files(
        self, directory: Path, required_files: list[str]
    ) -> list[str]:
        existing = {f.name for f in directory.iterdir()} if directory.exists() else set()
        return [f for f in required_files if f not in existing]

    def status(self) -> None:
        """Print download status for all datasets."""
        print("\n" + "=" * 60)
        print("  KAGGLE DATASET STATUS")
        print("=" * 60)
        for key, spec in DATASETS.items():
            target_dir = self.data_root / spec["target_dir"]
            missing = self._get_missing_files(target_dir, spec["required_files"])
            status_icon = "✅" if not missing else "❌"
            print(f"\n  {status_icon} {key}")
            print(f"     Slug:    {spec['slug']}")
            print(f"     Dir:     {target_dir}")
            if missing:
                print(f"     Missing: {missing}")
            else:
                files = list(target_dir.glob("*.csv"))
                total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
                print(f"     Files:   {len(files)} CSVs ({total_mb:.1f} MB)")
        print("=" * 60 + "\n")
