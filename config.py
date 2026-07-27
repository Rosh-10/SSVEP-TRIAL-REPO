"""
Central configuration for local data paths.

Each collaborator should only need to edit DATA_ROOT (if their data lives
somewhere other than the default data/raw location relative to repo root).
Every other script should import paths FROM this file rather than hardcoding
strings, so the whole project stays portable across machines/OSes.
"""

from pathlib import Path

# Repo root = folder containing this config.py file
REPO_ROOT = Path(__file__).resolve().parent

# --- EDIT THIS if your extracted data lives somewhere else ---
DATA_ROOT = REPO_ROOT / "data" / "raw"
# ---------------------------------------------------------------

BENCHMARK_DIR = DATA_ROOT / "benchmark"
BETA_DIR = DATA_ROOT / "beta"

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
