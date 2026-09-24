"""
Project configuration: paths, parameters, constants.

Works cross-platform (Mac, Windows, Linux) with pathlib.
All code imports from here — single source of truth.
"""

from pathlib import Path

# ============================================================
# PROJECT STRUCTURE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = RAW_DIR / "benchmark"
PROCESSED_DIR = DATA_DIR / "processed"

# Results/outputs
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if missing
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# KEY DATA FILES
# ============================================================

FREQ_PHASE_FILE = BENCHMARK_DIR / "Freq_Phase.mat"
CHANNELS_LOC_FILE = BENCHMARK_DIR / "64-channels.loc"
SUB_INFO_FILE = BENCHMARK_DIR / "Sub_info.txt"

# ============================================================
# SIGNAL PROCESSING PARAMETERS
# ============================================================

FS = 250.0  # Sampling rate (Hz)

# Butterworth bandpass filter
FILTER_ORDER = 5
LOW_CUTOFF = 6.0  # High-pass cutoff (Hz)
HIGH_CUTOFF = 65.0  # Low-pass cutoff (Hz)

# ============================================================
# DATASET STRUCTURE
# ============================================================

N_SUBJECTS = 35
N_CHANNELS = 64
N_SAMPLES = 1500  # Per trial (6 seconds at 250 Hz)
N_TARGETS = 40
N_BLOCKS = 6

# ============================================================
# VALIDATION
# ============================================================

def verify_raw_data():
    """Check that raw benchmark data exists."""
    if not BENCHMARK_DIR.exists():
        raise FileNotFoundError(
            f"Benchmark data not found at {BENCHMARK_DIR}\n"
            f"Check DATA_ROOT in config.py or ensure data files are placed there."
        )
    
    missing = [f"S{i}.mat" for i in range(1, N_SUBJECTS + 1)
               if not (BENCHMARK_DIR / f"S{i}.mat").exists()]
    
    if missing:
        raise FileNotFoundError(f"Missing subjects: {missing}")
    
    return True


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BENCHMARK_DIR: {BENCHMARK_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print()
    print(f"Filter: {LOW_CUTOFF}–{HIGH_CUTOFF} Hz (order {FILTER_ORDER})")
    print(f"FS: {FS} Hz")
