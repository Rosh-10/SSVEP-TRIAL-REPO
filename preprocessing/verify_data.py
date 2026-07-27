"""
Run this after setting DATA_ROOT in config.py to confirm the Benchmark
dataset files are placed where the code expects them.

Usage:
    python preprocessing/verify_data.py
"""

import sys
from pathlib import Path

# Allow running this script directly (adds repo root to path so `config` imports)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import BENCHMARK_DIR

REQUIRED_SUPPORT_FILES = [
    "64-channels.loc",
    "Freq_Phase.mat",
    "Sub_info.txt",
    "readme.txt",
]


def verify_benchmark_data():
    print(f"Checking Benchmark data at: {BENCHMARK_DIR}\n")

    if not BENCHMARK_DIR.exists():
        print(f"ERROR: {BENCHMARK_DIR} does not exist. Check DATA_ROOT in config.py.")
        return

    expected_subjects = [f"S{i}.mat" for i in range(1, 36)]
    missing_subjects = [f for f in expected_subjects if not (BENCHMARK_DIR / f).exists()]

    missing_support = [f for f in REQUIRED_SUPPORT_FILES if not (BENCHMARK_DIR / f).exists()]

    if missing_subjects:
        print(f"Missing {len(missing_subjects)}/35 subject files: {missing_subjects}")
    else:
        print("All 35 subject .mat files found.")

    if missing_support:
        print(f"Missing support files: {missing_support}")
    else:
        print("All support files (loc, Freq_Phase, Sub_info, readme) found.")

    if not missing_subjects and not missing_support:
        print("\nData check PASSED. Ready for Step 6a (FFT verification).")
    else:
        print("\nData check FAILED. Fix paths/placement before proceeding.")


if __name__ == "__main__":
    verify_benchmark_data()
