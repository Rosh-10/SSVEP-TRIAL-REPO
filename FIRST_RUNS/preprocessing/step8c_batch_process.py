"""
Step 8C: Batch process all 35 subjects.

Loads filter/freqs from Step 8A cache, applies preprocessing to each trial,
saves compressed .npz files for each subject.

Note: This script runs from preprocessing/ folder but imports config.py from repo root.
"""

import sys
from pathlib import Path

# Add repo root to path (go up one level from preprocessing/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BENCHMARK_DIR, PROCESSED_DIR, N_SUBJECTS, RESULTS_DIR
from step8b_preprocess_trial import preprocess_trial
from scipy.io import loadmat
import numpy as np
import pickle

print("=" * 70)
print("STEP 8C: Batch Preprocessing All Subjects")
print("=" * 70)

# ============================================================
# Load cache from Step 8A
# ============================================================

cache_file = RESULTS_DIR / "step8a_cache.pkl"
print(f"\nLoading cache from Step 8A: {cache_file}")

with open(cache_file, "rb") as f:
    cache = pickle.load(f)

sos = cache["sos"]
print(f"✓ Loaded SOS filter matrix (shape: {sos.shape})")

# ============================================================
# Process all subjects
# ============================================================

for subject in range(1, N_SUBJECTS + 1):
    
    # Load raw data
    filepath = BENCHMARK_DIR / f"S{subject}.mat"
    print(f"\nS{subject:02d}: Loading {filepath.name}...", end=" ")
    
    mat = loadmat(str(filepath))
    data_raw = mat["data"]
    
    assert data_raw.shape == (64, 1500, 40, 6), f"Unexpected shape: {data_raw.shape}"
    print(f"shape={data_raw.shape}")
    
    # Allocate output
    data_proc = np.zeros_like(data_raw, dtype=np.float32)
    
    # Process every target and block
    for target in range(40):
        for block in range(6):
            trial_raw = data_raw[:, :, target, block]
            data_proc[:, :, target, block] = preprocess_trial(trial_raw, sos)
    
    # Save as compressed .npz
    output_file = PROCESSED_DIR / f"S{subject}.npz"
    np.savez_compressed(output_file, data=data_proc)
    
    print(f"        → Saved: {output_file.name}")

print("\n" + "=" * 70)
print(f"STEP 8C COMPLETE: All {N_SUBJECTS} subjects processed")
print(f"Output directory: {PROCESSED_DIR}")
print("=" * 70)