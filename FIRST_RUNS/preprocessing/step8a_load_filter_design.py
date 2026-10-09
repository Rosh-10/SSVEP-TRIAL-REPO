"""
Step 8A: Load frequency/phase labels and design 6–65 Hz Butterworth filter.

Output: Saves filter (SOS) and frequencies to a pickle for Step 8C to load.

Note: This script runs from preprocessing/ folder but imports config.py from repo root.
"""

import sys
from pathlib import Path

# Add repo root to path (go up one level from preprocessing/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FREQ_PHASE_FILE, FILTER_ORDER, LOW_CUTOFF, HIGH_CUTOFF, FS, RESULTS_DIR
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter
import pickle

print("=" * 70)
print("STEP 8A: Load Freq/Phase & Design Filter")
print("=" * 70)

# ============================================================
# Load Freq_Phase.mat
# ============================================================

print(f"\nLoading Freq_Phase from: {FREQ_PHASE_FILE}")
mat = loadmat(str(FREQ_PHASE_FILE))
freqs = mat["freqs"].flatten()
phases = mat["phases"].flatten()

assert freqs.shape == (40,), f"Expected freqs shape (40,), got {freqs.shape}"
assert phases.shape == (40,), f"Expected phases shape (40,), got {phases.shape}"

print(f"✓ Loaded {len(freqs)} target frequencies")
print(f"  Range: {freqs.min():.1f} – {freqs.max():.1f} Hz")
print(f"  First 5: {freqs[:5]}")

# ============================================================
# Design Butterworth filter
# ============================================================

print(f"\nDesigning {FILTER_ORDER}-order Butterworth bandpass filter:")
print(f"  Low cutoff:  {LOW_CUTOFF} Hz")
print(f"  High cutoff: {HIGH_CUTOFF} Hz")
print(f"  Sampling rate: {FS} Hz")

sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)

print(f"✓ Filter designed (SOS matrix shape: {sos.shape})")

# ============================================================
# Save for Step 8C
# ============================================================

cache = {
    "freqs": freqs,
    "phases": phases,
    "sos": sos,
    "fs": FS,
    "filter_order": FILTER_ORDER,
    "low_cutoff": LOW_CUTOFF,
    "high_cutoff": HIGH_CUTOFF
}

cache_file = RESULTS_DIR / "step8a_cache.pkl"
with open(cache_file, "wb") as f:
    pickle.dump(cache, f)

print(f"\n✓ Cached to: {cache_file}")
print("\n" + "=" * 70)
print("STEP 8A COMPLETE")
print("=" * 70)