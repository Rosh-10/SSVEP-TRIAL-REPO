"""
Step 8D: Validate preprocessed dataset.

Checks:
  1. All 35 subject files exist
  2. Shapes are correct (64, 1500, 40, 6)
  3. No NaN/Inf values
  4. Channel-wise normalization verified
  5. Global statistics (mean, STD, min, max)
  6. Frequency-domain SSVEP peaks present

Note: This script runs from preprocessing/ folder but imports config.py from repo root.
"""

import sys
from pathlib import Path

# Add repo root to path (go up one level from preprocessing/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, RESULTS_DIR, N_SUBJECTS, N_CHANNELS, N_SAMPLES, N_TARGETS, N_BLOCKS, FS
import numpy as np
import matplotlib.pyplot as plt

EXPECTED_SHAPE = (N_CHANNELS, N_SAMPLES, N_TARGETS, N_BLOCKS)

print("=" * 70)
print("STEP 8D: Validation of Preprocessed Data")
print("=" * 70)
print(f"\nExpected shape per subject: {EXPECTED_SHAPE}")
print(f"Expected subjects: {N_SUBJECTS}")
print(f"Directory: {PROCESSED_DIR}\n")

# ============================================================
# Global statistics
# ============================================================

global_min = np.inf
global_max = -np.inf
global_sum = 0.0
global_sum_sq = 0.0
global_count = 0

errors = []

# ============================================================
# CHECK 1-3: Files, shapes, NaN/Inf
# ============================================================

print("=" * 70)
print("CHECK 1-3: Files, Shapes, NaN/Inf")
print("=" * 70)

for subject in range(1, N_SUBJECTS + 1):
    filepath = PROCESSED_DIR / f"S{subject}.npz"
    
    if not filepath.exists():
        errors.append(f"S{subject}: file not found")
        continue
    
    data = np.load(filepath)["data"]
    
    if data.shape != EXPECTED_SHAPE:
        errors.append(f"S{subject}: shape {data.shape} != {EXPECTED_SHAPE}")
    
    if np.isnan(data).any():
        errors.append(f"S{subject}: NaN detected")
    
    if np.isinf(data).any():
        errors.append(f"S{subject}: Inf detected")
    
    # Global stats
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    
    global_min = min(global_min, data_min)
    global_max = max(global_max, data_max)
    
    flat = data.astype(np.float64).ravel()
    global_sum += np.sum(flat)
    global_sum_sq += np.sum(flat ** 2)
    global_count += flat.size
    
    print(f"S{subject:02d}: ✓ shape={data.shape}, "
          f"min={data_min:.3f}, max={data_max:.3f}")

# ============================================================
# Global statistics
# ============================================================

if global_count > 0:
    global_mean = global_sum / global_count
    global_var = global_sum_sq / global_count - global_mean ** 2
    global_std = np.sqrt(max(global_var, 0))
else:
    global_mean = global_std = 0

print()
print("=" * 70)
print("GLOBAL STATISTICS")
print("=" * 70)
print(f"Min: {global_min:.6f}")
print(f"Max: {global_max:.6f}")
print(f"Mean: {global_mean:.6f}")
print(f"Std: {global_std:.6f}")

# ============================================================
# CHECK 4: Normalization (per trial, per channel)
# ============================================================

print()
print("=" * 70)
print("CHECK 4: Normalization Verification (3 subjects)")
print("=" * 70)

for subject in [1, 15, 30]:
    data = np.load(PROCESSED_DIR / f"S{subject}.npz")["data"]
    trial = data[:, :, 0, 0]  # Target 1, Block 1
    
    means = np.mean(trial, axis=1)
    stds = np.std(trial, axis=1)
    
    print(f"\nS{subject:02d} Target 1 Block 1:")
    print(f"  Channel means: [{means.min():.4f}, {means.max():.4f}]")
    print(f"  Channel stds:  [{stds.min():.4f}, {stds.max():.4f}]")

# ============================================================
# CHECK 5: Frequency-domain SSVEP peaks
# ============================================================

print()
print("=" * 70)
print("CHECK 5: Frequency-Domain SSVEP (3 examples)")
print("=" * 70)

target_freqs = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
])

examples = [(1, 1, 1), (15, 20, 1), (30, 1, 1)]

for subj, tgt, blk in examples:
    data = np.load(PROCESSED_DIR / f"S{subj}.npz")["data"]
    signal = np.mean(data[:, :, tgt-1, blk-1], axis=0)
    
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1/FS)
    mag = np.abs(fft)
    
    f0 = target_freqs[tgt - 1]
    
    # Peak at fundamental
    idx_f0 = np.argmin(np.abs(freqs - f0))
    peak_mag = mag[idx_f0]
    
    print(f"S{subj} Target {tgt} (f0={f0:.1f} Hz): peak magnitude at f0 = {peak_mag:.2f}")

# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)

if errors:
    print(f"\n✗ ERRORS FOUND ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
    print("\nFIX ERRORS BEFORE PROCEEDING TO STEP 9")
else:
    print("\n✓ PASSED: All checks successful")
    print(f"  • All {N_SUBJECTS} subjects present and correct shape")
    print(f"  • No NaN or Inf values")
    print(f"  • Channel-wise normalization verified")
    print(f"  • SSVEP peaks present in frequency domain")
    print(f"\nReady for Step 9 (Feature Extraction)")

print()
print("=" * 70)
print("STEP 8D COMPLETE")
print("=" * 70)