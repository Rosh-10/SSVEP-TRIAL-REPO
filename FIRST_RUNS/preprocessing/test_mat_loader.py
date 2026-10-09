"""
Step 6a - Part 1: Confirm which loader works for the Benchmark .mat files.

MATLAB .mat files come in different internal formats:
- v7.2 and earlier: scipy.io.loadmat works directly
- v7.3+ (HDF5-based): scipy.io.loadmat fails; need mat73 or h5py instead

We don't know yet which format this dataset uses - this script finds out.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BENCHMARK_DIR

s1_path = BENCHMARK_DIR / "S1.mat"

print(f"Attempting to load: {s1_path}\n")

# --- Attempt 1: scipy ---
try:
    import scipy.io as sio
    data = sio.loadmat(str(s1_path))
    print("SUCCESS with scipy.io.loadmat")
    print("Keys found:", list(data.keys()))

    eeg = data['data']
    print("\nShape of data['data']:", eeg.shape)
    print("Expected from readme.txt: (64, 1500, 40, 6)")
    print("Dtype:", eeg.dtype)
    print("Match:", eeg.shape == (64, 1500, 40, 6))

    print("\nUse scipy.io.loadmat going forward.")
except NotImplementedError as e:
    print("scipy.io.loadmat FAILED (likely v7.3/HDF5 format):", e)
    print("\nTrying mat73 instead...\n")

    import mat73
    data = mat73.loadmat(str(s1_path))
    print("SUCCESS with mat73.loadmat")
    print("Keys found:", list(data.keys()))
    print("\nUse mat73.loadmat going forward.")
except Exception as e:
    print("scipy.io.loadmat FAILED with unexpected error:", type(e).__name__, e)