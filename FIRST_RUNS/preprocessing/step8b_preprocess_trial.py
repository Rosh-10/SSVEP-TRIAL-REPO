"""
Step 8B: Preprocessing logic for a single trial (64 channels, 1500 samples).

Input: raw trial (64, 1500)
Output: preprocessed trial (64, 1500)

Transforms:
  1. Remove channel-wise DC offset
  2. Zero-phase bandpass filter (6–65 Hz via sosfiltfilt)
  3. Channel-wise z-score normalization

Note: This script runs from preprocessing/ folder but imports config.py from repo root.
"""

import numpy as np
from scipy.signal import sosfiltfilt
import sys
from pathlib import Path

# Add repo root to path (go up one level from preprocessing/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def preprocess_trial(trial, sos):
    """
    Preprocess one trial across all 64 channels.
    
    Parameters
    ----------
    trial : np.ndarray, shape (64, 1500)
        Raw EEG trial
    sos : np.ndarray, shape (N, 6)
        Butterworth SOS matrix from butter(..., output="sos")
    
    Returns
    -------
    np.ndarray, shape (64, 1500), dtype float32
        Preprocessed trial (DC-removed, filtered, normalized)
    """
    
    trial = trial.astype(np.float32)
    
    # --------------------------------------------------------
    # 1. Remove channel-wise DC offset
    # --------------------------------------------------------
    trial = trial - np.mean(trial, axis=1, keepdims=True)
    
    # --------------------------------------------------------
    # 2. Zero-phase bandpass filtering
    # --------------------------------------------------------
    filtered = sosfiltfilt(sos, trial, axis=1)
    
    # --------------------------------------------------------
    # 3. Channel-wise z-score normalization
    # --------------------------------------------------------
    mean = np.mean(filtered, axis=1, keepdims=True)
    std = np.std(filtered, axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)  # Avoid division by zero
    
    normalized = (filtered - mean) / std
    
    return normalized.astype(np.float32)


if __name__ == "__main__":
    # Test: load one trial and preprocess
    from config import BENCHMARK_DIR, FILTER_ORDER, LOW_CUTOFF, HIGH_CUTOFF, FS
    from scipy.io import loadmat
    from scipy.signal import butter
    
    print("Testing preprocessing on one trial...")
    
    # Load raw trial
    mat = loadmat(str(BENCHMARK_DIR / "S1.mat"))
    trial_raw = mat["data"][:, :, 0, 0]  # S1, Target 1, Block 1
    
    print(f"Raw trial shape: {trial_raw.shape}, dtype: {trial_raw.dtype}")
    print(f"Raw: mean={trial_raw.mean():.3f}, std={trial_raw.std():.3f}, "
          f"min={trial_raw.min():.3f}, max={trial_raw.max():.3f}")
    
    # Design filter
    sos = butter(FILTER_ORDER, [LOW_CUTOFF, HIGH_CUTOFF], 
                 btype="bandpass", fs=FS, output="sos")
    
    # Preprocess
    trial_proc = preprocess_trial(trial_raw, sos)
    
    print(f"Processed trial shape: {trial_proc.shape}, dtype: {trial_proc.dtype}")
    print(f"Processed: mean={trial_proc.mean():.3f}, std={trial_proc.std():.3f}, "
          f"min={trial_proc.min():.3f}, max={trial_proc.max():.3f}")
    print("✓ Test passed")