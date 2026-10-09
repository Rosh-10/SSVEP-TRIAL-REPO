"""
Step 9a: Filter Bank Feature Extraction
========================================
Extract power features from preprocessed EEG using a 9-subband filter bank.

Objective:
  - Design 9 subbands (6–65 Hz range)
  - Extract power features: (64 channels) × (9 subbands) per trial
  - Test on one trial (S01, target 1, block 1)
  - Visualize heatmap + bar chart
  - Sanity checks

Output:
  - features shape: (64, 9)
  - Plots: heatmap, bar chart
  - Validation: no NaN, power > 0
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt
import sys
from pathlib import Path

# Import config from project root (parent of feature_extraction/)
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.py'
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.py not found at {CONFIG_PATH}\nMake sure script is in feature_extraction/ subdirectory")

# Load config module
import importlib.util
spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

PROJECT_ROOT = config.PROJECT_ROOT
PROCESSED_DIR = config.PROCESSED_DIR
RESULTS_DIR = config.RESULTS_DIR

# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================

INPUT_DIR = PROCESSED_DIR           # Read from preprocessed data
OUTPUT_DIR = PROJECT_ROOT / 'feature_extraction'  # Write to feature extraction
RESULTS_DIR = RESULTS_DIR           # Save plots & logs

# Subband definitions: (f_low, f_high) in Hz
SUBBANDS = [
    (6, 10),
    (10, 14),
    (14, 18),
    (18, 22),
    (22, 26),
    (26, 30),
    (30, 34),
    (34, 42),
    (42, 65),
]
N_SUBBANDS = len(SUBBANDS)
N_CHANNELS = 64
SAMPLING_RATE = 250  # Hz (downsampled from 1000 Hz in preprocessing)
FILTER_ORDER = 5  # Butterworth order


# ============================================================================
# FILTER BANK DESIGN
# ============================================================================

def design_bandpass_filter(f_low, f_high, fs, order=5):
    """
    Design a bandpass filter using Butterworth.
    
    Args:
        f_low (float): Low cutoff frequency (Hz)
        f_high (float): High cutoff frequency (Hz)
        fs (float): Sampling rate (Hz)
        order (int): Filter order
    
    Returns:
        sos: Second-order sections (SOS) format for numerical stability
    """
    nyquist = fs / 2
    normalized_low = f_low / nyquist
    normalized_high = f_high / nyquist
    
    # Clamp to (0, 1) to avoid scipy warnings
    normalized_low = max(0.001, min(0.999, normalized_low))
    normalized_high = max(0.001, min(0.999, normalized_high))
    
    sos = butter(order, [normalized_low, normalized_high], btype='band', output='sos')
    return sos


def create_filterbank(subbands, fs=250, order=5):
    """
    Create a filter bank: list of SOS matrices, one per subband.
    
    Args:
        subbands: List of (f_low, f_high) tuples
        fs (float): Sampling rate
        order (int): Filter order
    
    Returns:
        filterbank: List of SOS matrices
    """
    filterbank = []
    for f_low, f_high in subbands:
        sos = design_bandpass_filter(f_low, f_high, fs, order)
        filterbank.append(sos)
    return filterbank


def compute_power(signal):
    """
    Compute power as mean-squared amplitude.
    
    Args:
        signal: 1D array of time-series data
    
    Returns:
        power: Scalar (float)
    """
    return np.mean(signal ** 2)


def extract_filterbank_features(trial_data, filterbank, subbands, fs=250):
    """
    Extract filter bank power features from a single trial.
    
    Args:
        trial_data: Shape (64, 1500) — raw EEG for one trial
        filterbank: List of SOS matrices (one per subband)
        subbands: List of (f_low, f_high) tuples (for reference/logging)
        fs: Sampling rate (Hz)
    
    Returns:
        features: Shape (64, 9) — power per channel per subband
    
    Process:
        For each of 64 channels:
            For each of 9 subbands:
                1. Apply bandpass filter to channel
                2. Compute power (mean-squared)
        Stack into (64, 9) array
    """
    n_channels, n_samples = trial_data.shape
    n_subbands = len(filterbank)
    
    features = np.zeros((n_channels, n_subbands))
    
    for ch_idx in range(n_channels):
        channel_signal = trial_data[ch_idx, :]
        
        for sb_idx, sos in enumerate(filterbank):
            # Zero-phase filtering (filtfilt)
            filtered_signal = sosfiltfilt(sos, channel_signal)
            
            # Compute power
            power = compute_power(filtered_signal)
            features[ch_idx, sb_idx] = power
    
    return features


# ============================================================================
# SANITY CHECKS
# ============================================================================

def validate_features(features, name="features"):
    """
    Sanity checks on extracted features.
    
    Args:
        features: Shape (64, 9) or similar
        name: Name for logging
    """
    print(f"\n--- Validating {name} ---")
    print(f"Shape: {features.shape}")
    print(f"Data type: {features.dtype}")
    print(f"NaN count: {np.isnan(features).sum()}")
    print(f"Inf count: {np.isinf(features).sum()}")
    print(f"Min value: {np.nanmin(features):.6e}")
    print(f"Max value: {np.nanmax(features):.6e}")
    print(f"Mean value: {np.nanmean(features):.6e}")
    print(f"Std value: {np.nanstd(features):.6e}")
    
    # Check for positivity (power should be non-negative)
    if np.any(features < 0):
        print(f"⚠️  WARNING: {np.sum(features < 0)} negative power values detected!")
    else:
        print("✓ All power values non-negative")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_filterbank_features(features, subbands, subject_id="S01", target_id=1, block_id=1):
    """
    Visualize filter bank power features.
    
    Args:
        features: Shape (64, 9)
        subbands: List of (f_low, f_high)
        subject_id, target_id, block_id: For plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Heatmap: 64 channels × 9 subbands ---
    ax = axes[0]
    im = ax.imshow(features, aspect='auto', cmap='viridis')
    ax.set_xlabel('Subband Index')
    ax.set_ylabel('Channel Index')
    subband_labels = [f"{f_low}-{f_high}" for f_low, f_high in subbands]
    ax.set_xticks(range(len(subbands)))
    ax.set_xticklabels(subband_labels, rotation=45, fontsize=8)
    ax.set_title(f'Filter Bank Power Heatmap\n({subject_id}, Target {target_id}, Block {block_id})')
    plt.colorbar(im, ax=ax, label='Power (μV²)')
    
    # --- Bar chart: mean power per subband ---
    ax = axes[1]
    mean_power_per_subband = np.mean(features, axis=0)
    subband_labels_short = [f"{f_low}-{f_high}" for f_low, f_high in subbands]
    ax.bar(range(len(subbands)), mean_power_per_subband, color='steelblue', alpha=0.7)
    ax.set_xlabel('Subband Index')
    ax.set_ylabel('Mean Power (μV²)')
    ax.set_title('Mean Power per Subband (avg. across 64 channels)')
    ax.set_xticks(range(len(subbands)))
    ax.set_xticklabels(subband_labels_short, rotation=45, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN: TEST ON SINGLE TRIAL
# ============================================================================

def main():
    print("=" * 70)
    print("STEP 9a: FILTER BANK FEATURE EXTRACTION - SINGLE TRIAL TEST")
    print("=" * 70)
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Load preprocessed data for S1 ---
    print("\n[1/5] Loading preprocessed data for S1...")
    subject_id = "S1"  # Note: S1, not S01 (no leading zero in your repo)
    data_path = INPUT_DIR / f"{subject_id}.npz"
    
    print(f"  Looking for: {data_path}")
    
    if not data_path.exists():
        print(f"✗ File not found: {data_path}")
        print(f"\n  Expected location:")
        print(f"    {INPUT_DIR}/")
        print(f"\n  Available files in that directory:")
        try:
            files = sorted(INPUT_DIR.glob("*.npz"))
            for f in files[:10]:  # Show first 10
                print(f"      - {f.name}")
            if len(files) > 10:
                print(f"      ... and {len(files) - 10} more")
        except:
            print(f"      (Directory doesn't exist or is empty)")
        return
    
    try:
        data = np.load(data_path)
        eeg_data = data['data']  # Shape: (64, 1500, 40, 6)
        print(f"✓ Loaded {subject_id}: shape {eeg_data.shape}")
    except Exception as e:
        print(f"✗ Error loading {data_path}: {e}")
        return
    
    # --- Extract one trial ---
    print("\n[2/5] Extracting one trial (target 0, block 0)...")
    target_idx = 0  # target 0 (0-indexed)
    block_idx = 0   # block 0 (0-indexed)
    trial_data = eeg_data[:, :, target_idx, block_idx]  # Shape: (64, 1500)
    print(f"✓ Trial shape: {trial_data.shape}")
    print(f"  Channels: {trial_data.shape[0]}, Time points: {trial_data.shape[1]}")
    
    # --- Design filter bank ---
    print("\n[3/5] Designing filter bank...")
    filterbank = create_filterbank(SUBBANDS, fs=SAMPLING_RATE, order=FILTER_ORDER)
    print(f"✓ Filter bank created: {len(filterbank)} subbands")
    for i, (f_low, f_high) in enumerate(SUBBANDS):
        print(f"  Subband {i}: {f_low}–{f_high} Hz")
    
    # --- Extract features ---
    print("\n[4/5] Extracting filter bank features...")
    features = extract_filterbank_features(trial_data, filterbank, SUBBANDS, fs=SAMPLING_RATE)
    print(f"✓ Features extracted: shape {features.shape}")
    
    # --- Validate ---
    print("\n[5/5] Validating features...")
    validate_features(features, name="S01_T1_B1")
    
    # --- Visualize ---
    print("\nGenerating visualization...")
    fig = visualize_filterbank_features(features, SUBBANDS, subject_id="S1", target_id=1, block_id=1)
    
    # --- Save results ---
    print("\nSaving results...")
    
    # Save features to feature extraction directory
    features_path = OUTPUT_DIR / 'step9a_single_trial_features.npz'
    np.savez_compressed(features_path, 
                        features=features,
                        subject='S1',
                        target=1,
                        block=1)
    print(f"✓ Saved: {features_path}")
    
    # Save visualization to results directory
    viz_path = RESULTS_DIR / 'step9a_visualization.png'
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_path}")
    plt.close()
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("STEP 9a: SUCCESS")
    print("=" * 70)
    print(f"Features shape: {features.shape}")
    print(f"Expected: (64, 9)")
    print(f"No NaN values: {np.isnan(features).sum() == 0}")
    print(f"All power > 0: {(features > 0).all()}")
    print("\nNext: Inspect visualization, then run Step 9b (batch extraction)")
    print("=" * 70)


if __name__ == '__main__':
    main()