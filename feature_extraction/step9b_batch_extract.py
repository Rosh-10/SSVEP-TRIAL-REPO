"""
STEP 9b: BATCH FEATURE EXTRACTION FOR ALL 35 SUBJECTS

Objective:
    Load all 35 preprocessed subjects and extract filter bank features
    for all 40 targets × 6 blocks per subject.

Input:
    Preprocessed data: data/processed/S1.npz, ..., S35.npz
    Each file: shape (64, 1500, 40, 6) — preprocessed EEG

Output:
    Feature files: feature_extraction/S1_features.npz, ..., S35_features.npz
    Each file: shape (40, 6, 64, 9) — (targets, blocks, channels, subbands)
    Log file: results/step9b_batch_log.txt

Time estimate: ~5-10 minutes for all 35 subjects
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
import sys
import time
from datetime import datetime
import importlib.util

# ============================================================================
# LOAD CONFIG
# ============================================================================

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.py'
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.py not found at {CONFIG_PATH}")

spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

PROJECT_ROOT = config.PROJECT_ROOT
PROCESSED_DIR = config.PROCESSED_DIR
RESULTS_DIR = config.RESULTS_DIR

OUTPUT_DIR = PROJECT_ROOT / 'feature_extraction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBBANDS = [
    (6, 10), (10, 14), (14, 18), (18, 22), (22, 26),
    (26, 30), (30, 34), (34, 42), (42, 65),
]
N_SUBBANDS = len(SUBBANDS)
SAMPLING_RATE = 250
FILTER_ORDER = 5
NUM_SUBJECTS = 35
NUM_TARGETS = 40
NUM_BLOCKS = 6
NUM_CHANNELS = 64

# ============================================================================
# FILTER DESIGN
# ============================================================================

def design_bandpass_filter(f_low, f_high, fs=250, order=5):
    """Design bandpass Butterworth filter (SOS format)."""
    nyquist = fs / 2
    low_norm = max(0.001, min(0.999, f_low / nyquist))
    high_norm = max(0.001, min(0.999, f_high / nyquist))
    sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
    return sos


def create_filterbank(subbands, fs=250, order=5):
    """Create filter bank: list of SOS matrices."""
    return [design_bandpass_filter(f_low, f_high, fs, order) for f_low, f_high in subbands]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_filterbank_features(trial_data, filterbank, fs=250):
    """
    Extract power features from a single trial.
    
    Args:
        trial_data: Shape (64, 1500)
        filterbank: List of SOS matrices
        fs: Sampling rate
    
    Returns:
        features: Shape (64, 9) — power per channel per subband
    """
    n_channels, n_samples = trial_data.shape
    n_subbands = len(filterbank)
    features = np.zeros((n_channels, n_subbands), dtype=np.float32)
    
    for ch_idx in range(n_channels):
        channel_signal = trial_data[ch_idx, :]
        for sb_idx, sos in enumerate(filterbank):
            filtered = sosfiltfilt(sos, channel_signal)
            power = np.mean(filtered ** 2)
            features[ch_idx, sb_idx] = power
    
    return features

# ============================================================================
# MAIN: BATCH EXTRACT
# ============================================================================

def batch_extract_features():
    """Load all 35 subjects, extract features, save results."""
    log_lines = []
    log_lines.append("=" * 76)
    log_lines.append("STEP 9b: BATCH FEATURE EXTRACTION FOR ALL 35 SUBJECTS")
    log_lines.append("=" * 76)
    log_lines.append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Output directory: {OUTPUT_DIR}")
    log_lines.append("")
    
    start_time = time.time()
    filterbank = create_filterbank(SUBBANDS, fs=SAMPLING_RATE, order=FILTER_ORDER)
    
    success_count = 0
    error_count = 0
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"
        
        try:
            # Load preprocessed data
            input_file = PROCESSED_DIR / f"{subject_id}.npz"
            if not input_file.exists():
                log_lines.append(f"{subject_id}: ✗ File not found")
                error_count += 1
                continue
            
            data = np.load(input_file)['arr_0']  # Shape: (64, 1500, 40, 6)
            
            if data.shape != (NUM_CHANNELS, 1500, NUM_TARGETS, NUM_BLOCKS):
                log_lines.append(f"{subject_id}: ✗ Unexpected shape: {data.shape}")
                error_count += 1
                continue
            
            # Extract features for all trials
            features = np.zeros(
                (NUM_TARGETS, NUM_BLOCKS, NUM_CHANNELS, N_SUBBANDS),
                dtype=np.float32
            )
            
            for target_idx in range(NUM_TARGETS):
                for block_idx in range(NUM_BLOCKS):
                    trial = data[:, :, target_idx, block_idx]  # (64, 1500)
                    features[target_idx, block_idx, :, :] = extract_filterbank_features(
                        trial, filterbank, fs=SAMPLING_RATE
                    )
            
            # Save features
            output_file = OUTPUT_DIR / f"{subject_id}_features.npz"
            np.savez_compressed(output_file, features=features)
            
            elapsed = time.time() - start_time
            log_lines.append(f"{subject_id}: ✓ shape={features.shape}, {elapsed:.1f}s elapsed")
            success_count += 1
            
        except Exception as e:
            log_lines.append(f"{subject_id}: ✗ Error: {str(e)}")
            error_count += 1
    
    # Summary
    total_time = time.time() - start_time
    log_lines.append("")
    log_lines.append("=" * 76)
    log_lines.append("STEP 9b: BATCH EXTRACTION COMPLETE")
    log_lines.append(f"Success: {success_count}/{NUM_SUBJECTS}")
    log_lines.append(f"Errors: {error_count}/{NUM_SUBJECTS}")
    log_lines.append(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    log_lines.append(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 76)
    
    # Write log
    log_file = RESULTS_DIR / "step9b_batch_log.txt"
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))
    
    # Print to console
    print('\n'.join(log_lines))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    batch_extract_features()