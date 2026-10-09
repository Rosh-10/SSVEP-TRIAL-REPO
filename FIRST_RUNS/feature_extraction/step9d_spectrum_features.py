"""
STEP 9d: FINE-RESOLUTION FFT SPECTRUM FEATURES FOR ALL 35 SUBJECTS (FIXED)

Objective:
    Extract high-resolution frequency spectrum (0.2 Hz bins) from preprocessed EEG.
    This captures discriminative frequency information at the same resolution as target spacing.

Input:
    Preprocessed data: data/processed/S1.npz, ..., S35.npz
    Each file: shape (64, 1500, 40, 6)

Output:
    Spectrum files: feature_extraction/S1_spectrum.npz, ..., S35_spectrum.npz
    Each file: shape (40, 6, 64, 280) — (targets, blocks, channels, freq_bins)
    Freq range: 8-64 Hz at 0.2 Hz resolution
    Log file: results/step9d_spectrum_log.txt

Time estimate: ~10-15 minutes for all 35 subjects

Why Step 9d?
    9-band power features achieved ~5% accuracy (chance for 40 targets).
    Targets spaced 0.2 Hz apart (8.0, 8.2, 8.4, ..., 15.8 Hz).
    4 Hz wide bands cannot distinguish targets only 0.2 Hz apart.
    
    FFT spectrum at 0.2 Hz resolution provides 280 frequency bins per channel.
    280 bins × 64 channels = 17,920 features/trial (30x more information).
    Expected accuracy: 80%+ for single-subject leave-one-block-out.
"""

import numpy as np
from pathlib import Path
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

SAMPLING_RATE = 250
FREQ_RESOLUTION = 0.2  # Hz (matches target spacing: 8.0, 8.2, 8.4, ...)
FREQ_LOW = 8.0
FREQ_HIGH = 64.0
NUM_SUBJECTS = 35
NUM_TARGETS = 40
NUM_BLOCKS = 6
NUM_CHANNELS = 64

# Calculate number of frequency bins
NUM_FREQ_BINS = int((FREQ_HIGH - FREQ_LOW) / FREQ_RESOLUTION)
print(f"[INFO] Configuration: {FREQ_LOW}-{FREQ_HIGH} Hz at {FREQ_RESOLUTION} Hz resolution = {NUM_FREQ_BINS} bins")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_spectrum_features(trial_data, sampling_rate=250, freq_low=8, freq_high=64, freq_res=0.2):
    """
    Extract FFT spectrum features from a single trial.
    
    Args:
        trial_data: Shape (64, 1500) — all channels, all time samples
        sampling_rate: 250 Hz
        freq_low, freq_high: Frequency range to extract
        freq_res: Desired frequency resolution (Hz)
    
    Returns:
        spectrum: Shape (64, num_freq_bins) — magnitude spectrum per channel
    """
    n_channels, n_samples = trial_data.shape
    
    # Desired output frequency bins
    target_freqs = np.arange(freq_low, freq_high, freq_res)
    n_freq_bins = len(target_freqs)
    
    # Compute FFT with zero-padding to next power of 2 for efficiency
    fft_len = 2 ** int(np.ceil(np.log2(n_samples)))  # Next power of 2 after 1500 = 2048
    
    spectrum = np.zeros((n_channels, n_freq_bins), dtype=np.float32)
    
    for ch_idx in range(n_channels):
        channel_signal = trial_data[ch_idx, :]
        
        # Compute FFT (automatically zero-pads to fft_len)
        fft_output = np.fft.rfft(channel_signal, n=fft_len)
        
        # Frequency array for FFT output
        fft_freqs = np.fft.rfftfreq(fft_len, 1.0 / sampling_rate)
        
        # Extract magnitude at desired frequencies (interpolate if needed)
        magnitude = np.abs(fft_output)
        spectrum[ch_idx, :] = np.interp(target_freqs, fft_freqs, magnitude)
    
    return spectrum

# ============================================================================
# MAIN: BATCH EXTRACT
# ============================================================================

def batch_extract_spectrum():
    """Load all 35 subjects, extract spectrum features, save results."""
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("STEP 9d: FINE-RESOLUTION FFT SPECTRUM FEATURES FOR ALL 35 SUBJECTS")
    log_lines.append("=" * 80)
    log_lines.append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Output directory: {OUTPUT_DIR}")
    log_lines.append(f"Frequency range: {FREQ_LOW}-{FREQ_HIGH} Hz at {FREQ_RESOLUTION} Hz resolution")
    log_lines.append(f"Frequency bins: {NUM_FREQ_BINS}")
    log_lines.append(f"Output shape per subject: ({NUM_TARGETS}, {NUM_BLOCKS}, {NUM_CHANNELS}, {NUM_FREQ_BINS})")
    log_lines.append("")
    
    # Check existing files
    existing = []
    missing = []
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"
        output_file = OUTPUT_DIR / f"{subject_id}_spectrum.npz"
        if output_file.exists():
            existing.append(subject_id)
        else:
            missing.append(subject_id)
    
    log_lines.append(f"Checking existing files: {len(existing)}/35 found")
    
    # If all exist, verify and report
    if len(missing) == 0:
        log_lines.append("[INFO] All 35 spectrum files already exist. Verifying...")
        log_lines.append("")
        
        all_valid = True
        for subject_id in existing:
            try:
                output_file = OUTPUT_DIR / f"{subject_id}_spectrum.npz"
                data = np.load(output_file)['spectrum']
                if data.shape == (NUM_TARGETS, NUM_BLOCKS, NUM_CHANNELS, NUM_FREQ_BINS):
                    log_lines.append(f"{subject_id}: [OK] shape={data.shape}")
                else:
                    log_lines.append(f"{subject_id}: [ERROR] Wrong shape: {data.shape}")
                    all_valid = False
            except Exception as e:
                log_lines.append(f"{subject_id}: [ERROR] {str(e)}")
                all_valid = False
        
        log_lines.append("")
        if all_valid:
            log_lines.append("=" * 80)
            log_lines.append("STEP 9d: ALL FILES VERIFIED")
            log_lines.append(f"Success: 35/35")
            log_lines.append(f"Errors: 0/35")
            log_lines.append(f"Status: Ready to retest accuracy and proceed to Step 10")
            log_lines.append("=" * 80)
            log_lines.append("")
            log_lines.append("Next: Run test_accuracy_spectrum.py to verify improved classification accuracy")
    
    # If any missing, extract them
    if len(missing) > 0:
        log_lines.append(f"[INFO] Extracting {len(missing)} missing files...")
        log_lines.append("")
        
        start_time = time.time()
        success_count = len(existing)  # Already have these
        error_count = 0
        
        for subject_id in missing:
            try:
                subject_idx = int(subject_id[1:])
                
                # Load preprocessed data
                input_file = PROCESSED_DIR / f"{subject_id}.npz"
                if not input_file.exists():
                    log_lines.append(f"{subject_id}: [ERROR] File not found at {input_file}")
                    error_count += 1
                    continue
                
                data = np.load(input_file)['data']  # Shape: (64, 1500, 40, 6)
                
                if data.shape != (NUM_CHANNELS, 1500, NUM_TARGETS, NUM_BLOCKS):
                    log_lines.append(f"{subject_id}: [ERROR] Unexpected shape: {data.shape}")
                    error_count += 1
                    continue
                
                # Extract spectrum for all trials
                spectrum = np.zeros(
                    (NUM_TARGETS, NUM_BLOCKS, NUM_CHANNELS, NUM_FREQ_BINS),
                    dtype=np.float32
                )
                
                for target_idx in range(NUM_TARGETS):
                    for block_idx in range(NUM_BLOCKS):
                        trial = data[:, :, target_idx, block_idx]  # (64, 1500)
                        spectrum[target_idx, block_idx, :, :] = extract_spectrum_features(
                            trial,
                            sampling_rate=SAMPLING_RATE,
                            freq_low=FREQ_LOW,
                            freq_high=FREQ_HIGH,
                            freq_res=FREQ_RESOLUTION
                        )
                
                # Save spectrum
                output_file = OUTPUT_DIR / f"{subject_id}_spectrum.npz"
                np.savez_compressed(output_file, spectrum=spectrum)
                
                elapsed = time.time() - start_time
                log_lines.append(f"{subject_id}: [OK] shape={spectrum.shape}, {elapsed:.1f}s elapsed")
                success_count += 1
                
            except Exception as e:
                log_lines.append(f"{subject_id}: [ERROR] {str(e)}")
                error_count += 1
        
        # Summary
        total_time = time.time() - start_time
        log_lines.append("")
        log_lines.append("=" * 80)
        log_lines.append("STEP 9d: BATCH EXTRACTION COMPLETE")
        log_lines.append(f"Success: {success_count}/{NUM_SUBJECTS}")
        log_lines.append(f"Errors: {error_count}/{NUM_SUBJECTS}")
        log_lines.append(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        log_lines.append(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append("=" * 80)
        log_lines.append("")
        log_lines.append("Next: Run test_accuracy_spectrum.py to verify improved classification accuracy")
    
    # Write log
    log_file = RESULTS_DIR / "step9d_spectrum_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    # Print to console
    print('\n'.join(log_lines))


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    batch_extract_spectrum()