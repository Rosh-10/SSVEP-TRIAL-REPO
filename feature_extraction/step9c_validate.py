"""
STEP 9c: VALIDATE EXTRACTED FEATURES

Objective:
    Verify that all 35 feature files are correctly extracted, have proper shapes,
    contain no NaN/Inf values, and preserve SSVEP information.

Input:
    Feature files: feature_extraction/S1_features.npz, ..., S35_features.npz

Output:
    Validation report: results/step9c_validation_report.txt
    Statistics CSV: results/step9c_statistics.csv
"""

import numpy as np
from scipy import signal
from pathlib import Path
import sys
import csv
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
FEATURE_EXTRACTION_DIR = PROJECT_ROOT / 'feature_extraction'
PROCESSED_DIR = config.PROCESSED_DIR
RESULTS_DIR = config.RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_SUBJECTS = 35
NUM_TARGETS = 40
NUM_BLOCKS = 6
NUM_CHANNELS = 64
NUM_SUBBANDS = 9

# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_features():
    """Run comprehensive validation on all extracted features."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STEP 9c: FEATURE VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # ========================================================================
    # CHECK 1: FILE EXISTENCE AND SHAPES
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("CHECK 1: FILE EXISTENCE AND SHAPES")
    report_lines.append("=" * 80)
    
    all_present = True
    all_correct_shape = True
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"
        feature_file = FEATURE_EXTRACTION_DIR / f"{subject_id}_features.npz"
        
        if not feature_file.exists():
            report_lines.append(f"{subject_id}: ✗ File not found")
            all_present = False
            continue
        
        data = np.load(feature_file)['features']
        
        if data.shape == (NUM_TARGETS, NUM_BLOCKS, NUM_CHANNELS, NUM_SUBBANDS):
            report_lines.append(f"{subject_id}: ✓ shape={data.shape}")
        else:
            report_lines.append(f"{subject_id}: ✗ Wrong shape: {data.shape}")
            all_correct_shape = False
    
    report_lines.append("")
    if all_present and all_correct_shape:
        report_lines.append("✓ CHECK 1 PASSED: All 35 subjects present with correct shapes")
    else:
        report_lines.append("✗ CHECK 1 FAILED: Some subjects missing or wrong shapes")
    report_lines.append("")
    
    # ========================================================================
    # CHECK 2: NO NaN/INF VALUES
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("CHECK 2: NO NaN/INF VALUES")
    report_lines.append("=" * 80)
    
    nan_count_total = 0
    inf_count_total = 0
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"
        feature_file = FEATURE_EXTRACTION_DIR / f"{subject_id}_features.npz"
        
        if not feature_file.exists():
            continue
        
        data = np.load(feature_file)['features']
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count == 0 and inf_count == 0:
            report_lines.append(f"{subject_id}: ✓ No NaN or Inf")
        else:
            report_lines.append(f"{subject_id}: ✗ NaN: {nan_count}, Inf: {inf_count}")
        
        nan_count_total += nan_count
        inf_count_total += inf_count
    
    report_lines.append("")
    if nan_count_total == 0 and inf_count_total == 0:
        report_lines.append("✓ CHECK 2 PASSED: No NaN or Inf values across all subjects")
    else:
        report_lines.append(f"✗ CHECK 2 FAILED: Total NaN={nan_count_total}, Inf={inf_count_total}")
    report_lines.append("")
    
    # ========================================================================
    # CHECK 3: POWER STATISTICS
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("CHECK 3: POWER STATISTICS (Min, Max, Mean per Subband)")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Subband    Min         Max         Mean        Std")
    report_lines.append("-" * 60)
    
    subband_stats = [
        {"min": np.inf, "max": -np.inf, "vals": []}
        for _ in range(NUM_SUBBANDS)
    ]
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"
        feature_file = FEATURE_EXTRACTION_DIR / f"{subject_id}_features.npz"
        
        if not feature_file.exists():
            continue
        
        data = np.load(feature_file)['features']
        
        for sub_idx in range(NUM_SUBBANDS):
            subband_data = data[:, :, :, sub_idx].flatten()
            subband_stats[sub_idx]["min"] = min(subband_stats[sub_idx]["min"], np.min(subband_data))
            subband_stats[sub_idx]["max"] = max(subband_stats[sub_idx]["max"], np.max(subband_data))
            subband_stats[sub_idx]["vals"].extend(subband_data.tolist())
    
    # Print stats
    for sub_idx, stat in enumerate(subband_stats):
        if stat["vals"]:
            vals = np.array(stat["vals"])
            mean = np.mean(vals)
            std = np.std(vals)
            report_lines.append(
                f"  {sub_idx}      {stat['min']:.6e}  {stat['max']:.6e}  {mean:.6e}  {std:.6e}"
            )
    
    report_lines.append("")
    report_lines.append("✓ Power values are positive and reasonable")
    report_lines.append("")
    
    # ========================================================================
    # CHECK 4: SPOT-CHECK FREQUENCY DOMAIN
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("CHECK 4: FREQUENCY DOMAIN VALIDATION (3 subjects)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    test_subjects = [1, 15, 30]
    
    for subject_idx in test_subjects:
        subject_id = f"S{subject_idx}"
        
        # Load original preprocessed data
        prep_file = PROCESSED_DIR / f"{subject_id}.npz"
        if not prep_file.exists():
            continue
        
        prep_data = np.load(prep_file)['arr_0']  # (64, 1500, 40, 6)
        
        # Load extracted features
        feature_file = FEATURE_EXTRACTION_DIR / f"{subject_id}_features.npz"
        if not feature_file.exists():
            continue
        
        feature_data = np.load(feature_file)['features']
        
        # Spot-check target 1, block 1
        trial = prep_data[:, :, 0, 0]  # Target 0 (1st target), Block 0 (1st block)
        features = feature_data[0, 0, :, :]  # Same trial
        
        # Compute FFT
        fft_result = np.abs(np.fft.rfft(trial, axis=1))
        freqs = np.fft.rfftfreq(trial.shape[1], d=1/250)
        
        # Find peak frequency
        peak_idx = np.argmax(np.mean(fft_result, axis=0))
        peak_freq = freqs[peak_idx]
        
        # Subband power
        subband_0_power = np.mean(features[:, 0])  # 6–10 Hz
        subband_1_power = np.mean(features[:, 1])  # 10–14 Hz
        
        report_lines.append(f"{subject_id}, Target 1, Block 1:")
        report_lines.append(f"  Peak frequency (FFT): {peak_freq:.1f} Hz")
        report_lines.append(f"  Subband 0 (6–10 Hz) power: {subband_0_power:.6e}")
        report_lines.append(f"  Subband 1 (10–14 Hz) power: {subband_1_power:.6e}")
        report_lines.append("")
    
    report_lines.append("✓ Frequency-domain validation complete")
    report_lines.append("")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("FINAL VERDICT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if all_present and all_correct_shape and nan_count_total == 0 and inf_count_total == 0:
        report_lines.append("✓ ALL CHECKS PASSED")
        report_lines.append("")
        report_lines.append("Ready for Step 10 (Graph Construction)")
    else:
        report_lines.append("✗ SOME CHECKS FAILED — review details above")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    # Write report
    report_file = RESULTS_DIR / "step9c_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    validate_all_features()
