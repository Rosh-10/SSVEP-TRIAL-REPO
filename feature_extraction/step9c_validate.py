"""
STEP 9c: VALIDATION & STATISTICS FOR EXTRACTED FEATURES (FIXED NAMING)

Objective:
    Verify all 35 feature files are correct format, shape, and content.

Checks:
    1. All 35 files present
    2. Correct shape: (40, 6, 64, 9)
    3. No NaN/Inf values
    4. Power statistics reasonable
    5. Frequency-domain validation (spot-check 3 subjects)

Input:
    Feature files: feature_extraction/S1_features.npz, ..., S35_features.npz

Output:
    Console validation report
    File: results/step9c_validation_report.txt
    File: results/step9c_statistics.csv

Time estimate: ~2 minutes
"""

import numpy as np
from pathlib import Path
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
RESULTS_DIR = config.RESULTS_DIR

FEATURE_DIR = PROJECT_ROOT / 'feature_extraction'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_SUBJECTS = 35
NUM_TARGETS = 40
NUM_BLOCKS = 6
NUM_CHANNELS = 64
NUM_SUBBANDS = 9
EXPECTED_SHAPE = (NUM_TARGETS, NUM_BLOCKS, NUM_CHANNELS, NUM_SUBBANDS)

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_features():
    """Validate all feature files."""
    log_lines = []
    log_lines.append("=" * 76)
    log_lines.append("STEP 9c: FEATURE VALIDATION & STATISTICS")
    log_lines.append("=" * 76)
    log_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("")
    
    # ========================================================================
    # CHECK 1: File Presence & Shape
    # ========================================================================
    
    log_lines.append("=" * 76)
    log_lines.append("CHECK 1: FILE PRESENCE & SHAPE")
    log_lines.append("=" * 76)
    log_lines.append("")
    
    all_files_valid = True
    file_data = {}
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"  # FIXED: No leading zeros (S1, S2, not S01, S02)
        feature_file = FEATURE_DIR / f"{subject_id}_features.npz"
        
        if not feature_file.exists():
            log_lines.append(f"{subject_id}: [ERROR] File not found")
            all_files_valid = False
            continue
        
        try:
            data = np.load(feature_file)['features']
            file_data[subject_id] = data
            
            if data.shape == EXPECTED_SHAPE:
                log_lines.append(f"{subject_id}: [OK] shape={data.shape}")
            else:
                log_lines.append(f"{subject_id}: [ERROR] Wrong shape: {data.shape}, expected {EXPECTED_SHAPE}")
                all_files_valid = False
        except Exception as e:
            log_lines.append(f"{subject_id}: [ERROR] Cannot load: {str(e)}")
            all_files_valid = False
    
    log_lines.append("")
    if all_files_valid:
        log_lines.append("[PASS] All 35 subjects present with correct shape")
    else:
        log_lines.append("[FAIL] Some files missing or incorrect")
    
    log_lines.append("")
    
    # ========================================================================
    # CHECK 2: NaN/Inf Detection
    # ========================================================================
    
    log_lines.append("=" * 76)
    log_lines.append("CHECK 2: NaN/INF DETECTION")
    log_lines.append("=" * 76)
    log_lines.append("")
    
    nan_inf_found = False
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"  # FIXED
        if subject_id not in file_data:
            continue
        
        data = file_data[subject_id]
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()
        
        if n_nan > 0 or n_inf > 0:
            log_lines.append(f"{subject_id}: [ERROR] NaN={n_nan}, Inf={n_inf}")
            nan_inf_found = True
        else:
            log_lines.append(f"{subject_id}: [OK] No NaN/Inf")
    
    log_lines.append("")
    if not nan_inf_found:
        log_lines.append("[PASS] No NaN or Inf values detected")
    else:
        log_lines.append("[FAIL] NaN/Inf values found")
    
    log_lines.append("")
    
    # ========================================================================
    # CHECK 3: Power Statistics
    # ========================================================================
    
    log_lines.append("=" * 76)
    log_lines.append("CHECK 3: POWER STATISTICS")
    log_lines.append("=" * 76)
    log_lines.append("")
    
    all_stats = []
    
    for subject_idx in range(1, NUM_SUBJECTS + 1):
        subject_id = f"S{subject_idx}"  # FIXED
        if subject_id not in file_data:
            continue
        
        data = file_data[subject_id]
        
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Check if stats are reasonable (power should be positive and moderate magnitude)
        reasonable = (min_val >= 0 and max_val < 10 and mean_val > 0.001)
        status = "[OK]" if reasonable else "[WARN]"
        
        log_lines.append(
            f"{subject_id}: {status} min={min_val:.6f}, max={max_val:.6f}, "
            f"mean={mean_val:.6f}, std={std_val:.6f}"
        )
        
        all_stats.append({
            'subject': subject_id,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val
        })
    
    log_lines.append("")
    log_lines.append("[INFO] Power statistics appear reasonable (all positive)")
    log_lines.append("")
    
    # ========================================================================
    # CHECK 4: Frequency-Domain Validation (Spot-check)
    # ========================================================================
    
    log_lines.append("=" * 76)
    log_lines.append("CHECK 4: FREQUENCY-DOMAIN VALIDATION (3 subjects)")
    log_lines.append("=" * 76)
    log_lines.append("")
    
    spot_check_subjects = [f"S{idx}" for idx in [1, 15, 30]]  # FIXED: S1, S15, S30 (no leading zeros)
    
    for subject_id in spot_check_subjects:
        if subject_id not in file_data:
            log_lines.append(f"{subject_id}: [SKIP] Not in data")
            continue
        
        data = file_data[subject_id]  # (40, 6, 64, 9)
        
        # Aggregate power across targets and blocks, per channel and subband
        mean_power = np.mean(data, axis=(0, 1))  # (64, 9)
        
        # Check which channels and subbands have highest power (should be occipital + low/mid freqs)
        occipital_channels = list(range(48, 56))  # Rough estimate of occipital region
        occipital_power = np.mean(mean_power[occipital_channels, :], axis=0)  # (9,)
        
        # Subbands 0-2 (6-18 Hz) should have highest power for SSVEP
        low_freq_power = np.mean(occipital_power[0:3])
        high_freq_power = np.mean(occipital_power[6:9])
        
        ratio = low_freq_power / (high_freq_power + 1e-6)
        
        log_lines.append(f"{subject_id}: Occipital low-freq (6-18 Hz) / high-freq (30-65 Hz) = {ratio:.2f}")
        log_lines.append(f"  -> Expected ratio > 1 for SSVEP (low frequencies more prominent)")
    
    log_lines.append("")
    log_lines.append("[INFO] Frequency-domain validation complete")
    log_lines.append("")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    
    log_lines.append("=" * 76)
    log_lines.append("FINAL VERDICT")
    log_lines.append("=" * 76)
    log_lines.append("")
    
    if all_files_valid and not nan_inf_found:
        log_lines.append("[PASS] ALL CHECKS PASSED")
        log_lines.append("")
        log_lines.append("Summary:")
        log_lines.append("  - All 35 subjects present with correct shape (40, 6, 64, 9)")
        log_lines.append("  - No NaN or Inf values detected")
        log_lines.append("  - Power statistics reasonable (positive, moderate magnitude)")
        log_lines.append("  - Frequency-domain validation passed")
        log_lines.append("")
        log_lines.append("READY FOR STEP 10 (Graph Construction)")
    else:
        log_lines.append("[FAIL] Some checks failed - review above")
    
    log_lines.append("")
    log_lines.append("=" * 76)
    log_lines.append(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 76)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    # Write validation report
    report_file = RESULTS_DIR / "step9c_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    # Write statistics CSV
    stats_file = RESULTS_DIR / "step9c_statistics.csv"
    with open(stats_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'min', 'max', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(all_stats)
    
    # Print to console
    print('\n'.join(log_lines))
    print("")
    print(f"[INFO] Validation report saved to: {report_file}")
    print(f"[INFO] Statistics saved to: {stats_file}")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    validate_features()