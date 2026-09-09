import os
import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./SSVEP-TRIAL-REPO/data/raw/benchmark"
OUTPUT_DIR = "./SSVEP-TRIAL-REPO/data/processed"

FS = 250.0

LOW_CUTOFF = 6.0
HIGH_CUTOFF = 65.0

FILTER_ORDER = 5

N_SUBJECTS = 35


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DESIGN BUTTERWORTH FILTER
# ============================================================

sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# ============================================================
# PREPROCESS ONE TRIAL
# ============================================================

def preprocess_trial(trial):
    """
    Input:
        trial shape = (64, 1500)

    Output:
        processed shape = (64, 1500)
    """

    trial = trial.astype(np.float32)

    # --------------------------------------------------------
    # 1. Remove channel-wise DC offset
    # --------------------------------------------------------

    trial = trial - np.mean(trial, axis=1, keepdims=True)

    # --------------------------------------------------------
    # 2. Bandpass filtering
    # --------------------------------------------------------

    filtered = sosfiltfilt(
        sos,
        trial,
        axis=1
    )

    # --------------------------------------------------------
    # 3. Channel-wise normalization
    #
    # Normalize each EEG channel within this trial.
    # --------------------------------------------------------

    mean = np.mean(filtered, axis=1, keepdims=True)
    std = np.std(filtered, axis=1, keepdims=True)

    std = np.maximum(std, 1e-6)

    normalized = (filtered - mean) / std

    return normalized.astype(np.float32)


# ============================================================
# PROCESS ALL SUBJECTS
# ============================================================

for subject in range(1, N_SUBJECTS + 1):

    filename = f"S{subject}.mat"
    filepath = os.path.join(DATA_DIR, filename)

    print("=" * 70)
    print(f"Processing Subject {subject:02d}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load MATLAB file
    # --------------------------------------------------------

    mat = scipy.io.loadmat(filepath)

    data = mat["data"]

    print("Original shape:", data.shape)

    # Expected:
    # 64 channels
    # 1500 samples
    # 40 targets
    # 6 blocks

    assert data.shape == (64, 1500, 40, 6), \
        f"Unexpected shape: {data.shape}"

    # --------------------------------------------------------
    # Allocate output array
    # --------------------------------------------------------

    processed = np.zeros_like(
        data,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Process every target and block
    # --------------------------------------------------------

    for target in range(40):

        for block in range(6):

            trial = data[:, :, target, block]

            processed[:, :, target, block] = preprocess_trial(
                trial
            )

    # --------------------------------------------------------
    # Save subject
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"S{subject}.npz"
    )

    np.savez_compressed(
        output_file,
        data=processed
    )

    print("Processed shape:", processed.shape)
    print("Saved:", output_file)

print()
print("=" * 70)
print("STEP 8 COMPLETE")
print("=" * 70)