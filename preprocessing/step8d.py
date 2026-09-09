import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = "./SSVEP-TRIAL-REPO/data/processed"
RESULTS_DIR = "./SSVEP-TRIAL-REPO/results/step8d"

FS = 250.0

EXPECTED_SHAPE = (64, 1500, 40, 6)

N_SUBJECTS = 35

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("STEP 8D: PREPROCESSED DATA VALIDATION")
print("=" * 70)

print()
print("Expected shape:", EXPECTED_SHAPE)
print("Expected subjects:", N_SUBJECTS)
print()


# ============================================================
# GLOBAL STATISTICS
# ============================================================

global_min = np.inf
global_max = -np.inf

global_sum = 0.0
global_sum_sq = 0.0
global_count = 0

max_abs_value = 0.0

nan_found = False
inf_found = False

shape_errors = []
missing_files = []


# ============================================================
# CHECK 1 + 2 + 3 + 4 + 5
# ============================================================

print("=" * 70)
print("CHECK 1-5: FILE / SHAPE / NaN / Inf / STATISTICS")
print("=" * 70)

for subject in range(1, N_SUBJECTS + 1):

    filename = f"S{subject}.npz"
    filepath = os.path.join(PROCESSED_DIR, filename)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.exists(filepath):
        print(f"ERROR: Missing {filename}")
        missing_files.append(filename)
        continue

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:
        data = np.load(filepath)["data"]
    except Exception as e:
        print(f"ERROR loading {filename}: {e}")
        shape_errors.append(filename)
        continue

    # --------------------------------------------------------
    # CHECK SHAPE
    # --------------------------------------------------------

    if data.shape != EXPECTED_SHAPE:

        print(
            f"ERROR: {filename} has shape "
            f"{data.shape}"
        )

        shape_errors.append(filename)

    # --------------------------------------------------------
    # CHECK NaN
    # --------------------------------------------------------

    if np.isnan(data).any():

        print(f"ERROR: NaN found in {filename}")

        nan_found = True

    # --------------------------------------------------------
    # CHECK INF
    # --------------------------------------------------------

    if np.isinf(data).any():

        print(f"ERROR: Inf found in {filename}")

        inf_found = True

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    data_min = float(np.min(data))
    data_max = float(np.max(data))

    global_min = min(global_min, data_min)
    global_max = max(global_max, data_max)

    abs_max = float(np.max(np.abs(data)))

    max_abs_value = max(
        max_abs_value,
        abs_max
    )

    # Use float64 for global accumulation
    flat = data.astype(np.float64).ravel()

    global_sum += np.sum(flat)
    global_sum_sq += np.sum(flat ** 2)
    global_count += flat.size

    print(
        f"S{subject:02d}: "
        f"shape={data.shape}, "
        f"min={data_min:.3f}, "
        f"max={data_max:.3f}"
    )


# ============================================================
# GLOBAL STATISTICS
# ============================================================

global_mean = global_sum / global_count

global_variance = (
    global_sum_sq / global_count
    - global_mean ** 2
)

global_std = np.sqrt(
    max(global_variance, 0)
)


print()
print("=" * 70)
print("GLOBAL STATISTICS")
print("=" * 70)

print(f"Global minimum : {global_min:.6f}")
print(f"Global maximum : {global_max:.6f}")
print(f"Global mean    : {global_mean:.6f}")
print(f"Global STD     : {global_std:.6f}")
print(f"Maximum |value|: {max_abs_value:.6f}")


# ============================================================
# CHECK NORMALIZATION ON REPRESENTATIVE TRIALS
# ============================================================

print()
print("=" * 70)
print("CHECK 4: CHANNEL-WISE NORMALIZATION")
print("=" * 70)

representative_subjects = [1, 15, 30]

for subject in representative_subjects:

    filepath = os.path.join(
        PROCESSED_DIR,
        f"S{subject}.npz"
    )

    data = np.load(filepath)["data"]

    # Target 1, Block 1
    trial = data[:, :, 0, 0]

    channel_means = np.mean(
        trial,
        axis=1
    )

    channel_stds = np.std(
        trial,
        axis=1
    )

    print()
    print(f"Subject {subject:02d}")
    print(
        "Mean range:",
        f"{channel_means.min():.6f}",
        "to",
        f"{channel_means.max():.6f}"
    )

    print(
        "STD range :",
        f"{channel_stds.min():.6f}",
        "to",
        f"{channel_stds.max():.6f}"
    )


# ============================================================
# CHECK 6: TIME-DOMAIN PLOTS
# ============================================================

print()
print("=" * 70)
print("CHECK 6: TIME-DOMAIN VISUALIZATION")
print("=" * 70)


examples = [
    (1, 1, 1),
    (15, 20, 1),
    (30, 1, 1),
    (30, 40, 1),
]


for subject, target, block in examples:

    filepath = os.path.join(
        PROCESSED_DIR,
        f"S{subject}.npz"
    )

    data = np.load(filepath)["data"]

    # MATLAB-style target/block numbering
    target_idx = target - 1
    block_idx = block - 1

    trial = data[
        :,
        :,
        target_idx,
        block_idx
    ]

    # Plot first 4 channels
    plt.figure(figsize=(12, 6))

    time = np.arange(1500) / FS

    for ch in range(4):

        plt.plot(
            time,
            trial[ch],
            label=f"Channel {ch + 1}"
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized amplitude")
    plt.title(
        f"S{subject} Target {target} "
        f"Block {block}"
    )

    plt.legend()
    plt.grid(True)

    output = os.path.join(
        RESULTS_DIR,
        f"time_S{subject}_T{target}_B{block}.png"
    )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print("Saved:", output)


# ============================================================
# CHECK 7: FREQUENCY-DOMAIN VALIDATION
# ============================================================

print()
print("=" * 70)
print("CHECK 7: FREQUENCY-DOMAIN VALIDATION")
print("=" * 70)


# Target frequencies
freqs_target = np.array([
    8.0, 9.0, 10.0, 11.0,
    12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2,
    12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4,
    12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6,
    12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8,
    12.8, 13.8, 14.8, 15.8
])


frequency_examples = [
    (1, 1, 1),
    (15, 20, 1),
    (30, 1, 1),
    (30, 40, 1),
]


for subject, target, block in frequency_examples:

    filepath = os.path.join(
        PROCESSED_DIR,
        f"S{subject}.npz"
    )

    data = np.load(filepath)["data"]

    target_idx = target - 1
    block_idx = block - 1

    trial = data[
        :,
        :,
        target_idx,
        block_idx
    ]

    # Average across channels
    signal = np.mean(
        trial,
        axis=0
    )

    # FFT
    fft = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / FS
    )

    magnitude = np.abs(fft)

    # Limit plot to 0-70 Hz
    mask = frequencies <= 70

    plt.figure(figsize=(12, 6))

    plt.plot(
        frequencies[mask],
        magnitude[mask]
    )

    # Expected target frequency
    f0 = freqs_target[target_idx]

    for harmonic in range(1, 5):

        harmonic_freq = harmonic * f0

        if harmonic_freq <= 70:

            plt.axvline(
                harmonic_freq,
                linestyle="--",
                alpha=0.7
            )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(
        f"S{subject} Target {target} "
        f"Frequency Spectrum"
    )

    plt.xlim(0, 70)
    plt.grid(True)

    output = os.path.join(
        RESULTS_DIR,
        f"fft_S{subject}_T{target}_B{block}.png"
    )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print("Saved:", output)


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 70)
print("STEP 8D FINAL VERDICT")
print("=" * 70)

passed = True


if missing_files:
    print("FAIL: Missing files:", missing_files)
    passed = False
else:
    print("PASS: All 35 subject files found.")


if shape_errors:
    print("FAIL: Shape errors:", shape_errors)
    passed = False
else:
    print("PASS: All files have expected shape.")


if nan_found:
    print("FAIL: NaN values found.")
    passed = False
else:
    print("PASS: No NaN values.")


if inf_found:
    print("FAIL: Infinite values found.")
    passed = False
else:
    print("PASS: No infinite values.")


print()
print(
    f"Global mean = {global_mean:.6f}"
)

print(
    f"Global STD  = {global_std:.6f}"
)

print(
    f"Maximum absolute value = "
    f"{max_abs_value:.6f}"
)


if passed:

    print()
    print("================================================")
    print("STEP 8D PASSED")
    print("Preprocessed dataset is structurally valid.")
    print("================================================")

else:

    print()
    print("================================================")
    print("STEP 8D FAILED")
    print("Investigate errors before Step 9.")
    print("================================================")