import os
import numpy as np
import scipy.io as sio


# ============================================================
# Step 6E: Third Harmonic Check
#
# Purpose:
#   Measure the strength of the 3rd harmonic (3*f0)
#   relative to the fundamental (f0).
#
# Dataset:
#   Tsinghua SSVEP Benchmark Dataset
#
# Data shape:
#   (64 channels, 1500 samples, 40 targets, 6 blocks)
#
# Sampling rate:
#   250 Hz
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "raw",
    "benchmark"
)

FS = 250

SUBJECTS = [1, 15, 30]

NUM_TARGETS = 40
NUM_BLOCKS = 6


# Target frequencies in the exact benchmark order.
# DO NOT SORT THESE.
FREQS = np.array([
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


# ------------------------------------------------------------
# FFT magnitude at a target frequency
# ------------------------------------------------------------

def fft_magnitude_at_frequency(signal, target_freq, fs):
    """
    Calculate the FFT magnitude at the frequency bin
    closest to target_freq.
    """

    n = len(signal)

    # Remove DC component
    signal = signal - np.mean(signal)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    magnitude = np.abs(fft_values)

    index = np.argmin(
        np.abs(frequencies - target_freq)
    )

    return magnitude[index]


# ------------------------------------------------------------
# Calculate harmonic ratio
# ------------------------------------------------------------

def calculate_harmonic_ratios(eeg_trial, f0):
    """
    eeg_trial shape:
        (64, 1500)

    Returns:
        R2 = 2nd harmonic / fundamental
        R3 = 3rd harmonic / fundamental

    We average the FFT magnitude across all 64 channels.
    """

    fundamental_freq = f0
    second_freq = 2 * f0
    third_freq = 3 * f0

    fundamental_values = []
    second_values = []
    third_values = []

    for channel in range(eeg_trial.shape[0]):

        signal = eeg_trial[channel, :]

        mag_f0 = fft_magnitude_at_frequency(
            signal,
            fundamental_freq,
            FS
        )

        mag_f1 = fft_magnitude_at_frequency(
            signal,
            second_freq,
            FS
        )

        mag_f2 = fft_magnitude_at_frequency(
            signal,
            third_freq,
            FS
        )

        fundamental_values.append(mag_f0)
        second_values.append(mag_f1)
        third_values.append(mag_f2)

    fundamental = np.mean(fundamental_values)
    second = np.mean(second_values)
    third = np.mean(third_values)

    # Avoid division by zero
    if fundamental == 0:
        r2 = 0
        r3 = 0
    else:
        r2 = second / fundamental
        r3 = third / fundamental

    return r2, r3


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

print()
print("=" * 70)
print("Step 6E: Third Harmonic Check")
print("=" * 70)
print()

all_results = {}


for subject in SUBJECTS:

    file_path = os.path.join(
        DATA_DIR,
        f"S{subject}.mat"
    )

    print(f"Loading Subject {subject:02d}...")

    mat = sio.loadmat(file_path)

    eeg = mat["data"]

    print(f"Data shape: {eeg.shape}")

    r2_values = []
    r3_values = []

    per_target = []

    for target_index in range(NUM_TARGETS):

        f0 = FREQS[target_index]

        target_r2 = []
        target_r3 = []

        for block in range(NUM_BLOCKS):

            # Shape:
            # (64 channels, 1500 samples)

            trial = eeg[
                :,
                :,
                target_index,
                block
            ]

            r2, r3 = calculate_harmonic_ratios(
                trial,
                f0
            )

            target_r2.append(r2)
            target_r3.append(r3)

        # Average across 6 blocks
        mean_r2 = np.mean(target_r2)
        mean_r3 = np.mean(target_r3)

        r2_values.append(mean_r2)
        r3_values.append(mean_r3)

        per_target.append(
            (
                target_index + 1,
                f0,
                mean_r2,
                mean_r3
            )
        )

    r2_values = np.array(r2_values)
    r3_values = np.array(r3_values)

    all_results[subject] = {
        "r2": r2_values,
        "r3": r3_values
    }

    print()
    print(
        f"Subject {subject:02d}: "
        f"mean R2 = {np.mean(r2_values):.3f}, "
        f"min R2 = {np.min(r2_values):.3f}, "
        f"max R2 = {np.max(r2_values):.3f}"
    )

    print(
        f"Subject {subject:02d}: "
        f"mean R3 = {np.mean(r3_values):.3f}, "
        f"min R3 = {np.min(r3_values):.3f}, "
        f"max R3 = {np.max(r3_values):.3f}"
    )


# ------------------------------------------------------------
# Detailed results for Subject 1
# ------------------------------------------------------------

print()
print("=" * 70)
print("Subject 01: Per-target harmonic details")
print("=" * 70)
print()

print(
    f"{'Target':>6} "
    f"{'f0':>7} "
    f"{'R2':>10} "
    f"{'R3':>10}"
)

print("-" * 40)

subject = 1

r2_values = all_results[subject]["r2"]
r3_values = all_results[subject]["r3"]

for target_index in range(NUM_TARGETS):

    f0 = FREQS[target_index]

    print(
        f"{target_index + 1:6d} "
        f"{f0:7.1f} "
        f"{r2_values[target_index]:10.3f} "
        f"{r3_values[target_index]:10.3f}"
    )


# ------------------------------------------------------------
# Strongest 3rd harmonic targets
# ------------------------------------------------------------

print()
print("=" * 70)
print("Strongest 3rd-harmonic responses")
print("=" * 70)

for subject in SUBJECTS:

    r3_values = all_results[subject]["r3"]

    strongest_indices = np.argsort(
        r3_values
    )[::-1][:5]

    print()
    print(f"Subject {subject:02d}:")

    for index in strongest_indices:

        f0 = FREQS[index]

        print(
            f"  Target {index + 1:2d} "
            f"({f0:4.1f} Hz): "
            f"R3 = {r3_values[index]:.3f} "
            f"(3f0 = {3*f0:.1f} Hz)"
        )


print()
print("=" * 70)
print("STEP 6E COMPLETE")
print("=" * 70)
print()