import os
import numpy as np
import scipy.io as sio

from scipy.signal import butter, sosfiltfilt


# ============================================================
# Step 7C: Validate 6-65 Hz Filter
#
# Test subjects:
#   S1, S15, S30
#
# Test targets:
#   1, 10, 20, 30, 40
#
# Test block:
#   Block 1
#
# Channels:
#   All 64
#
# Purpose:
#   Check whether the filter consistently preserves
#   SSVEP harmonics across subjects and targets.
# ============================================================


FS = 250.0

LOW_CUTOFF = 6.0
HIGH_CUTOFF = 65.0

FILTER_ORDER = 5

SUBJECTS = [1, 15, 30]

TARGETS = [1, 10, 20, 30, 40]

BLOCK = 1


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
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "benchmark"
)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# ------------------------------------------------------------
# FFT magnitude
# ------------------------------------------------------------

def get_fft(signal):

    signal = signal - np.mean(signal)

    n = len(signal)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / FS
    )

    magnitude = np.abs(fft_values)

    return frequencies, magnitude


def magnitude_at(
    frequencies,
    magnitude,
    target_frequency
):

    index = np.argmin(
        np.abs(
            frequencies - target_frequency
        )
    )

    return magnitude[index]


# ------------------------------------------------------------
# Analyze one trial
# ------------------------------------------------------------

def analyze_trial(trial, f0):

    filtered = sosfiltfilt(
        sos,
        trial,
        axis=1
    )

    raw_frequencies = []
    raw_magnitudes = []

    filtered_frequencies = []
    filtered_magnitudes = []

    # Analyze all 64 channels
    for channel in range(64):

        raw_signal = trial[channel]

        filtered_signal = filtered[channel]

        rf, rm = get_fft(
            raw_signal
        )

        ff, fm = get_fft(
            filtered_signal
        )

        raw_frequencies.append(rf)
        raw_magnitudes.append(rm)

        filtered_frequencies.append(ff)
        filtered_magnitudes.append(fm)

    # Average across channels
    important = [
        f0,
        2 * f0,
        3 * f0,
        4 * f0
    ]

    raw_values = []
    filtered_values = []

    for frequency in important:

        channel_raw = []

        channel_filtered = []

        for channel in range(64):

            raw_value = magnitude_at(
                raw_frequencies[channel],
                raw_magnitudes[channel],
                frequency
            )

            filtered_value = magnitude_at(
                filtered_frequencies[channel],
                filtered_magnitudes[channel],
                frequency
            )

            channel_raw.append(
                raw_value
            )

            channel_filtered.append(
                filtered_value
            )

        raw_values.append(
            np.mean(channel_raw)
        )

        filtered_values.append(
            np.mean(channel_filtered)
        )

    return filtered, raw_values, filtered_values


# ============================================================
# Main
# ============================================================

print()
print("=" * 75)
print("Step 7C: Multi-Trial Filter Validation")
print("=" * 75)
print()

print(
    f"Filter: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz"
)

print(
    f"Subjects: {SUBJECTS}"
)

print(
    f"Targets: {TARGETS}"
)

print(
    f"Block: {BLOCK}"
)

print()


all_changes = []


for subject in SUBJECTS:

    file_path = os.path.join(
        DATA_DIR,
        f"S{subject}.mat"
    )

    print("=" * 75)
    print(f"Subject {subject:02d}")
    print("=" * 75)

    mat = sio.loadmat(
        file_path
    )

    data = mat["data"]

    for target in TARGETS:

        f0 = FREQS[target - 1]

        trial = data[
            :,
            :,
            target - 1,
            BLOCK - 1
        ]

        filtered, raw_values, filtered_values = (
            analyze_trial(
                trial,
                f0
            )
        )

        print()
        print(
            f"Target {target:02d} "
            f"(f0 = {f0:.1f} Hz)"
        )

        print(
            f"{'Component':<15}"
            f"{'Frequency':<12}"
            f"{'Raw':<15}"
            f"{'Filtered':<15}"
            f"{'Change %':<12}"
        )

        print("-" * 70)

        labels = [
            "Fundamental",
            "2nd harmonic",
            "3rd harmonic",
            "4th harmonic"
        ]

        for i in range(4):

            frequency = (
                (i + 1) * f0
            )

            raw_value = raw_values[i]

            filtered_value = filtered_values[i]

            if raw_value != 0:

                change = (
                    (filtered_value - raw_value)
                    / raw_value
                    * 100
                )

            else:

                change = 0

            print(
                f"{labels[i]:<15}"
                f"{frequency:<12.1f}"
                f"{raw_value:<15.2f}"
                f"{filtered_value:<15.2f}"
                f"{change:<12.2f}"
            )

            all_changes.append(
                change
            )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 75)
print("Step 7C Summary")
print("=" * 75)
print()

print(
    f"Number of test trials: "
    f"{len(SUBJECTS) * len(TARGETS)}"
)

print(
    "All 64 channels were included in each trial."
)

print()

print(
    "The purpose of this test is to verify that the "
    "6-65 Hz filter consistently preserves the SSVEP "
    "frequency components."
)

print()
print("=" * 75)
print("STEP 7C COMPLETE")
print("=" * 75)
print()