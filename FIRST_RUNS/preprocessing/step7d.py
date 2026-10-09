import os
import numpy as np
import scipy.io as sio

from scipy.signal import butter, sosfiltfilt


# ============================================================
# Step 7D
# Investigate S30 Target 1
#
# Purpose:
# Determine why the 8 Hz component changed substantially
# after filtering.
# ============================================================


FS = 250.0

LOW_CUTOFF = 6.0
HIGH_CUTOFF = 65.0

FILTER_ORDER = 5

SUBJECT = 30
TARGET = 1
BLOCK = 1

F0 = 8.0


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

DATA_FILE = os.path.join(
    DATA_DIR,
    f"S{SUBJECT}.mat"
)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print()
print("=" * 75)
print("Step 7D: Investigate S30 Target 1")
print("=" * 75)
print()

print(f"Loading: {DATA_FILE}")

mat = sio.loadmat(DATA_FILE)

data = mat["data"]

print(f"Dataset shape: {data.shape}")


# ------------------------------------------------------------
# Extract trial
# ------------------------------------------------------------

trial = data[
    :,
    :,
    TARGET - 1,
    BLOCK - 1
]

print(f"Trial shape: {trial.shape}")


# ------------------------------------------------------------
# Filter
# ------------------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(
    sos,
    trial,
    axis=1
)


# ------------------------------------------------------------
# FFT helper
# ------------------------------------------------------------

def fft_magnitude(signal):

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
# Frequencies
# ------------------------------------------------------------

frequencies_to_check = [
    8.0,
    16.0,
    24.0,
    32.0
]


# ------------------------------------------------------------
# Calculate channel-wise results
# ------------------------------------------------------------

results = {}

for frequency in frequencies_to_check:

    raw_values = []
    filtered_values = []
    changes = []

    for channel in range(64):

        raw_signal = trial[channel]

        filtered_signal = filtered[channel]

        rf, rm = fft_magnitude(
            raw_signal
        )

        ff, fm = fft_magnitude(
            filtered_signal
        )

        raw_value = magnitude_at(
            rf,
            rm,
            frequency
        )

        filtered_value = magnitude_at(
            ff,
            fm,
            frequency
        )

        if raw_value != 0:

            change = (
                (filtered_value - raw_value)
                / raw_value
                * 100
            )

        else:

            change = 0.0

        raw_values.append(raw_value)

        filtered_values.append(
            filtered_value
        )

        changes.append(change)

    results[frequency] = {
        "raw": np.array(raw_values),
        "filtered": np.array(filtered_values),
        "change": np.array(changes)
    }


# ============================================================
# Print statistics
# ============================================================

for frequency in frequencies_to_check:

    r = results[frequency]

    print()
    print("=" * 75)
    print(f"Frequency: {frequency:.1f} Hz")
    print("=" * 75)

    print(
        f"Raw mean       : {np.mean(r['raw']):.2f}"
    )

    print(
        f"Filtered mean  : {np.mean(r['filtered']):.2f}"
    )

    print(
        f"Raw median     : {np.median(r['raw']):.2f}"
    )

    print(
        f"Filtered median: {np.median(r['filtered']):.2f}"
    )

    print(
        f"Mean change    : {np.mean(r['change']):.2f}%"
    )

    print(
        f"Median change  : {np.median(r['change']):.2f}%"
    )

    print(
        f"Minimum change : {np.min(r['change']):.2f}%"
    )

    print(
        f"Maximum change : {np.max(r['change']):.2f}%"
    )


# ============================================================
# Channel-wise 8 Hz results
# ============================================================

r = results[8.0]

print()
print("=" * 75)
print("8 Hz: Channel-wise Results")
print("=" * 75)

print()

print(
    f"{'Channel':<10}"
    f"{'Raw':<15}"
    f"{'Filtered':<15}"
    f"{'Change %':<15}"
)

print("-" * 55)

for channel in range(64):

    print(
        f"{channel + 1:<10}"
        f"{r['raw'][channel]:<15.2f}"
        f"{r['filtered'][channel]:<15.2f}"
        f"{r['change'][channel]:<15.2f}"
    )


# ============================================================
# Identify strongest changes
# ============================================================

changes = r["change"]

largest_loss_indices = np.argsort(
    changes
)[:10]

largest_gain_indices = np.argsort(
    changes
)[-10:][::-1]


print()
print("=" * 75)
print("10 Channels With Largest 8 Hz Reduction")
print("=" * 75)

for index in largest_loss_indices:

    print(
        f"Channel {index + 1:02d}: "
        f"{changes[index]:.2f}%"
    )


print()
print("=" * 75)
print("10 Channels With Largest 8 Hz Increase")
print("=" * 75)

for index in largest_gain_indices:

    print(
        f"Channel {index + 1:02d}: "
        f"{changes[index]:.2f}%"
    )


# ============================================================
# Check first/last samples
# ============================================================

print()
print("=" * 75)
print("Boundary Check")
print("=" * 75)

for channel in [0, 1, 9, 19, 29, 39, 49, 59, 63]:

    raw_std = np.std(
        trial[channel]
    )

    filtered_std = np.std(
        filtered[channel]
    )

    print(
        f"Channel {channel + 1:02d}: "
        f"raw STD = {raw_std:.3f}, "
        f"filtered STD = {filtered_std:.3f}"
    )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 75)
print("STEP 7D COMPLETE")
print("=" * 75)
print()