import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt


# ============================================================
# Step 7B: Apply Filter to One EEG Trial
#
# Subject : S1
# Target  : 1
# Block   : 1
#
# Target frequency = 8.0 Hz
#
# Filter:
#   5th-order Butterworth
#   6-65 Hz
#   zero-phase filtering
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

FS = 250.0

LOW_CUTOFF = 6.0
HIGH_CUTOFF = 65.0

FILTER_ORDER = 5

SUBJECT = 1

TARGET = 1

BLOCK = 1

TARGET_FREQUENCY = 8.0


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

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


DATA_FILE = os.path.join(
    DATA_DIR,
    f"S{SUBJECT}.mat"
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print()
print("=" * 70)
print("Step 7B: Apply Filter to One Trial")
print("=" * 70)
print()

print(f"Loading: {DATA_FILE}")

mat = sio.loadmat(
    DATA_FILE
)

data = mat["data"]

print(
    f"Full dataset shape: {data.shape}"
)


# ------------------------------------------------------------
# Extract one trial
#
# MATLAB-style indexing:
#
# Subject 1
# Target 1
# Block 1
#
# Python:
# target index = 0
# block index  = 0
# ------------------------------------------------------------

trial = data[
    :,
    :,
    TARGET - 1,
    BLOCK - 1
]


print(
    f"Trial shape: {trial.shape}"
)

print(
    f"Channels: {trial.shape[0]}"
)

print(
    f"Samples: {trial.shape[1]}"
)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [
        LOW_CUTOFF,
        HIGH_CUTOFF
    ],
    btype="bandpass",
    fs=FS,
    output="sos"
)


print()
print("Butterworth filter created.")


# ------------------------------------------------------------
# Apply zero-phase filtering
#
# Filtering is performed independently for each channel.
#
# axis=1 means filtering across time.
# ------------------------------------------------------------

filtered_trial = sosfiltfilt(
    sos,
    trial,
    axis=1
)


print(
    "Filtering completed."
)

print(
    f"Filtered trial shape: "
    f"{filtered_trial.shape}"
)


# ------------------------------------------------------------
# Select representative channel
#
# Channel 1 = Python index 0
# ------------------------------------------------------------

CHANNEL = 0

raw_signal = trial[
    CHANNEL,
    :
]

filtered_signal = filtered_trial[
    CHANNEL,
    :
]


# ------------------------------------------------------------
# Time axis
# ------------------------------------------------------------

time = np.arange(
    trial.shape[1]
) / FS


# ------------------------------------------------------------
# FFT function
# ------------------------------------------------------------

def calculate_fft(signal):

    n = len(signal)

    signal = (
        signal
        - np.mean(signal)
    )

    fft_values = np.fft.rfft(
        signal
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / FS
    )

    magnitude = np.abs(
        fft_values
    )

    return frequencies, magnitude


raw_freq, raw_mag = calculate_fft(
    raw_signal
)

filtered_freq, filtered_mag = calculate_fft(
    filtered_signal
)


# ------------------------------------------------------------
# Find magnitude near frequency
# ------------------------------------------------------------

def get_magnitude(
    frequencies,
    magnitude,
    target_frequency
):

    index = np.argmin(
        np.abs(
            frequencies
            - target_frequency
        )
    )

    return magnitude[index]


important_frequencies = [
    8.0,
    16.0,
    24.0,
    32.0,
    40.0,
    48.0,
    56.0,
    63.2
]


# ------------------------------------------------------------
# Print FFT comparison
# ------------------------------------------------------------

print()
print("=" * 70)
print("FFT Comparison")
print("=" * 70)
print()

print(
    f"{'Frequency':>12}"
    f"{'Raw':>15}"
    f"{'Filtered':>15}"
    f"{'Change %':>15}"
)

print("-" * 60)


for frequency in important_frequencies:

    raw_value = get_magnitude(
        raw_freq,
        raw_mag,
        frequency
    )

    filtered_value = get_magnitude(
        filtered_freq,
        filtered_mag,
        frequency
    )

    if raw_value != 0:

        change = (
            (filtered_value - raw_value)
            / raw_value
            * 100
        )

    else:

        change = 0

    print(
        f"{frequency:8.1f} Hz"
        f"{raw_value:15.2f}"
        f"{filtered_value:15.2f}"
        f"{change:14.2f}%"
    )


# ============================================================
# Plot 1: Raw vs filtered time signal
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    raw_signal,
    label="Raw"
)

plt.plot(
    time,
    filtered_signal,
    label="Filtered 6-65 Hz"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "S1 Target 1 Block 1 - "
    "Raw vs Filtered EEG"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


time_plot = os.path.join(
    RESULTS_DIR,
    "step7b_time_domain.png"
)

plt.savefig(
    time_plot,
    dpi=300
)

plt.close()


# ============================================================
# Plot 2: Raw FFT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    raw_freq,
    raw_mag
)

plt.axvline(
    8,
    linestyle="--",
    label="8 Hz"
)

plt.axvline(
    16,
    linestyle="--",
    label="16 Hz"
)

plt.axvline(
    24,
    linestyle="--",
    label="24 Hz"
)

plt.axvline(
    32,
    linestyle="--",
    label="32 Hz"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Raw EEG Spectrum - S1 Target 1 Block 1"
)

plt.xlim(
    0,
    100
)

plt.legend()

plt.grid(True)

plt.tight_layout()


raw_fft_plot = os.path.join(
    RESULTS_DIR,
    "step7b_raw_fft.png"
)

plt.savefig(
    raw_fft_plot,
    dpi=300
)

plt.close()


# ============================================================
# Plot 3: Filtered FFT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    filtered_freq,
    filtered_mag
)

plt.axvline(
    8,
    linestyle="--",
    label="8 Hz"
)

plt.axvline(
    16,
    linestyle="--",
    label="16 Hz"
)

plt.axvline(
    24,
    linestyle="--",
    label="24 Hz"
)

plt.axvline(
    32,
    linestyle="--",
    label="32 Hz"
)

plt.axvline(
    65,
    linestyle=":",
    label="65 Hz cutoff"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Filtered EEG Spectrum - "
    "S1 Target 1 Block 1"
)

plt.xlim(
    0,
    100
)

plt.legend()

plt.grid(True)

plt.tight_layout()


filtered_fft_plot = os.path.join(
    RESULTS_DIR,
    "step7b_filtered_fft.png"
)

plt.savefig(
    filtered_fft_plot,
    dpi=300
)

plt.close()


# ============================================================
# Plot 4: Raw vs filtered FFT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    raw_freq,
    raw_mag,
    label="Raw"
)

plt.plot(
    filtered_freq,
    filtered_mag,
    label="Filtered 6-65 Hz"
)

plt.axvline(
    65,
    linestyle=":",
    label="65 Hz cutoff"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Raw vs Filtered EEG Spectrum"
)

plt.xlim(
    0,
    100
)

plt.legend()

plt.grid(True)

plt.tight_layout()


comparison_plot = os.path.join(
    RESULTS_DIR,
    "step7b_fft_comparison.png"
)

plt.savefig(
    comparison_plot,
    dpi=300
)

plt.close()


# ============================================================
# Complete
# ============================================================

print()
print("=" * 70)
print("Plots saved:")
print("=" * 70)

print(time_plot)
print(raw_fft_plot)
print(filtered_fft_plot)
print(comparison_plot)

print()
print("=" * 70)
print("STEP 7B COMPLETE")
print("=" * 70)
print()