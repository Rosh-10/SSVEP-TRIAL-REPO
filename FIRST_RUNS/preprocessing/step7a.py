import numpy as np
from scipy.signal import butter, sosfreqz
import matplotlib.pyplot as plt


# ============================================================
# Step 7A: Butterworth Bandpass Filter Design
#
# Final candidate based on Steps 6C-6G:
#
# Low cutoff  = 6 Hz
# High cutoff = 65 Hz
#
# Sampling rate = 250 Hz
# Filter order = 5
#
# Zero-phase filtering will be performed later using
# scipy.signal.sosfiltfilt()
# ============================================================


FS = 250.0

LOW_CUTOFF = 6.0
HIGH_CUTOFF = 65.0

FILTER_ORDER = 5


print()
print("=" * 70)
print("Step 7A: Butterworth Filter Design")
print("=" * 70)
print()

print(f"Sampling frequency : {FS} Hz")
print(f"Low cutoff         : {LOW_CUTOFF} Hz")
print(f"High cutoff        : {HIGH_CUTOFF} Hz")
print(f"Filter order       : {FILTER_ORDER}")
print()


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


print("Butterworth filter successfully designed.")
print()

print("Second-order-section matrix:")
print(sos)

print()


# ------------------------------------------------------------
# Calculate frequency response
# ------------------------------------------------------------

frequencies, response = sosfreqz(
    sos,
    worN=4096,
    fs=FS
)

magnitude_db = 20 * np.log10(
    np.maximum(np.abs(response), 1e-12)
)


# ------------------------------------------------------------
# Print response at important frequencies
# ------------------------------------------------------------

important_frequencies = [
    6.0,
    8.0,
    15.8,
    24.0,
    31.6,
    47.4,
    63.2,
    65.0,
    70.0,
    80.0,
    100.0
]


print("=" * 70)
print("Filter response at important frequencies")
print("=" * 70)
print()

for freq in important_frequencies:

    index = np.argmin(
        np.abs(frequencies - freq)
    )

    print(
        f"{freq:6.1f} Hz : "
        f"{magnitude_db[index]:8.2f} dB"
    )


# ------------------------------------------------------------
# Plot frequency response
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    frequencies,
    magnitude_db
)

plt.axvline(
    LOW_CUTOFF,
    linestyle="--",
    label="Low cutoff"
)

plt.axvline(
    HIGH_CUTOFF,
    linestyle="--",
    label="High cutoff"
)

plt.axvline(
    15.8,
    linestyle=":",
    label="Maximum fundamental"
)

plt.axvline(
    31.6,
    linestyle=":",
    label="Maximum 2nd harmonic"
)

plt.axvline(
    47.4,
    linestyle=":",
    label="Maximum 3rd harmonic"
)

plt.axvline(
    63.2,
    linestyle=":",
    label="Maximum 4th harmonic"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("5th-Order Butterworth Bandpass Filter: 6–65 Hz")

plt.xlim(0, 100)
plt.ylim(-80, 5)

plt.grid(True)
plt.legend()

plt.tight_layout()


# ------------------------------------------------------------
# Save plot
# ------------------------------------------------------------

RESULTS_DIR = "results"

import os

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

output_path = os.path.join(
    RESULTS_DIR,
    "step7a_filter_response.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()


print()
print(f"Filter response plot saved to:")
print(output_path)

print()
print("=" * 70)
print("STEP 7A COMPLETE")
print("=" * 70)
print()