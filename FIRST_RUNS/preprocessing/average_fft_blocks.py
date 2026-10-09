import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.data_loading import load_subject, extract_trial

# -----------------------------
# Parameters (1-indexed)
# -----------------------------
SUBJECT = 1
TARGET = 1
CHANNEL = 62      # Oz

SAMPLING_RATE = 250  # Hz

# -----------------------------
# Load subject
# -----------------------------
eeg = load_subject(SUBJECT)

# Frequency axis (same for every trial)
freqs = np.fft.rfftfreq(1500, d=1 / SAMPLING_RATE)

fft_sum = np.zeros(len(freqs))

# -----------------------------
# Process all six blocks
# -----------------------------
for block in range(1, 7):

    signal = extract_trial(
        eeg,
        target=TARGET,
        block=block,
        channel=CHANNEL
    )

    fft = np.fft.rfft(signal)

    magnitude = np.abs(fft)

    fft_sum += magnitude

# -----------------------------
# Average
# -----------------------------
average_magnitude = fft_sum / 6

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(freqs, average_magnitude)

plt.xlim(0,40)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title(
    f"Average FFT Magnitude\n"
    f"Subject {SUBJECT}, Target {TARGET}, Channel Oz"
)

plt.grid(True)

plt.show()