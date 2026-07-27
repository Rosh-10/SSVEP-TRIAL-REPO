"""
Step 6a: Empirical FFT check.

Objective: before deciding on filter cutoffs, confirm empirically where
SSVEP signal energy actually lives - expect concentration near 8 Hz
(fundamental for Target 1) and its harmonics (16, 24 Hz), and check for
any leftover 50 Hz powerline noise despite the recording-time notch filter.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loading import load_subject, extract_trial

import numpy as np
import matplotlib.pyplot as plt

FS = 250  # sampling rate in Hz, per readme.txt (downsampled from 1000 Hz)

# --- Load one trial: Subject 1, Target 1 (8.0 Hz), Block 1, channel Oz ---
eeg = load_subject(1)
trial = extract_trial(eeg, target=1, block=1, channel=62)  # 62 = Oz, per .loc file

# --- Compute FFT ---
n = len(trial)                          # 1500 time points
fft_vals = np.fft.fft(trial)             # complex-valued spectrum
fft_freqs = np.fft.fftfreq(n, d=1/FS)    # corresponding frequency for each bin

# FFT output is symmetric for real-valued input; only keep the first half
# (0 Hz up to Nyquist frequency = FS/2 = 125 Hz)
half = n // 2
freqs_pos = fft_freqs[:half]
magnitude = np.abs(fft_vals[:half])      # magnitude = how much energy at this frequency

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(freqs_pos, magnitude)
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT: Subject 1, Target 1 (8.0 Hz), Block 1, Channel Oz")
plt.axvline(8, color='red', linestyle='--', alpha=0.5, label='8 Hz (fundamental)')
plt.axvline(16, color='orange', linestyle='--', alpha=0.5, label='16 Hz (2nd harmonic)')
plt.axvline(24, color='green', linestyle='--', alpha=0.5, label='24 Hz (3rd harmonic)')
plt.axvline(50, color='purple', linestyle='--', alpha=0.5, label='50 Hz (powerline)')
plt.legend()
plt.tight_layout()

save_path = Path(__file__).resolve().parent.parent / "results" / "step6a_fft_check.png"
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()

# --- Print the actual peak values near frequencies of interest, for numeric confirmation ---
def magnitude_near(target_hz, tolerance=0.3):
    mask = np.abs(freqs_pos - target_hz) <= tolerance
    idx = np.argmax(magnitude[mask])
    return freqs_pos[mask][idx], magnitude[mask][idx]

for f in [8, 16, 24, 50]:
    actual_freq, mag = magnitude_near(f)
    print(f"Near {f} Hz -> closest bin at {actual_freq:.3f} Hz, magnitude = {mag:.2f}")
