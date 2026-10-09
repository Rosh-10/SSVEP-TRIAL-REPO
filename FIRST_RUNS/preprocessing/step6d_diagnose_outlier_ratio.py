"""
Step 6d: Diagnose the outlier ratio (Subject 30, ratio = 2.851).

Objective: distinguish "genuinely strong harmonic" from "noisy/weak
fundamental inflating the ratio" by inspecting raw magnitudes, not just
the ratio.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loading import load_subject, extract_trial
from step6b_load_freqphase import load_freq_phase

import numpy as np

FS = 250


def magnitude_at(freqs_pos, magnitude, target_hz, tolerance=0.3):
    mask = np.abs(freqs_pos - target_hz) <= tolerance
    if not np.any(mask):
        return None
    idx = np.argmax(magnitude[mask])
    return magnitude[mask][idx]


def diagnose(subject_id, target_index, freqs, block=1, channel=62):
    eeg = load_subject(subject_id)
    trial = extract_trial(eeg, target=target_index, block=block, channel=channel)

    n = len(trial)
    fft_vals = np.fft.fft(trial)
    fft_freqs = np.fft.fftfreq(n, d=1 / FS)
    half = n // 2
    freqs_pos = fft_freqs[:half]
    magnitude = np.abs(fft_vals[:half])

    f0 = freqs[target_index - 1]
    f1 = 2 * f0

    mag_f0 = magnitude_at(freqs_pos, magnitude, f0)
    mag_f1 = magnitude_at(freqs_pos, magnitude, f1)

    print(f"Subject {subject_id}, Target {target_index} ({f0} Hz):")
    print(f"  mag_f0 (fundamental) = {mag_f0:.2f}")
    print(f"  mag_f1 (2nd harmonic) = {mag_f1:.2f}")
    print(f"  ratio = {mag_f1 / mag_f0:.3f}")

    return mag_f0, mag_f1


if __name__ == "__main__":
    freqs, _ = load_freq_phase()

    # Re-run all 40 targets for Subject 30 to find which one hit ~2.851
    subj = 30
    best_target, best_ratio = None, -1

    for tgt in range(1, 41):
        eeg = load_subject(subj)
        trial = extract_trial(eeg, target=tgt, block=1, channel=62)
        n = len(trial)
        fft_vals = np.fft.fft(trial)
        fft_freqs = np.fft.fftfreq(n, d=1 / FS)
        half = n // 2
        freqs_pos = fft_freqs[:half]
        magnitude = np.abs(fft_vals[:half])

        f0 = freqs[tgt - 1]
        f1 = 2 * f0
        mag_f0 = magnitude_at(freqs_pos, magnitude, f0)
        mag_f1 = magnitude_at(freqs_pos, magnitude, f1)

        if mag_f0 and mag_f1:
            ratio = mag_f1 / mag_f0
            if ratio > best_ratio:
                best_ratio = ratio
                best_target = tgt

    print(f"\nHighest ratio target for Subject 30: Target {best_target}, ratio = {best_ratio:.3f}\n")

    # Now show the raw magnitudes for that specific target
    diagnose(subj, best_target, freqs)

    # For comparison, show a "normal" target's raw magnitudes too (Target 1)
    print()
    diagnose(subj, 1, freqs)