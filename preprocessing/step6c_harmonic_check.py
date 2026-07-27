"""
Step 6c: Harmonic ratio check across all 40 targets, multiple subjects.

Objective: test whether "2nd harmonic magnitude ~ fundamental magnitude"
(seen for Subject 1, Target 1) holds generally, or was specific to that trial.

We compute harmonic_ratio = magnitude(2*f) / magnitude(f) for every target's
real stimulation frequency f, across a few subjects.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loading import load_subject, extract_trial
from step6b_load_freqphase import load_freq_phase  # adjust import name to your actual filename

import numpy as np

FS = 250

def magnitude_at(freqs_pos, magnitude, target_hz, tolerance=0.3):
    """Find the magnitude of the FFT bin closest to target_hz."""
    mask = np.abs(freqs_pos - target_hz) <= tolerance
    if not np.any(mask):
        return None  # target frequency out of range or no nearby bin
    idx = np.argmax(magnitude[mask])
    return magnitude[mask][idx]


def compute_harmonic_ratio(subject_id, target_index, freqs, block=1, channel=62):
    """
    Returns harmonic_ratio = magnitude(2f) / magnitude(f) for one (subject, target).
    target_index is 1-indexed (matches readme convention: Target 1 ... Target 40).
    """
    eeg = load_subject(subject_id)
    trial = extract_trial(eeg, target=target_index, block=block, channel=channel)

    n = len(trial)
    fft_vals = np.fft.fft(trial)
    fft_freqs = np.fft.fftfreq(n, d=1 / FS)

    half = n // 2
    freqs_pos = fft_freqs[:half]
    magnitude = np.abs(fft_vals[:half])

    f0 = freqs[target_index - 1]        # fundamental, real Hz value
    f1 = 2 * f0                          # 2nd harmonic

    mag_f0 = magnitude_at(freqs_pos, magnitude, f0)
    mag_f1 = magnitude_at(freqs_pos, magnitude, f1)

    if mag_f0 is None or mag_f1 is None or mag_f0 == 0:
        return None  # couldn't compute (e.g. harmonic exceeds Nyquist, or fundamental had zero magnitude)

    return mag_f1 / mag_f0


if __name__ == "__main__":
    freqs, _ = load_freq_phase()

    subjects = [1, 15, 30]  # S01 (experienced), S15 (naive), S30 (naive)
    targets = list(range(1, 41))  # all 40 targets

    results = {}  # {subject_id: [ratio_for_target_1, ratio_for_target_2, ...]}

    for subj in subjects:
        ratios = []
        for tgt in targets:
            ratio = compute_harmonic_ratio(subj, tgt, freqs)
            ratios.append(ratio)
        results[subj] = ratios

        valid_ratios = [r for r in ratios if r is not None]
        print(f"\nSubject {subj:02d}: mean ratio = {np.mean(valid_ratios):.3f}, "
              f"min = {np.min(valid_ratios):.3f}, max = {np.max(valid_ratios):.3f}")

    # Print per-target detail for Subject 1, so we can see any frequency-dependent trend
    print("\n--- Subject 1, per-target detail ---")
    for tgt, ratio in zip(targets, results[1]):
        f0 = freqs[tgt - 1]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        print(f"Target {tgt:2d} ({f0:5.1f} Hz): ratio = {ratio_str}")