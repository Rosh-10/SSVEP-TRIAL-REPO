from step6b_load_freqphase import load_freq_phase
import numpy as np

freqs, phases = load_freq_phase()

FREQS_HARDCODED = np.array([
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
])

print("Match:", np.allclose(freqs, FREQS_HARDCODED))
print("Loaded freqs:", freqs)
print("Hardcoded    :", FREQS_HARDCODED)