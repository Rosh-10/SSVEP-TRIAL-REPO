from pathlib import Path
from scipy.io import loadmat
import numpy as np


def load_freq_phase():
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "benchmark" / "Freq_Phase.mat"
    mat = loadmat(path)

    freqs = mat["freqs"].flatten()
    phases = mat["phases"].flatten()

    assert freqs.shape == (40,), f"Expected shape (40,), got {freqs.shape}"
    assert phases.shape == (40,), f"Expected shape (40,), got {phases.shape}"

    return freqs, phases


if __name__ == "__main__":
    freqs, phases = load_freq_phase()
    print("First 5 frequencies:", freqs[:5])
    print("First 5 phases:", phases[:5])
    print("Last 5 frequencies:", freqs[-5:])