"""
Core data loading / extraction utilities for the Benchmark SSVEP dataset.

IMPORTANT - INDEXING CONVENTION:
The readme.txt, Sub_info.txt, and the 64-channels.loc file all use MATLAB's
1-indexed convention (Target 1, Block 1, Channel 1 = the FIRST one).

Python/NumPy is 0-indexed. To avoid silent bugs, every function in this
file accepts 1-indexed arguments (matching paper/dataset documentation)
and internally converts to 0-indexed before touching the array.

Example of the bug this prevents:
    eeg[:, :, 1, 1]  # WRONG if you meant "Target 1, Block 1"
                     # This actually gives Target 2, Block 2 (index 1 = 2nd item)

    extract_trial(eeg, target=1, block=1, channel=62)  # CORRECT
                     # Internally converts to index 0 for all three
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BENCHMARK_DIR

import scipy.io as sio
import numpy as np


def load_subject(subject_num: int) -> np.ndarray:
    """
    Load one subject's full EEG data array.

    Parameters
    ----------
    subject_num : int
        1-indexed subject number (1 to 35), matching filename S1.mat ... S35.mat

    Returns
    -------
    np.ndarray, shape (64, 1500, 40, 6)
        [Electrode, Time, Target, Block] - still in MATLAB's original
        1-indexed CONCEPTUAL ordering, but now a 0-indexed Python array.
    """
    path = BENCHMARK_DIR / f"S{subject_num}.mat"
    mat = sio.loadmat(str(path))
    return mat['data']


def extract_trial(eeg: np.ndarray, target: int, block: int, channel: int) -> np.ndarray:
    """
    Extract one electrode's time series for one target/block combination.

    All arguments are 1-INDEXED, matching paper/dataset documentation
    (e.g. "Target 1" = 8.0 Hz, "Channel 62" = Oz per 64-channels.loc).

    Parameters
    ----------
    eeg : np.ndarray, shape (64, 1500, 40, 6)
        Full subject array as returned by load_subject()
    target : int
        1-indexed target number (1 to 40)
    block : int
        1-indexed block number (1 to 6)
    channel : int
        1-indexed electrode number (1 to 64), per 64-channels.loc

    Returns
    -------
    np.ndarray, shape (1500,)
        Time series for the specified electrode/target/block.
    """
    assert 1 <= target <= 40, f"target must be 1-40, got {target}"
    assert 1 <= block <= 6, f"block must be 1-6, got {block}"
    assert 1 <= channel <= 64, f"channel must be 1-64, got {channel}"

    # Convert 1-indexed (human/paper convention) -> 0-indexed (Python/NumPy)
    target_idx = target - 1
    block_idx = block - 1
    channel_idx = channel - 1

    return eeg[channel_idx, :, target_idx, block_idx]


if __name__ == "__main__":
    # Quick manual test: Subject 1, Target 1 (8.0 Hz), Block 1, channel Oz (62)
    eeg = load_subject(1)
    print("Loaded subject 1, shape:", eeg.shape)

    trial = extract_trial(eeg, target=1, block=1, channel=62)
    print("Extracted trial shape:", trial.shape)
    print("Expected: (1500,)")
    print("First 10 values:", trial[:10])
