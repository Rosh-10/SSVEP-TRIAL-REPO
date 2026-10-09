"""
Step 6b (part 1): Inspect Freq_Phase.mat structure.

Objective: find out the exact variable names scipy.io.loadmat gives us,
before writing a reusable loader function. We don't guess key names.
"""

from pathlib import Path
from scipy.io import loadmat

# Path: from preprocessing/ up to project root, then into data/raw/benchmark/
path = Path(__file__).resolve().parent.parent / "data" / "raw" / "benchmark" / "Freq_Phase.mat"

mat = loadmat(path)

print("Keys found in Freq_Phase.mat:")
for key in mat.keys():
    print(f"  {key!r}  ->  type: {type(mat[key])}", end="")
    if hasattr(mat[key], "shape"):
        print(f", shape: {mat[key].shape}")
    else:
        print()