"""STEP 11c: training-free harmonic-sum SNR rule (no learning -> no leakage, uses all 240 trials).
Run: python training1/step11c_harmonic_rule.py [subjects...]   (place next to data_loader.py)
Score(k) = mean over occipital channels and harmonics m=1..H of log-SNR at m*f_k; prediction = argmax_k."""
import sys, numpy as np
from scipy.io import loadmat
from data_loader import load_subject, cfg
f = loadmat(cfg.PROJECT_ROOT/'data/raw/benchmark/Freq_Phase.mat')['freqs'].ravel()
lab = np.load(cfg.PROJECT_ROOT/'graph_construction/electrode_adjacency.npz')['labels']
occ = [i for i, l in enumerate(lab) if str(l).upper().startswith(('O', 'PO'))]
print('occipital channels used:', [(i, str(lab[i])) for i in occ])
idx = np.round((np.arange(1, 5)[None] * f[:, None] - 8) / 0.2).astype(int)           # (40,4): bin index of m*f_k
for s in map(int, sys.argv[1:] or [1, 15, 30]):
    X, y, b = load_subject(s)
    S = X[:, occ][:, :, idx]                                                          # (240, n_occ, 40, 4)
    print(f'S{s}:', ' | '.join(f'H={h}: {(S[..., :h].mean((1, 3)).argmax(-1) == y).mean():.3f}' for h in range(1, 5)))
