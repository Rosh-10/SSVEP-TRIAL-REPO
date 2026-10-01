"""Training-free check: fraction of trials whose strongest Oz bin in 8-15.8 Hz equals the target frequency (S1).
Compares old spectrum (full 1500-sample window) vs new spectrum5s (exact 5 s window)."""
import numpy as np, importlib.util as u
from pathlib import Path
from scipy.io import loadmat
s_ = u.spec_from_file_location('config', Path(__file__).resolve().parent.parent/'config.py'); cfg = u.module_from_spec(s_); s_.loader.exec_module(cfg)
f = loadmat(cfg.PROJECT_ROOT/'data/raw/benchmark/Freq_Phase.mat')['freqs'].ravel()
tgt = np.round((f-8)/0.2).astype(int)[:, None]                                      # target bin index, (40,1)
for name in ['spectrum', 'spectrum5s']:
    S = np.load(cfg.PROJECT_ROOT/f'feature_extraction/S1_{name}.npz')['spectrum'][:, :, 61, :40]   # Oz, 8.0-15.8 Hz -> (40,6,40)
    print(name, (S.argmax(-1) == tgt).mean())
