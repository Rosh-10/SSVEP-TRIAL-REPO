"""STEP 11a: data loader. Put in SSVEP-TRIAL-REPO/training/data_loader.py
Input : feature_extraction/S{n}_spectrum5s.npz (key 'spectrum', (40,6,64,280))
Output: X (240,64,280) log-SNR float32, y (240,) labels 0-39 (keyboard order), b (240,) block 0-5
"""
import numpy as np, importlib.util as u
from pathlib import Path
from scipy.ndimage import uniform_filter1d
s_ = u.spec_from_file_location('config', Path(__file__).resolve().parent.parent/'config.py'); cfg = u.module_from_spec(s_); s_.loader.exec_module(cfg)

def snr_log(X, w=11):
    """log of (bin / mean of the other w-1 neighbouring bins). Deterministic per trial -> no leakage."""
    m = (uniform_filter1d(X, w, axis=-1, mode='nearest')*w - X) / (w-1)
    return np.log((X+1e-8) / (m+1e-8))

def load_subject(s):
    X = np.load(cfg.PROJECT_ROOT/f'feature_extraction/S{s}_spectrum5s.npz')['spectrum']      # (40,6,64,280)
    X = snr_log(X.astype(np.float64)).reshape(240, 64, 280)                                  # trial index = target*6 + block
    return X.astype(np.float32), np.repeat(np.arange(40), 6), np.tile(np.arange(6), 40)

def split(X, y, b, test_block, val_block):
    """Train = remaining 4 blocks. Mean/std fit on TRAIN only, applied to val/test."""
    tr, va, te = ~np.isin(b, [test_block, val_block]), b == val_block, b == test_block
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    f = lambda m: ((X[m]-mu)/sd).astype(np.float32)
    return (f(tr), y[tr]), (f(va), y[va]), (f(te), y[te])

if __name__ == '__main__':
    X, y, b = load_subject(1)
    print(X.shape, y.shape, b.shape, np.isfinite(X).all(), X.dtype)
    print('trial 0-5 labels:', y[:6], '| blocks:', b[:6])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, b, test_block=5, val_block=4)
    print('train/val/test:', Xtr.shape, Xva.shape, Xte.shape, '| classes per split:', len(set(ytr)), len(set(yva)), len(set(yte)))
    print('train mean/std (should be ~0/~1): %.3f %.3f' % (Xtr.mean(), Xtr.std()))
