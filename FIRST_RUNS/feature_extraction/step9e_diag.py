"""Why does argmax-at-target fail? (S1, spectrum5s). Training-free."""
import numpy as np, importlib.util as u
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
s_ = u.spec_from_file_location('config', Path(__file__).resolve().parent.parent/'config.py'); cfg = u.module_from_spec(s_); s_.loader.exec_module(cfg)
f = loadmat(cfg.PROJECT_ROOT/'data/raw/benchmark/Freq_Phase.mat')['freqs'].ravel()
tgt = np.round((f-8)/0.2).astype(int)[:, None]                                   # (40,1)
X = np.load(cfg.PROJECT_ROOT/'feature_extraction/S1_spectrum5s.npz')['spectrum']  # (40,6,64,280)

# 1) bin offset of the winning bin at Oz
off = X[:, :, 61, :40].argmax(-1) - tgt
print('Oz: exact %.3f | +-1 bin %.3f | +-2..5 bins %.3f | farther %.3f' % tuple(
    [(off == 0).mean(), (abs(off) == 1).mean(), ((abs(off) >= 2) & (abs(off) <= 5)).mean(), (abs(off) > 5).mean()]))

# 2) hit rate per channel
hit = (X[..., :40].argmax(-1) == tgt[:, :, None]).mean((0, 1))                    # (64,)
print('best channels (python idx, hit rate):', [(int(i), round(float(hit[i]), 3)) for i in np.argsort(-hit)[:6]])

# 3) local SNR: bin / mean of the other 10 bins in an 11-bin neighbourhood
snr = X / ((uniform_filter1d(X, 11, axis=-1, mode='nearest')*11 - X) / 10)
print('Oz hit rate, SNR spectrum: %.3f' % (snr[:, :, 61, :40].argmax(-1) == tgt).mean())
print('best-channel hit rate, SNR spectrum: %.3f' % (snr[..., :40].argmax(-1) == tgt[:, :, None]).mean((0, 1)).max())
