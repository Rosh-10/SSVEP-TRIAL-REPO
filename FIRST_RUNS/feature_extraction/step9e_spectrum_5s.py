"""STEP 9e: exact 0.2 Hz-bin spectrum from the 5 s flicker window.
Input : data/processed/S{n}.npz (key 'data', (64,1500,40,6))
Output: feature_extraction/S{n}_spectrum5s.npz (key 'spectrum', (40,6,64,280)), 8.0-63.8 Hz
Window: samples 160:1410 = 0.5 s cue + 0.14 s latency, then 5 s (1250 samples) -> rfft bin k = k*0.2 Hz.
"""
import numpy as np, importlib.util as u
from pathlib import Path
s_ = u.spec_from_file_location('config', Path(__file__).resolve().parent.parent/'config.py'); cfg = u.module_from_spec(s_); s_.loader.exec_module(cfg)
START, N = 160, 1250
for s in range(1, 36):
    x = np.load(cfg.PROCESSED_DIR/f'S{s}.npz')['data'][:, START:START+N]            # (64,1250,40,6)
    sp = np.abs(np.fft.rfft(x, axis=1))[:, 40:320]                                  # bin k = f*5 -> (64,280,40,6)
    sp = sp.transpose(2, 3, 0, 1).astype(np.float32)                                # (40,6,64,280)
    assert sp.shape == (40, 6, 64, 280) and np.isfinite(sp).all()
    np.savez_compressed(cfg.PROJECT_ROOT/'feature_extraction'/f'S{s}_spectrum5s.npz', spectrum=sp)
    print(f'S{s}: OK {sp.shape}')
