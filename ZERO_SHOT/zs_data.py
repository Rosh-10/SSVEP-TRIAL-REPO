"""Stage-1 data module: load -> crop -> filter banks A/B/C -> FB_stand -> LOSO folds."""
import json
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import cheb1ord, cheby1, filtfilt, sosfiltfilt
import zs_config as C

# ---------- loading ----------
def sub_path(sid):
    for n in (f"S{sid}.mat", f"S{sid:02d}.mat"):          # repo uses S1.mat; readme says S01.mat
        if (C.DATA_DIR / n).exists(): return C.DATA_DIR / n
    raise FileNotFoundError(f"S{sid}.mat / S{sid:02d}.mat not in {C.DATA_DIR}")

def load_raw64(sid):
    return loadmat(sub_path(sid))["data"]                  # [REPO] key 'data', (64,1500,40,6)

def load_raw9(sid):
    """(9,1500,40,6) float64, cached. Channel selection only; no cropping here (so T2 is explicit)."""
    C.CACHE_DIR.mkdir(exist_ok=True, parents=True)
    f = C.CACHE_DIR / f"S{sid}_ch9.npy"
    if f.exists(): return np.load(f)
    d = load_raw64(sid)[C.CHANNELS].astype(np.float64)
    np.save(f, d); return d

def load_freq_phase():
    m = loadmat(C.DATA_DIR / "Freq_Phase.mat")
    freqs = np.asarray(m["freqs"]).ravel().astype(float)   # keyboard-layout order, NOT sorted
    ph_mat = np.asarray(m["phases"]).ravel().astype(float) if "phases" in m else None
    return freqs, C.PHASE, ph_mat

def window_idx(sl):
    T = int(sl * C.FS); return C.LATENCY, C.LATENCY + T, T

def get_trials(sid, sl):
    """-> X (240,9,T), y (240,), blk (240,). Order = label-major, block-minor (index = label*6+block) [REPO]."""
    s, e, T = window_idx(sl)
    w = load_raw9(sid)[:, s:e]                              # (9,T,40,6)
    X = w.transpose(2, 3, 0, 1).reshape(C.N_CLS * C.N_BLK, len(C.CHANNELS), T)
    return X, np.repeat(np.arange(C.N_CLS), C.N_BLK), np.tile(np.arange(C.N_BLK), C.N_CLS)

# ---------- filter banks ----------
def _cheb_ba(pb, sb, hp, hs, fs, gpass=3, gstop=40, rp=0.5):
    nyq = fs / 2
    N, Wn = cheb1ord([pb / nyq, hp / nyq], [sb / nyq, hs / nyq], gpass, gstop)
    B, A = cheby1(N, rp, Wn, "bandpass")
    return B, A, N

def bank_A_coefs(fs=C.FS):   # DNN path, 3 bands
    return [_cheb_ba(p, s, C.A_HI_PASS, C.A_HI_STOP, fs, C.A_GPASS, C.A_GSTOP, C.A_RP) for p, s in zip(C.A_PASS, C.A_STOP)]

def bank_C_coefs(fs=C.FS):   # FBCCA baseline, 5 bands
    return [_cheb_ba(p, s, C.C_HI_PASS, C.C_HI_STOP, fs) for p, s in zip(C.C_PASS, C.C_STOP)]

def bank_B_sos(fs=C.FS):     # FBTtCCA bank, order 15, SOS
    return [cheby1(C.B_ORDER, C.B_RP, wp, btype="bandpass", output="sos", fs=fs) for wp in C.B_WP]

def _apply_ba(X, coefs):
    out = []
    for B, A, _ in coefs:
        out.append(filtfilt(B, A, X, axis=-1, padtype="odd", padlen=3 * (max(len(B), len(A)) - 1)))  # [REPO]
    return np.stack(out, axis=-3)                           # (...,bands,C,T)

def apply_bank_A(X): return _apply_ba(X, bank_A_coefs())   # (n,9,T)->(n,3,9,T)
def apply_bank_C(X): return _apply_ba(X, bank_C_coefs())   # (n,9,T)->(n,5,9,T)
def apply_bank_B(X):   # [MY CHOICE] sosfiltfilt defaults (odd pad); repo's CCA.py (FBTtCCA) NOT attached -> unverified
    return np.stack([sosfiltfilt(s, X, axis=-1) for s in bank_B_sos()], axis=-3)

def fb_stand(x):
    """[REPO] function.FB_stand: per-trial, per-band, per-channel z-score over time (ddof=0). x (...,C,T)."""
    m = x.mean(-1, keepdims=True); s = x.std(-1, keepdims=True)
    return (x - m) / np.where(s == 0, 1.0, s)

def prep(sid, sl):
    """Everything one subject contributes: raw (for baselines) and standardized bank-A tensor (for the GNN)."""
    X, y, blk = get_trials(sid, sl)
    return dict(raw=X, A=fb_stand(apply_bank_A(X)), y=y, blk=blk, sid=np.full(len(y), sid))

# ---------- LOSO folds ----------
def make_fold(test_sid, n_val=C.N_VAL, seed=C.SEED, subjects=None):
    subjects = subjects or C.SUBJECTS
    src = [s for s in subjects if s != test_sid]
    rng = np.random.RandomState(seed * 1000 + test_sid)
    val = sorted(int(v) for v in rng.choice(src, n_val, replace=False))
    return dict(test=test_sid, val=val, train=[s for s in src if s not in val], seed=seed)

def build_fold(fold, sl):
    """Concatenate subjects per split. Returns dict split -> arrays (+ 'sid' per trial for leakage audit)."""
    out = {}
    for name, ids in (("train", fold["train"]), ("val", fold["val"]), ("test", [fold["test"]])):
        parts = [prep(s, sl) for s in ids]
        out[name] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return out

def save_fold_manifest(sl_list=(0.5, 1.0), subjects=None):
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    folds = [make_fold(s, subjects=subjects) for s in (subjects or C.SUBJECTS)]
    (C.RESULTS_DIR / "stage1_folds.json").write_text(json.dumps(dict(seed=C.SEED, n_val=C.N_VAL, folds=folds), indent=1))
    return folds
