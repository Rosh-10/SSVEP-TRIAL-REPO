"""Stage-2 baselines: FBCCA and FBTtCCA (repo-exact wrappers).
All methods operate on raw (n,9,T) float64, no fb_stand. [REPO] unless tagged [MY CHOICE]."""
import sys, types, importlib.util
import numpy as np
from pathlib import Path
import zs_config as C


# ---------- load official modules once ----------
def _load_official():
    od = Path(C.ROOT / "official")
    # num_class=40 required: fbcca() uses config.num_class internally
    st = types.SimpleNamespace(
        Dataset=1, C=9, rfs=250, num_class=40, Nh=C.NH, Nm=C.NM,
        T=125, sample_length=0.5, nBlock=C.N_BLK,
        list_freqs=np.linspace(8, 15.8, 40), list_phase=C.PHASE,
        dropout=0.95)
    m = types.ModuleType("config")
    m.config = st
    sys.modules["config"] = m
    def _load(name):
        sp = importlib.util.spec_from_file_location("off_" + name, od / f"{name}.py")
        mo = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mo)
        return mo
    return {n: _load(n) for n in ("function", "FBCCA", "CCA")}


_off = None
def off():
    global _off
    if _off is None:
        _off = _load_official()
    return _off


# ---------- ITR ----------
def itr(p, n=40, t=None):
    """[REPO] main.py ITR formula. t = T/250 + 0.5 seconds."""
    if p <= 1 / n:
        return 0.0
    return (np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))) * 60 / t


# ---------- CCA reference signals ----------
def cca_reference(freqs, fs, T_samples, n_harmonics):
    """[REPO] function.cca_reference: takes T in SAMPLES (num_smpls). sin/cos, no phase."""
    return off()["function"].cca_reference(freqs, fs, T_samples, n_harmonics)


def generate_cca_references(freqs, fs, sl, n_harmonics):
    """[REPO] function.generate_cca_references: takes T in SECONDS (duration).
    Note: multiplies phases by pi internally; C.PHASE already in radians
    -> double-pi bug, kept repo-exact. [REPO]"""
    return off()["function"].generate_cca_references(
        freqs, fs, sl, phases=C.PHASE, n_harmonics=n_harmonics)


# ---------- FBCCA ----------
def run_fbcca(X, freqs, fs=C.FS, n_harmonics=C.NH):
    """
    X: (n_trials, 9, T) raw EEG, label-major order (index = label*N_BLK + block).

    [REPO] fbcca(eeg, y_ref, ...) expects eeg shape (num_targs=40, C, T) — one
    block of 40 targets, raw unfiltered (applies filterbank internally).
    Returns (results, features):
      results:  (40,) argmax per target — not used
      features: (40, 40) weighted CCA scores (num_targs, num_class)

    Iterate over 6 blocks, call fbcca() once per block, reassemble label-major.
    Returns preds (240,) int, scores (240, 40) float.
    """
    FB = off()["FBCCA"]
    n_trials, n_ch, T = X.shape
    n_blocks = n_trials // C.N_CLS  # 6
    # cca_reference takes T in samples
    Yf = cca_reference(freqs, fs, T, n_harmonics)  # (40, 2*Nh, T)

    # X label-major: X[label*6 + block]
    # -> (N_CLS, N_BLK, 9, T) -> (N_BLK, N_CLS, 9, T)
    Xb = X.reshape(C.N_CLS, n_blocks, n_ch, T).transpose(1, 0, 2, 3)

    all_feat = np.zeros((n_blocks, C.N_CLS, C.N_CLS))
    for b in range(n_blocks):
        _, feat = FB.fbcca(Xb[b], Yf, fs, num_harms=n_harmonics, num_fbs=5)
        all_feat[b] = feat  # (40, 40)

    # (N_BLK, N_CLS, N_CLS) -> (N_CLS, N_BLK, N_CLS) -> (240, 40)
    scores = all_feat.transpose(1, 0, 2).reshape(n_trials, C.N_CLS)
    preds = np.argmax(scores, axis=1)
    return preds, scores


# ---------- FBTtCCA with Yf only ----------
def make_fbttcca_yf(freqs, sl, fs=C.FS, n_harmonics=5):  # was C.NH=4
    """
    [REPO] main.py estimator_1 path: FBTtCCA with sin/cos Yf reference only.
    sl: window length in SECONDS (e.g. 0.5 or 1.0) — generate_cca_references
        uses T as duration in seconds: t = linspace(0, T, int(T*srate)).
    Returns (est, Yf).
    """
    import zs_data as D
    CCA_mod = off()["CCA"]
    sos = D.bank_B_sos()
    filterweights = np.array([(k+1)**-1.25 + 0.25 for k in range(5)])
    est = CCA_mod.FBTtCCA(filterbank=sos, n_components=1, filterweights=filterweights)
    # generate_cca_references takes sl in seconds, not samples
    Yf = generate_cca_references(freqs, fs, sl, n_harmonics)
    return est, Yf


def fit_fbttcca(est, X_train, y_train, Yf):
    """
    X_train: (n_trials, 9, T) raw EEG from source subjects.
    y_train: (n_trials,) int labels 0..39.
    [REPO] FBTtCCA.fit(X, y, Yf=Yf).
    """
    est.fit(X_train, y_train, Yf=Yf)
    return est


def pred_fbttcca(est, X_test):
    """
    [REPO] FBTtCCA.predict(X) returns (labels, features).
    labels:   (n_trials,) predicted class indices
    features: (n_trials, n_classes) scores
    Returns preds (n_trials,) int, scores (n_trials, 40) float.
    """
    labels, features = est.predict(X_test)
    return labels.astype(int), features