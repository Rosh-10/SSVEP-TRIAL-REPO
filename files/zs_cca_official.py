"""zs_cca_official.py - official-repo-faithful pieces for CCA-Net (function.py / FBCCA.py / generate_cca_references)."""
import numpy as np, scipy.signal as ss
from scipy.linalg import eigh, qr, pinv
from scipy.stats import pearsonr
from joblib import Parallel, delayed

def official_filter_bank(X, fs=250):
    """[REPO function.filter_bank] X:(N,C,T) -> (N,3,C,T). cheb1ord ba filters, filtfilt odd pad."""
    pb, sb, out = [6,14,22], [4,10,16], []
    for k in range(3):
        N, Wn = ss.cheb1ord([pb[k]/(fs/2), 90/(fs/2)], [sb[k]/(fs/2), 100/(fs/2)], 3, 40)
        B, A = ss.cheby1(N, 0.5, Wn, 'bandpass')
        out.append(ss.filtfilt(B, A, X, axis=-1, padlen=3*(max(len(B), len(A))-1)))
    return np.stack(out, 1)

def generate_cca_references(freqs, phases, T, fs=250, n_harmonics=5):
    """[REPO function.generate_cca_references] incl. quirks: linspace endpoint, np.pi*phases, phase not x harmonic."""
    t = np.linspace(0, T/fs, T)
    return np.stack([np.stack([f(2*np.pi*h*fr*t + np.pi*ph) for h in range(1, n_harmonics+1) for f in (np.sin, np.cos)])
                     for fr, ph in zip(freqs, phases)])

def cca_reference(freqs, T, fs=250, nh=4):
    """[REPO FBCCA.cca_reference] plain sin/cos, no phase, t=1..T/fs."""
    t = np.arange(1, T+1)/fs
    return np.stack([np.stack([f(2*np.pi*t*h*fr) for h in range(1, nh+1) for f in (np.sin, np.cos)]) for fr in freqs])

def _kernel(X, Y):
    Q, R = qr(Y.T, mode='economic'); P = Q @ Q.T; Z = X.T
    A = (P.T @ Z).T @ (P.T @ Z); B = Z.T @ Z
    _, W = eigh(A, B); U = W[:, ::-1][:, :min(X.shape[0], Y.shape[0])]
    return U, pinv(R) @ Q.T @ X.T @ U

def _one(x, refs):
    r = []
    for ref in refs:
        U, V = _kernel(x - x.mean(-1, keepdims=True), ref - ref.mean(-1, keepdims=True))
        r.append(pearsonr((U[:, :1].T @ x).ravel(), (V[:, :1].T @ ref).ravel())[0])
    return r

def _fb5(X, fs=250):
    pb, sb, out = [6,14,22,30,38], [4,10,16,24,32], []
    for k in range(5):
        N, Wn = ss.cheb1ord([pb[k]/(fs/2), 90/(fs/2)], [sb[k]/(fs/2), 100/(fs/2)], 3, 40)
        B, A = ss.cheby1(N, 0.5, Wn, 'bandpass')
        out.append(ss.filtfilt(B, A, X, axis=-1, padlen=3*(max(len(B), len(A))-1)))
    return out

def fbcca_official(X, refs, fs=250):
    """[REPO FBCCA.fbcca] X:(N,C,T), refs:(40,H,T) -> features (N,40). Exact CCA kernel instead of sklearn NIPALS (diff ~2e-3)."""
    w = np.arange(1, 6)**-1.25 + 0.25; f = 0
    for b, Xb in enumerate(_fb5(X, fs)):
        f = f + w[b]*np.array(Parallel(n_jobs=-1)(delayed(_one)(x, refs) for x in Xb))
    return f
