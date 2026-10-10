"""
zs_cca_net.py - CCA-Net (Deng et al., IEEE TIM 2025) replication, faithful to the official repo
(Sungden/Zero-shot-SSVEP-classification: main.py, model.py, function.py, CCA.py, FBCCA.py).

Pieces verified numerically against the official code (see check_official.py / check_net.py):
  GLST, per-subject LST, get_LST_transform -> data_mean (SST), FBTtCCA, official filter_bank,
  cca_reference, FBCCA (sklearn NIPALS vs exact kernel differ by ~1e-3).
Outputs per fold (all from the same trained CNN + CCA features, mirrors official p_labels_new_*):
  acc_cnn  CNN alone | acc_tt  Modified FBtt-CCA | acc_cca  FBCCA(SST) | acc_mod  tt+cca
  acc      CNN + (FBCCA_SST + FBtt_SST)/2           (official p_labels_new_4)
  acc_full CNN + (FBCCA_sincos + FBtt_SST + FBCCA_SST)/3  (official p_labels_new)

Usage (from ZERO_SHOT/):
  python3 zs_cca_net.py --smoke
  python3 zs_cca_net.py --sl 0.5 --subjects 1 --device cuda
  python3 zs_cca_net.py --sl 0.5 1.0 --resume --device cuda 2>&1 | tee -a results_cca_net_off_all.txt
"""
import argparse, json, sys, time, warnings
from functools import partial
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import pinv, eigh, qr
from scipy.signal import sosfiltfilt, cheby1
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

import zs_config as C
import zs_data as D
from zs_gnn_train import set_seed, compute_itr, DEVICE
from zs_cca_official import official_filter_bank, cca_reference, fbcca_official

warnings.filterwarnings("ignore")

METHOD_TAG = "cca_net_off"          # new tag: old buggy cca_net_*.json files are NOT reused by --resume
EPOCHS, BATCH_SIZE, LR = 100, 128, 2e-4
NH, N_CLS, N_CH, FS = C.NH, C.N_CLS, len(C.CHANNELS), C.FS     # 4, 40, 9, 250
FB_WP = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]      # [REPO main.py] FBTtCCA bank
FB_WEIGHTS = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(5)])

# ---------------------------------------------------------------- 1. LST / GMLST / SST
def GLST(data, fs, Nh, f, phase):
    """[REPO function.GLST] data:(block,C,T) or (C,T) -> P:(2Nh,C)"""
    if data.ndim != 3: data = data[None]
    X_a = data.mean(0); t = np.arange(X_a.shape[1]) / fs
    Yf = np.zeros((X_a.shape[1], 2 * Nh))
    for h in range(Nh):
        Yf[:, 2*h]   = np.sin(2*np.pi*f*(h+1)*t + (h+1)*phase)
        Yf[:, 2*h+1] = np.cos(2*np.pi*f*(h+1)*t + (h+1)*phase)
    return Yf.T @ X_a.T @ pinv(X_a @ X_a.T)

def per_subject_lst(X, y, freqs, phases, nh=NH, fs=FS):
    """[REPO main.py inner loop] one subject: X:(N,C,T), y:(N,) any order -> (N,2Nh,T). P fit on that class's blocks."""
    out = np.zeros((len(X), 2 * nh, X.shape[2]))
    for c in range(N_CLS):
        idx = np.where(y == c)[0]
        P = GLST(X[idx], fs, nh, freqs[c], phases[c])
        out[idx] = P @ X[idx]                                   # (2Nh,C)@(n,C,T) -> (n,2Nh,T)
    return out

def fit_gmlst(X, X_lst, y, freqs, phases, nh=NH, fs=FS):
    """[REPO function.get_LST_transform, P_lst and data_mean parts]
    X:(N,C,T) raw, X_lst:(N,2Nh,T) per-subject LST, y:(N,) -> P_lst:(40,2Nh,C), data_mean(SST):(40,2Nh,T)"""
    P_lst = np.zeros((N_CLS, 2 * nh, X.shape[1])); sst = np.zeros((N_CLS, 2 * nh, X.shape[2]))
    for c in range(N_CLS):
        m = y == c
        P_lst[c] = GLST(X[m].mean(0), fs, nh, freqs[c], phases[c])
        sst[c] = X_lst[m].mean(0)
    return P_lst, sst

def apply_lst_all(X, P_lst, chunk=512):
    """out[n,c] = P_lst[c] @ X[n]  ->  (N,40,2Nh,T) float32 (chunked, memory-safe)"""
    out = np.empty((len(X), N_CLS, P_lst.shape[1], X.shape[2]), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i+chunk] = np.einsum('chk,nkt->ncht', P_lst, X[i:i+chunk]).astype(np.float32)
    return out

def FB_stand(x, chunk=512):
    """[REPO function.FB_stand] z-score over time for every (.., row). In place, float32."""
    for i in range(0, len(x), chunk):
        c = x[i:i+chunk]; s = c.std(-1, keepdims=True)
        x[i:i+chunk] = (c - c.mean(-1, keepdims=True)) / np.where(s == 0, 1.0, s)
    return x

# ---------------------------------------------------------------- 2. CSDuDoFN [REPO model.py Siamese]
class CSDuDoFN(nn.Module):
    """Returns x-branch logits only (as repo). x_cca/merge branches are dead code (no gradient) but kept so
    dropout RNG consumption matches the repo."""
    def __init__(self, T, n_bands=3, n_ch=N_CH, n_cls=N_CLS, nh=NH):
        super().__init__()
        self.conv1   = nn.Conv2d(n_bands, 1, (1, 1))
        self.conv1_1 = nn.Conv2d(n_cls,   1, (1, 1))
        self.conv2   = nn.Conv2d(1, 120, (n_ch, 1), bias=True)
        self.conv2_1 = nn.Conv2d(1, 120, (2*nh, 1), bias=True)
        self.drop1   = nn.Dropout2d(0.1)
        self.conv3   = nn.Conv2d(120, 120, (1, 2), stride=(1, 2), bias=True)
        self.conv3_1 = nn.Conv2d(120, 120, (1, 2), stride=(1, 2), bias=True)
        self.conv3_2 = nn.Conv2d(120, 120, (1, 2), stride=(1, 2), bias=True)
        self.drop2   = nn.Dropout2d(0.1)
        self.relu    = nn.ReLU()
        self.conv4   = nn.Conv2d(120, 120, (1, 10), padding='same', bias=True)
        self.conv4_1 = nn.Conv2d(120, 120, (1, 10), padding='same', bias=True)
        self.conv4_2 = nn.Conv2d(120, 120, (1, 10), padding='same', bias=True)
        self.drop3   = nn.Dropout2d(0.95)
        self.flatten = nn.Flatten()
        self.fc   = nn.Linear(120 * (T // 2), n_cls)
        self.fc_1 = nn.Linear(120 * (T // 2), n_cls)
        self.fc_2 = nn.Linear(120 * (T // 2), n_cls)

    def forward(self, x, x_cca):
        x1 = self.conv1(x); x2 = self.drop1(self.conv2(x1))
        x3 = self.drop2(self.conv3(x2)); x3_ = x3; x3 = self.relu(x3)
        x = self.fc(self.flatten(self.drop3(self.conv4(x3))))
        xc = self.drop1(self.conv2_1(self.conv1_1(x_cca)))
        xm = self.conv3_2(xc + x2)
        xc = self.drop2(self.conv3_1(xc))
        xm = self.drop3(self.conv4_2(xc + x3_ + xm)); _ = self.fc_2(self.flatten(xm))
        xc = self.drop3(self.conv4_1(self.relu(xc))); _ = self.fc_1(self.flatten(xc))
        return x

# ---------------------------------------------------------------- 3. Modified FBtt-CCA [REPO CCA.py] (verified: diff 0.0)
def _ged_wong(Z, P=None, n_components=1):
    A = Z if P is None else P.T @ Z
    A = A.T @ A; B = Z.T @ Z
    D_, W = eigh(A, B); ind = np.argsort(D_)[::-1]
    return D_[ind[:n_components]], W[:, ind[:n_components]]

def _scca_kernel(X, Yf):
    Q, R = qr(Yf.T, mode='economic')
    _, U = _ged_wong(X.T, Q @ Q.T, n_components=min(X.shape[0], Yf.shape[0]))
    return U, pinv(R) @ Q.T @ X.T @ U

def _ttcca_feature(X, templates, Yf, Us, n_components=1):
    rhos = []
    for Xk, Y, U2 in zip(templates, Yf, Us):
        U1, V1 = _scca_kernel(X, Y); r = []
        a = U1[:, :n_components].T @ X
        r.append(pearsonr(a.ravel(), (V1[:, :n_components].T @ Y).ravel())[0])    # rho1
        r.append(pearsonr(a.ravel(), (U1[:, :n_components].T @ Xk).ravel())[0])   # rho3
        r.append(pearsonr((U2[:, :n_components].T @ X).ravel(), (U2[:, :n_components].T @ Xk).ravel())[0])  # rho2
        r = np.array(r); rhos.append(np.sum(np.sign(r) * r ** 2))
    return rhos

class _TtCCA:
    def __init__(s, n_components=1): s.n_components = n_components
    def fit(s, X, y, Yf):
        X = X - X.mean(-1, keepdims=True); cl = np.unique(y).astype(int)
        s.templates_ = np.stack([X[y == c].mean(0) for c in cl])
        s.Yf_ = Yf - Yf.mean(-1, keepdims=True)
        s.Us_ = np.stack([_scca_kernel(s.templates_[i], s.Yf_[i])[0] for i in range(len(cl))]); return s
    def transform(s, X):
        X = X - X.mean(-1, keepdims=True)
        fn = partial(_ttcca_feature, templates=s.templates_, Yf=s.Yf_, Us=s.Us_, n_components=s.n_components)
        return np.array(Parallel(n_jobs=-1)(delayed(fn)(x) for x in X))

_FB = [cheby1(15, 0.5, wp, btype='bandpass', output='sos', fs=FS) for wp in FB_WP]

class FBTtCCA:
    def __init__(s, n_components=1): s.n_components = n_components
    def fit(s, X, y, Yf):
        s.est_ = [_TtCCA(s.n_components).fit(sosfiltfilt(sos, X, axis=-1), y, Yf) for sos in _FB]; return s
    def predict(s, X):
        f = np.sum([e.transform(sosfiltfilt(sos, X, axis=-1)) * w for sos, e, w in zip(_FB, s.est_, FB_WEIGHTS)], 0)
        return f.argmax(1), f

def _minmax(s):
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / np.where(hi - lo == 0, 1.0, hi - lo)

# ---------------------------------------------------------------- 4. One LOSO fold
def run_fold_cca_net(sid, sl, device=DEVICE):
    set_seed(C.SEED); T = int(sl * C.FS); t0 = time.time()
    freqs, _, phases = D.load_freq_phase()
    fold = D.make_fold(sid); src_ids = fold["train"] + fold["val"]
    print(f"  S{sid:02d} sl={sl} CCA-Net  {len(src_ids)} sources  device={device}", flush=True)

    parts = [D.get_trials(s, sl) for s in src_ids]                       # raw (N,9,T), y, blk
    X_src = np.concatenate([p[0] for p in parts]).astype(np.float64)
    y_src = np.concatenate([p[1] for p in parts]).astype(int)
    X_te, y_te, _ = D.get_trials(sid, sl); X_te = X_te.astype(np.float64); y_te = np.asarray(y_te).astype(int)

    lst_ps = np.concatenate([per_subject_lst(p[0].astype(np.float64), np.asarray(p[1]).astype(int), freqs, phases)
                             for p in parts])                            # per-subject LST (main.py)
    P_lst, data_mean = fit_gmlst(X_src, lst_ps, y_src, freqs, phases)    # GMLST + SST
    del lst_ps
    X_src_lst = FB_stand(apply_lst_all(X_src, P_lst)); X_te_lst = FB_stand(apply_lst_all(X_te, P_lst))
    X_src_fb = FB_stand(official_filter_bank(X_src).astype(np.float32))
    X_te_fb  = FB_stand(official_filter_bank(X_te).astype(np.float32))
    print(f"  S{sid:02d} preprocessing done ({time.time()-t0:.1f}s)", flush=True)

    model = CSDuDoFN(T).to(device); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = DataLoader(TensorDataset(torch.from_numpy(X_src_fb), torch.from_numpy(X_src_lst),
                                  torch.from_numpy(y_src).long()),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    t1 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
        for xb, xlb, yb in dl:
            xb, xlb, yb = xb.to(device), xlb.to(device), yb.to(device)
            opt.zero_grad(); loss = F.cross_entropy(model(xb, xlb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if ep % 25 == 0 or ep == 1: print(f"    ep {ep:3d}  loss={tot/n:.4f}", flush=True)
    print(f"  S{sid:02d} CNN done ({time.time()-t1:.1f}s)", flush=True)
    del dl, X_src_fb, X_src_lst

    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X_te_fb), 64):
            out.append(model(torch.from_numpy(X_te_fb[i:i+64]).to(device),
                             torch.from_numpy(X_te_lst[i:i+64]).to(device)).cpu().numpy())
    prediction = np.concatenate(out)

    t2 = time.time()
    f2 = FBTtCCA(1).fit(X_src, y_src, data_mean).predict(X_te)[1]        # Modified FBtt-CCA (SST)
    f3 = fbcca_official(X_te, data_mean)                                 # FBCCA with SST
    f1 = fbcca_official(X_te, cca_reference(freqs, T))                   # FBCCA, sin/cos
    print(f"  S{sid:02d} CCA branch done ({time.time()-t2:.1f}s)", flush=True)

    pr, f1, f2, f3 = _minmax(prediction), _minmax(f1), _minmax(f2), _minmax(f3)
    A = lambda s: float((s.argmax(1) == y_te).mean())
    acc, acc_full = A(pr + (f3 + f2) / 2), A(pr + (f1 + f2 + f3) / 3)
    res = dict(subject=sid, method=METHOD_TAG, sl=sl, acc=acc, acc_full=acc_full,
               acc_cnn=A(pr), acc_tt=A(f2), acc_cca=A(f3), acc_mod=A(f2 + f3),
               itr=compute_itr(acc, T), elapsed=round(time.time() - t0, 1), seed=C.SEED,
               preds=(pr + (f3 + f2) / 2).argmax(1).tolist(), labels=y_te.tolist())
    print(f"  S{sid:02d} sl={sl} acc={acc:.4f} full={acc_full:.4f} | cnn={res['acc_cnn']:.4f} tt={res['acc_tt']:.4f} "
          f"cca={res['acc_cca']:.4f} mod={res['acc_mod']:.4f}  ({res['elapsed']}s)", flush=True)
    return res

# ---------------------------------------------------------------- 5. checkpoints / summary / smoke / main
def ckpt_path(sid, sl):
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"{METHOD_TAG}_s{sid:02d}_{str(sl).replace('.','p')}.json"

def summarise(results, sl):
    rs = [r for r in results if r["sl"] == sl]
    if not rs: return
    print(f"\nCCA-Net  sl={sl}  n={len(rs)}   (paper Benchmark: CCA-Net 54.27/84.02, CSDuDoFN 42.82/70.12, ModFBtt 43.96/80.77, FBCCA-SST 25.17/67.04)")
    for k, nm in [("acc", "CNN+(f3+f2)/2 [p_labels_new_4]"), ("acc_full", "CNN+(f1+f2+f3)/3 [p_labels_new]"),
                  ("acc_cnn", "CNN alone"), ("acc_tt", "Modified FBtt-CCA"), ("acc_cca", "FBCCA (SST)"), ("acc_mod", "tt + cca")]:
        a = np.array([r[k] for r in rs]); print(f"  {nm:34s} mean={a.mean():.4f} std={a.std():.4f}")
    print(f"  ITR(acc) mean={np.mean([r['itr'] for r in rs]):.2f}")
    print(f"  per-subject acc: {[round(r['acc'], 3) for r in rs]}")

def smoke_test():
    print("\n=== CCA-Net smoke test ===")
    exp = {125: 471884, 250: 774284}                                     # hand count of x-branch params
    for T in (125, 250):
        m = CSDuDoFN(T).to(DEVICE); x = torch.randn(4, 3, N_CH, T, device=DEVICE); xc = torch.randn(4, 40, 2*NH, T, device=DEVICE)
        assert m(x, xc).shape == (4, 40)
        m.train(); F.cross_entropy(m(x, xc), torch.randint(0, 40, (4,), device=DEVICE)).backward()
        xb = [(n, p) for n, p in m.named_parameters() if not any(s in n for s in ('_1', '_2'))]
        assert all(p.grad is not None for _, p in xb), "x-branch param without grad"
        n = sum(p.numel() for _, p in xb); print(f"  T={T}: x-branch params={n:,} (hand count {exp[T]:,})  total={sum(p.numel() for p in m.parameters()):,}")
        assert n == exp[T], "param count differs from hand count"
    rng = np.random.RandomState(0); X = rng.randn(240, N_CH, 125); y = np.repeat(np.arange(40), 6)
    fr = np.linspace(8, 15.8, 40); ph = np.zeros(40)
    L = per_subject_lst(X, y, fr, ph); P, sst = fit_gmlst(X, L, y, fr, ph)
    assert L.shape == (240, 8, 125) and P.shape == (40, 8, N_CH) and sst.shape == (40, 8, 125)
    assert apply_lst_all(X, P).shape == (240, 40, 8, 125)
    print("  LST/GMLST/SST shapes OK\nSmoke test PASSED.\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", nargs="+", type=float, default=[0.5]); ap.add_argument("--subjects", nargs="+", type=int, default=C.SUBJECTS)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    if a.smoke: smoke_test(); sys.exit(0)
    print(f"\nDevice: {a.device}  Subjects: {a.subjects}  Windows: {a.sl}\nEpochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}  Seed: {C.SEED}\n")
    allr = []
    for sl in a.sl:
        print(f"\n=== sl={sl} ===")
        for sid in a.subjects:
            p = ckpt_path(sid, sl)
            if a.resume and p.exists(): print(f"  S{sid:02d} sl={sl} [skipping]"); allr.append(json.loads(p.read_text())); continue
            r = run_fold_cca_net(sid, sl, a.device); p.write_text(json.dumps(r, indent=1)); allr.append(r)
    print("\n=== SUMMARY ===")
    for sl in a.sl: summarise([r for r in allr if r["sl"] == sl], sl)
