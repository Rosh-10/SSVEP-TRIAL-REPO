"""Compares zs_cca_net.py preprocessing/CCA pieces to the official repo (main.py logic). Set OFF to the official repo path."""
import sys, types, numpy as np
import scipy.signal, scipy.stats, scipy.linalg, sklearn.preprocessing, sklearn.cross_decomposition, joblib
OFF = '/home/claude/off/Zero-shot-SSVEP-classification-main'
# ---- stubs so this runs without torch / your data ----
def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
_Mod = type('Module', (), {})
tn = mod('torch.nn', Module=_Mod); tn.functional = mod('torch.nn.functional')
tu = mod('torch.utils'); tu.data = mod('torch.utils.data', DataLoader=None, TensorDataset=None)
tt = mod('torch', Tensor=type('Tensor', (), {})); tt.nn = tn; tt.utils = tu
mod('zs_config', NH=4, N_CLS=40, CHANNELS=list(range(9)), FS=250, SEED=41, SUBJECTS=[], RESULTS_DIR=None)
mod('zs_data'); mod('zs_gnn_train', set_seed=None, compute_itr=None, DEVICE='cpu')
freqs = np.arange(8, 15.9, 0.2)[:40]
ph = np.array([0,.5,1,1.5,0,.5,1,1.5,.5,1,1.5,0,.5,1,1.5,0,1,1.5,0,.5,1,1.5,0,.5,1.5,0,.5,1,1.5,0,.5,1,0,.5,1,1.5,0,.5,1,1.5])*np.pi
T = 125
cfg = types.SimpleNamespace(rfs=250, Nh=4, Nm=3, num_class=40, C=9, T=T, list_freqs=freqs, list_phase=ph, dropout=.95, sample_length=.5)
mod('config', config=cfg); sys.path.insert(0, OFF); sys.path.insert(0, '.')
import function as OF, CCA as OC
import zs_cca_net as Z
rng = np.random.RandomState(0); nsub = 4
# ---- official main.py logic on synthetic data_raw (label, block, C, T) ----
raws = []
for s in range(nsub):
    d = rng.randn(40, 6, 9, T)
    for c in range(40): d[c] += 0.3*np.sin(2*np.pi*freqs[c]*np.arange(T)/250+ph[c])[None, None, :]*rng.randn(1, 9, 1)
    raws.append(d)
tr_lst = []
for d in raws:
    L = np.zeros((40, 6, 8, T))
    for cla in range(40):
        P = OF.GLST(d[cla], 250, 4, freqs[cla], ph[cla])
        for b in range(6): L[cla, b] = P @ d[cla, b]
    tr_lst.append(L)
raw_all = np.concatenate(raws, 1); lst_all = np.concatenate(tr_lst, 1)       # label*block*...
N = 40 * 6 * nsub
tr_raw = np.zeros((N, 9, T)); tr_l = np.zeros((N, 8, T)); lab = np.zeros(N)
for j in range(40):
    k = 6*nsub; tr_raw[k*j:k*j+k] = raw_all[j]; tr_l[k*j:k*j+k] = lst_all[j]; lab[k*j:k*j+k] = j
o_lst, o_P, o_mean = OF.get_LST_transform(tr_raw, tr_l, lab, list_freqs=freqs, fs=250, num_harms=4)
# ---- my pipeline: per-subject, SHUFFLED trial order to prove label-based grouping ----
mine_l, Xs, ys = [], [], []
for d in raws:
    X = d.reshape(240, 9, T); y = np.repeat(np.arange(40), 6); p = rng.permutation(240); X, y = X[p], y[p]
    mine_l.append(Z.per_subject_lst(X, y, freqs, ph)); Xs.append(X); ys.append(y)
X_src = np.concatenate(Xs); y_src = np.concatenate(ys); L_src = np.concatenate(mine_l)
P, sst = Z.fit_gmlst(X_src, L_src, y_src, freqs, ph)
print('P_lst   max|diff|', np.abs(P - o_P).max()); print('SST     max|diff|', np.abs(sst - o_mean).max())
Xt = tr_raw[::37][:6]
o_t = np.stack([np.einsum('chk,kt->cht', o_P, x) for x in Xt]); m_t = Z.apply_lst_all(Xt, P)
print('apply   max|diff|', np.abs(o_t - m_t).max())
# FB_stand vs sklearn scale
from sklearn import preprocessing
a = m_t[:2].copy(); ref = np.stack([[preprocessing.scale(a[i, c], axis=1) for c in range(1)] for i in range(2)])
b = Z.FB_stand(a.copy()); print('FB_stand max|diff|', np.abs(b[:, :1] - ref).max() if False else
      np.abs(b[0, 0] - preprocessing.scale(a[0, 0], axis=1)).max())
# FBTtCCA(SST) vs official
est_o = OC.FBTtCCA(filterbank=Z._FB, n_components=1, filterweights=Z.FB_WEIGHTS, n_jobs=1).fit(tr_raw, lab, Yf=o_mean)
fo = est_o.predict(tr_raw[::41][:6])[1]; fm = Z.FBTtCCA(1).fit(X_src, y_src, sst).predict(tr_raw[::41][:6])[1]
print('FBTtCCA max|diff|', np.abs(fo - fm).max(), ' argmax agree', (fo.argmax(1) == fm.argmax(1)).mean())
