import numpy as np, torch, glob, sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zs_config as C
from zs_gnn_v2c import _sym_row_norm
tag = sys.argv[1]; win = sys.argv[2] if len(sys.argv) > 2 else "1p0"
names = ["Pz","PO5","PO3","POz","PO4","PO6","O1","Oz","O2"]
rows = [l.split() for l in open("benchmark/64-channels.loc") if len(l.split()) >= 4]
lab = {r[3].upper(): (float(r[1]), float(r[2])) for r in rows}
a = np.array([lab[n.upper()] for n in names])
if np.abs(a[:, 1]).max() <= 1.5:
    th = np.deg2rad(a[:, 0]); xy = np.c_[a[:, 1]*np.sin(th), a[:, 1]*np.cos(th)]; print("polar .loc")
else:
    xy = a; print("cartesian .loc")
print(dict(zip(names, xy.round(3).tolist())))
fs = sorted(glob.glob(str(C.RESULTS_DIR / f"adj_{tag}_s*_{win}.npz"))); print(len(fs), "subjects")
keys = list(np.load(fs[0]).keys()); nb = np.load(fs[0])[keys[0]].reshape(-1, 9, 9).shape[0]
iu = np.triu_indices(9, 1)
fig, ax = plt.subplots(len(keys), nb, figsize=(6*nb, 5.5*len(keys)), squeeze=False)
for r, k in enumerate(keys):
    S = np.stack([torch.stack([_sym_row_norm(torch.tensor(m)) for m in np.load(f)[k].reshape(-1, 9, 9)]).numpy() for f in fs])
    M = S.mean(0)
    for b in range(nb):
        W = (M[b] + M[b].T) / 2; w = W[iu]; lo, hi = w.min(), w.max()
        deg = W.sum(1) - np.diag(W); axx = ax[r, b]
        for t in np.argsort(w)[::-1][:12]:
            i, j = iu[0][t], iu[1][t]
            axx.plot(*xy[[i, j]].T, color="gray", lw=0.5 + 5*(w[t]-lo)/(hi-lo+1e-9), zorder=1)
        sc = axx.scatter(xy[:, 0], xy[:, 1], c=deg, s=900, cmap="viridis", zorder=2)
        for n, (x, y) in zip(names, xy): axx.text(x, y, n, ha="center", va="center", fontsize=7, color="w", zorder=3)
        plt.colorbar(sc, ax=axx, label="degree", fraction=0.046)
        axx.set_aspect("equal"); axx.margins(0.25)
        axx.set_title(f"{k} band {b+1}\nedge weights {lo:.3f}-{hi:.3f} (uniform = 0.111)", fontsize=9)
plt.tight_layout(); plt.savefig(f"learned_head_{tag}.png", dpi=200)
