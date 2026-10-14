"""
zs_fuse.py - fuse saved network logits with saved CCA features (official CCA-Net decision rule). Seconds, no retraining.
Needs per fold: results/logits_<tag>_sXX_<0p5|1p0>.npz (logits (240,40), y)  and  results/ccafeats_sXX_*.npz.
Prints mean acc/ITR for: net alone | CCA only (f2+f3) | net+(f3+f2)/2 | net+(f1+f2+f3)/3.
Usage: python3 zs_fuse.py --tag cnnref --sl 0.5 1.0 [--vs gnnv2c_g2]    (--vs = paired Wilcoxon on fused acc)
"""
import argparse
import numpy as np
from scipy.stats import wilcoxon
import zs_config as C

def mm(s): lo = s.min(1, keepdims=True); return (s - lo) / np.where(np.ptp(s, 1, keepdims=True) == 0, 1, np.ptp(s, 1, keepdims=True))
def itr(a, T):
    a = np.clip(a, 1e-9, 1 - 1e-9); return (np.log2(40) + a*np.log2(a) + (1-a)*np.log2((1-a)/39)) * 60 / T

def fused_acc(tag, sl, sids):
    k = str(sl).replace('.', 'p'); out = {n: [] for n in ("net", "cca", "fuse4", "fuse_full")}; got = []
    for s in sids:
        lp, fp = C.RESULTS_DIR / f"logits_{tag}_s{s:02d}_{k}.npz", C.RESULTS_DIR / f"ccafeats_s{s:02d}_{k}.npz"
        if not (lp.exists() and fp.exists()): continue
        L, F = np.load(lp), np.load(fp); assert (L["y"] == F["y"]).all(), f"S{s}: label order differs (logits vs features)"
        n, f1, f2, f3, y = mm(L["logits"]), mm(F["f1"]), mm(F["f2"]), mm(F["f3"]), F["y"]
        A = lambda z: float((z.argmax(1) == y).mean())
        out["net"].append(A(n)); out["cca"].append(A(f2 + f3)); out["fuse4"].append(A(n + (f3 + f2) / 2)); out["fuse_full"].append(A(n + (f1 + f2 + f3) / 3)); got.append(s)
    return {k: np.array(v) for k, v in out.items()}, got

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--vs"); ap.add_argument("--sl", nargs="+", type=float, default=[0.5, 1.0])
    a = ap.parse_args(); sids = range(1, 36)
    for sl in a.sl:
        r, got = fused_acc(a.tag, sl, sids); T = sl + 0.5
        if not got: print(f"sl={sl}: no files for tag {a.tag}"); continue
        print(f"\n{a.tag}  sl={sl}  n={len(got)}")
        for k, nm in [("net", "network alone"), ("cca", "CCA only (f2+f3)"), ("fuse4", "net+(f3+f2)/2"), ("fuse_full", "net+(f1+f2+f3)/3")]:
            print(f"  {nm:20s} acc={r[k].mean():.4f} std={r[k].std():.4f}  ITR={itr(r[k], T).mean():.2f}")
        if a.vs:
            r2, g2 = fused_acc(a.vs, sl, sids); c = sorted(set(got) & set(g2)); i1, i2 = [got.index(s) for s in c], [g2.index(s) for s in c]
            for k in ("net", "fuse4", "fuse_full"):
                d = r2[k][i2] - r[k][i1]; p = wilcoxon(d).pvalue if np.any(d) else 1.0
                print(f"  {a.vs} - {a.tag} [{k}]: mean diff {d.mean():+.4f}  wins/losses {int((d>0).sum())}/{int((d<0).sum())}  Wilcoxon p={p:.3f}  (n={len(c)})")
