"""zs_fuse_ens.py -- fuse the AVERAGE of several networks (seeds / model types) with the cached CCA features.
Each network's logits are min-max normalised per trial, averaged, re-normalised, then fused exactly as zs_fuse.py (fuse4, full). No tuning.
Usage: python3 zs_fuse_ens.py --nets gnnv3g_dense19 gnnv3k_s42d19 gnnv3k_s43d19 [--sl 0.5 1.0] [--ref cnnref]   (--ref: paired Wilcoxon on fuse4/full)"""
import argparse
import numpy as np
from scipy.stats import wilcoxon
import zs_config as C
from zs_fuse import mm, itr, fused_acc

def ens_acc(tags, sl, sids=range(1, 36)):
    k = str(sl).replace(".", "p"); out = {n: [] for n in ("net", "cca", "fuse4", "fuse_full")}; got = []
    for s in sids:
        fp = C.RESULTS_DIR / f"ccafeats_s{s:02d}_{k}.npz"; lps = [C.RESULTS_DIR / f"logits_{t}_s{s:02d}_{k}.npz" for t in tags]
        if not (fp.exists() and all(p.exists() for p in lps)): continue
        F, Ls = np.load(fp), [np.load(p) for p in lps]; y = F["y"]; assert all((L["y"] == y).all() for L in Ls), f"S{s}: label order differs"
        n = mm(np.mean([mm(L["logits"]) for L in Ls], 0)); f1, f2, f3 = mm(F["f1"]), mm(F["f2"]), mm(F["f3"]); A = lambda z: float((z.argmax(1) == y).mean())
        out["net"].append(A(n)); out["cca"].append(A(f2 + f3)); out["fuse4"].append(A(n + (f3 + f2) / 2)); out["fuse_full"].append(A(n + (f1 + f2 + f3) / 3)); got.append(s)
    return {k: np.array(v) for k, v in out.items()}, got

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--nets", nargs="+", required=True); ap.add_argument("--ref"); ap.add_argument("--sl", nargs="+", type=float, default=[0.5, 1.0])
    a = ap.parse_args()
    for sl in a.sl:
        r, got = ens_acc(a.nets, sl); T = sl + 0.5
        if not got: print(f"sl={sl}: missing logits/features for {a.nets}"); continue
        print(f"\nENS[{' + '.join(a.nets)}]  sl={sl}  n={len(got)}")
        for k, nm in [("net", "network ensemble"), ("cca", "CCA only (f2+f3)"), ("fuse4", "ens+(f3+f2)/2"), ("fuse_full", "ens+(f1+f2+f3)/3")]:
            print(f"  {nm:20s} acc={r[k].mean():.4f} se={r[k].std() / np.sqrt(len(got)):.4f}  ITR={itr(r[k], T).mean():.2f}")
        if a.ref:
            r2, g2 = fused_acc(a.ref, sl, got)
            c = sorted(set(got) & set(g2)); i1, i2 = [got.index(s) for s in c], [g2.index(s) for s in c]
            for k in ("net", "fuse4", "fuse_full"):
                d = r[k][i1] - r2[k][i2]; p = wilcoxon(d).pvalue if np.any(d) else 1.0
                print(f"  ENS - {a.ref} [{k}]: mean diff {d.mean():+.4f}  wins/losses {int((d>0).sum())}/{int((d<0).sum())}  Wilcoxon p={p:.3f}  (n={len(c)})")
